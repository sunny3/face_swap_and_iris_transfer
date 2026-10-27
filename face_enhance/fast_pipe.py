"""Fast single-process post-swap face pipeline (alaska GPU 3, pytorch container, cwd=/w/CodeFormer).

Reproduces codeformer_w1.0_x2_skindiff (+ avatar iris) with the official model code; only orchestration and
precision change:
  * one RetinaFace detection (facelib, resize=640) reused by every stage
  * CodeFormer w=1.0 on the 512 aligned face (fp16 autocast or fp32)
  * RealESRGAN_x2plus fp16 UNTILED on the 512 restored face (= --face_upsample) and on the full frame (bg)
    or GPU bicubic x2 for the bg (variant *_bgbicubic)
  * FaceRestoreHelper.paste_faces_to_input_image re-implemented on GPU inside the face quad's ROI only
    (grid_sample inverse affine, replicate-pad erosion, separable Gaussian, ParseNet mask, same constants)
  * SkinDiffDetail-Lite fp16 on the face bbox (+40%), same feathered ellipse
  * avatar iris transfer (iris_variants.avatar_one logic) with MediaPipe on the face crop only and avatar
    landmarks cached per bot (outside timing)
usage: python fast_pipe.py <variant> [save=1] [opts: cf32 det16 cl]
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.detection import init_detection_model
from facelib.utils.face_restoration_helper import FaceRestoreHelper

IN, OUT, LAT, MOD, AV = '/b/in', '/b/out', '/b/lat', '/b/models_fast', '/b/avatars'
VARIANT = sys.argv[1]
SAVE = (sys.argv[2] == '1') if len(sys.argv) > 2 else True
OPTS = set(sys.argv[3:])
IRIS_ON = 'iris' in VARIANT
BG_BICUBIC = 'bgbicubic' in VARIANT
CF_HALF = 'cf32' not in OPTS
DET_HALF = 'det16' in OPTS
CL = 'cl' in OPTS
dev = torch.device('cuda')
FINE = os.environ.get('FINE') == '1'

# orchestration-only: RetinaFace rebuilds its anchor grid in a Python loop on every call; cache it per input size
import facelib.detection.retinaface.retinaface as _RF  # noqa: E402
_PB = _RF.PriorBox


class _CachedPriorBox(_PB):
    _c = {}

    def forward(self):
        k = tuple(self.image_size)
        if k not in self._c:
            self._c[k] = _PB.forward(self)
        return self._c[k]


if 'nopriorcache' not in sys.argv:
    _RF.PriorBox = _CachedPriorBox
BENCH = 'bench' in OPTS            # fixed-shape models only (detector 640x960, CF/ESRGAN/parse 512, bg 512x768)
SKIN_BENCH = 'skinbench' in OPTS   # optimistic: autotune per skin-crop shape
PARSE_HALF = 'parse16' in OPTS
GRAPH = 'graph' in OPTS            # CUDA graphs for the fixed-shape nets (launch-overhead bound)
FACEFP = 'facefp' in VARIANT or 'facefp' in OPTS   # face ESRGAN at the canvas footprint instead of 512->1024
torch.backends.cudnn.benchmark = BENCH
t0 = time.time()

# ------------------------------------------------------------------ models (official code / weights)
helper = FaceRestoreHelper(2, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png',
                           use_parse=True, device=dev)
if DET_HALF:
    helper.face_detector = init_detection_model('retinaface_resnet50', half=True, device=dev)
parse_net = helper.face_parse
net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                      connect_list=['32', '64', '128', '256']).to(dev)
net.load_state_dict(torch.load('weights/CodeFormer/codeformer.pth')['params_ema']); net.eval()
esr = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
ld = torch.load('weights/realesrgan/RealESRGAN_x2plus.pth', map_location='cpu')
esr.load_state_dict(ld['params_ema'] if 'params_ema' in ld else ld['params'], strict=True)
esr = esr.eval().to(dev).half()
import spandrel  # noqa: E402
skin = spandrel.ModelLoader().load_from_file(f'{MOD}/1x-ITF-SkinDiffDetail-Lite-v1.pth').cuda().eval().half()  # descriptor call (as bench_fast)
print('skin arch', skin.architecture.name, flush=True)
if CL:
    esr = esr.to(memory_format=torch.channels_last); skin.model.to(memory_format=torch.channels_last)

if IRIS_ON:
    from concurrent.futures import ThreadPoolExecutor
    POOL = ThreadPoolExecutor(1)
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    LM = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f'{MOD}/face_landmarker.task'), num_faces=1))


class Graphed:
    """CUDA-graph a pure-tensor fn for one input shape (static in/out buffers, shared pool). Outputs are cloned."""
    pool = None

    def __init__(self, fn, x):
        self.x = x.clone()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                fn(self.x)
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, pool=Graphed.pool):
            self.y = fn(self.x)
        Graphed.pool = self.g.pool()

    def __call__(self, x):
        self.x.copy_(x); self.g.replay()
        return tuple(t.clone() for t in self.y) if isinstance(self.y, tuple) else self.y.clone()


class GraphCache:
    def __init__(self, fn):
        self.fn, self.c = fn, {}

    def __call__(self, x):
        if not GRAPH:
            return self.fn(x)
        k = (tuple(x.shape), x.dtype, x.is_contiguous(memory_format=torch.channels_last))
        if k not in self.c:
            self.c[k] = Graphed(self.fn, x)
        return self.c[k](x)


def _esr_fn(x):
    with torch.no_grad():
        return esr(x)


def _cf_fn(x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=CF_HALF, cache_enabled=False):
        return net(x, w=1.0, adain=True)[0].float()


def _parse_fn(x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=PARSE_HALF, cache_enabled=False):
        return parse_net(x)[0].float()


ESR_G, CF_G, PARSE_G = GraphCache(_esr_fn), GraphCache(_cf_fn), GraphCache(_parse_fn)
if GRAPH:   # capture-safe twin of VectorQuantizer.get_codebook_feat (official code builds the one-hot on CPU, then .to(gpu))
    import basicsr.archs.vqgan_arch as _VQ

    def _gcf(self, indices, shape):
        indices = indices.view(-1, 1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size, device=indices.device, dtype=indices.dtype)
        min_encodings.scatter_(1, indices, 1)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q
    _VQ.VectorQuantizer.get_codebook_feat = _gcf
_det = helper.face_detector
_det_fwd = _det.forward


def _det_fn(x):
    with torch.no_grad():
        return _det_fwd(x)


DET_G = GraphCache(_det_fn)
_det.forward = lambda x: DET_G(x)     # RetinaFace.__detect_faces calls self(inputs)
torch.cuda.synchronize()
cold_load_s = time.time() - t0

# ------------------------------------------------------------------ GPU helpers
def gauss_k(ks, sigma):
    if sigma <= 0:
        sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8          # cv2.getGaussianKernel rule
    x = torch.arange(ks, device=dev, dtype=torch.float32) - (ks - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma)); return g / g.sum()


def gblur(x, ks, sigma=0):
    """cv2.GaussianBlur (BORDER_REFLECT_101) on (1,1,H,W) float."""
    if ks <= 1:
        return x
    g = gauss_k(ks, sigma); p = ks // 2
    mode = 'reflect' if min(x.shape[-2:]) > p else 'replicate'
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode=mode), g.view(1, 1, 1, ks))
    return F.conv2d(F.pad(x, (0, 0, p, p), mode=mode), g.view(1, 1, ks, 1))


def erode(x, k):
    """cv2.erode with a kxk ones kernel (anchor k//2, border = ignore) on (1,1,H,W)."""
    if k <= 1:
        return x
    l, r = k // 2, k - 1 - k // 2
    x = F.pad(x, (l, r, l, r), mode='replicate')
    x = -F.max_pool2d(-x, (1, k), stride=1)
    return -F.max_pool2d(-x, (k, 1), stride=1)


def warp_roi(src, Minv, roi, mode='bilinear'):
    """cv2.warpAffine(src, M, ...)[roi] for (1,C,h,w) src; Minv maps dst->src pixel coords."""
    x0, y0, x1, y1 = roi
    ys, xs = torch.meshgrid(torch.arange(y0, y1, device=dev, dtype=torch.float32),
                            torch.arange(x0, x1, device=dev, dtype=torch.float32), indexing='ij')
    M = torch.tensor(Minv, device=dev, dtype=torch.float32)
    px = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
    py = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
    h, w = src.shape[-2:]
    grid = torch.stack((px * 2 / (w - 1) - 1, py * 2 / (h - 1) - 1), -1)[None]
    return F.grid_sample(src, grid, mode=mode, padding_mode='zeros', align_corners=True)


def esrgan(x01):
    """RealESRGANer.enhance(outscale=2) without tiling: (1,3,H,W) RGB [0,1] -> (1,3,2H,2W) RGB 0..255 rounded."""
    h, w = x01.shape[-2:]
    ph, pw = h % 2, w % 2
    x = F.pad(x01.half(), (0, pw, 0, ph), 'reflect') if (ph or pw) else x01.half()
    if CL:
        x = x.contiguous(memory_format=torch.channels_last)
    y = ESR_G(x)
    y = y[..., :2 * h, :2 * w].float().clamp_(0, 1)
    return (y * 255).round_()


_BM = {}


def blur_mat(n, ks, sigma):
    """n x n matrix of a 1-D cv2 Gaussian with BORDER_REFLECT_101 (so a separable blur = K @ X @ K.T)."""
    key = (n, ks, sigma)
    if key not in _BM:
        g = gauss_k(ks, sigma).double().cpu().numpy(); p = ks // 2
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(-p, p + 1):
                q = i + j
                q = -q if q < 0 else (2 * (n - 1) - q if q > n - 1 else q)
                K[i, q] += g[j + p]
        _BM[key] = torch.from_numpy(K @ K).float().to(dev)          # the helper blurs twice
    return _BM[key]


PARSE_LUT = torch.tensor([0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0],
                         device=dev, dtype=torch.float32)


def parse_mask(face255, f512=None):
    """helper parse branch: restored(1024)->512 linear->ParseNet->colormap->2x Gauss(101,11)->border 0->/255->face size.
    f512: optional 512 face to parse directly (facefp mode)."""
    if f512 is None:
        f512 = F.avg_pool2d(face255, 2).round()
    t = (f512 / 255 - 0.5) / 0.5
    out = PARSE_G(t)
    m = PARSE_LUT[out.argmax(1)][0]                             # (512,512)
    K = blur_mat(512, 101, 11)
    m = (K @ m @ K.T)[None, None]
    m[..., :10, :] = 0; m[..., -10:, :] = 0; m[..., :, :10] = 0; m[..., :, -10:] = 0
    return F.interpolate(m / 255., size=face255.shape[-2:], mode='bilinear', align_corners=False)


def paste(canvas, face255, inv_aff, f512=None):
    """GPU port of paste_faces_to_input_image (face_upsampler path) restricted to the warped face quad ROI.
    canvas: (1,3,H,W) float 0..255 (modified in place). face255: (1,3,1024,1024). inv_aff: 2x3 face->canvas."""
    H, W = canvas.shape[-2:]
    fs = face255.shape[-1]
    corners = np.array([[0, 0, 1], [fs - 1, 0, 1], [0, fs - 1, 1], [fs - 1, fs - 1, 1]], np.float64) @ inv_aff.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 3
    roi = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        return
    Minv = cv2.invertAffineTransform(inv_aff)
    t = fine('p_start', time.perf_counter())
    inv_restored = warp_roi(face255, Minv, roi)
    ones = torch.ones((1, 1, fs, fs), device=dev)
    inv_mask = warp_roi(ones, Minv, roi)
    k = int(2 * 2)
    inv_mask_erosion = erode(inv_mask, k)
    pasted = inv_mask_erosion * inv_restored
    area = float(inv_mask_erosion.sum())
    w_edge = int(area ** 0.5) // 20
    t = fine('p_warp_erode', t)
    soft = gblur(erode(inv_mask_erosion, w_edge * 2), w_edge * 2 + 1)
    t = fine('p_softmask', t)
    pm = parse_mask(face255, f512)
    t = fine('p_parse', t)
    pm = warp_roi(pm, Minv, roi)
    soft = torch.minimum(soft, pm)
    X0, Y0, X1, Y1 = roi
    bg = canvas[..., Y0:Y1, X0:X1]
    canvas[..., Y0:Y1, X0:X1] = soft * pasted + (1 - soft) * bg


def skin_face(img_u8_t, boxes):
    """bench_fast.skin_face on GPU: (1,3,H,W) RGB uint8-valued float canvas -> same, in place copy."""
    H, W = img_u8_t.shape[-2:]
    out = img_u8_t.clone()
    torch.backends.cudnn.benchmark = SKIN_BENCH          # crop shape changes per image: no per-shape autotune
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        bw, bh = x2 - x1, y2 - y1; m = 0.4
        X1, Y1 = int(max(0, x1 - m * bw)), int(max(0, y1 - m * bh))
        X2, Y2 = int(min(W, x2 + m * bw)), int(min(H, y2 + m * bh))
        crop = img_u8_t[..., Y1:Y2, X1:X2]
        x = (crop / 255.).half()
        if CL:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            r = skin(x).float().clamp(0, 1)
        r = (r * 255).round()
        h, w_ = crop.shape[-2:]
        mask = np.zeros((h, w_), np.float32)
        cx, cy = (x1 + x2) / 2 - X1, (y1 + y2) / 2 - Y1
        cv2.ellipse(mask, (int(cx), int(cy)), (int(bw * 0.55), int(bh * 0.65)), 0, 0, 360, 1, -1)
        kk = int(max(bw, bh) * 0.15) | 1
        mask = torch.from_numpy(cv2.GaussianBlur(mask, (kk, kk), 0)).to(dev)[None, None]
        out[..., Y1:Y2, X1:X2] = (r * mask + crop * (1 - mask)).round()
    torch.backends.cudnn.benchmark = BENCH
    return out


# ------------------------------------------------------------------ iris (port of iris_variants.avatar_one)
IRIS = {"l": 468, "r": 473}
LIDS = {"l": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "r": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]}
CORNERS = {"l": (33, 133), "r": (263, 362)}
SIGMA, MIN_COVER, MAX_GAZE = 2.0, 0.85, 0.12


def lm(img):
    h, w = img.shape[:2]
    r = LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    if not r.face_landmarks:
        return None
    return np.array([(p.x * w, p.y * h) for p in r.face_landmarks[0]])


def lm_crop(img, box, side=2.0):
    """MediaPipe on a square crop (side x face size) around a RetinaFace box -> full-frame landmarks."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    s = max(x2 - x1, y2 - y1) * side / 2; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X0, Y0, X1, Y1 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
    p = lm(img[Y0:Y1, X0:X1])
    return None if p is None else p + (X0, Y0)


def eye(pts, k):
    c0 = IRIS[k]
    c = pts[c0]
    r = np.linalg.norm(pts[c0 + 1:c0 + 5] - c, axis=1).mean()
    a, b = pts[CORNERS[k][0]], pts[CORNERS[k][1]]
    if a[0] > b[0]:
        a, b = b, a
    ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    ew = np.linalg.norm(b - a)
    u = (b - a) / ew
    v = np.array([-u[1], u[0]])
    d = c - (a + b) / 2
    gaze = np.array([d @ u, d @ v]) / ew
    return {"c": c, "r": r, "lid": pts[LIDS[k]], "ang": ang, "ew": ew, "gaze": gaze}


def roi_of(e, shape, pad=1.6):
    h, w = shape[:2]
    s = int(np.ceil(e["r"] * pad)) + 4
    x, y = np.round(e["c"]).astype(int)
    return max(0, x - s), max(0, y - s), min(w, x + s + 1), min(h, y + s + 1)


def vis_mask(e, roi, scale=1.08):
    x0, y0, x1, y1 = roi
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(m, tuple(np.round((e["c"] - (x0, y0)) * 8).astype(int)), int(round(e["r"] * scale * 8)), 1, -1, shift=3)
    lid = np.zeros_like(m)
    cv2.fillPoly(lid, [np.round((e["lid"] - (x0, y0)) * 8).astype(np.int32)], 1, shift=3)
    return (m & lid).astype(np.float32)


def lab_lock(orig, edit, region):
    o, e = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) for x in (orig, edit))
    r = region > 0.5
    for c in range(3):
        mo, so, me, se = o[..., c][r].mean(), o[..., c][r].std(), e[..., c][r].mean(), e[..., c][r].std()
        e[..., c] = (e[..., c] - me) * (so / max(se, 1e-3)) + mo
    return cv2.cvtColor(np.clip(e, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def blend(base, src_roi, mask_hard, roi):
    x0, y0, x1, y1 = roi
    m = cv2.GaussianBlur(mask_hard, (0, 0), SIGMA)[..., None] * (cv2.dilate(mask_hard, np.ones((3, 3))) > 0)[..., None]
    o = base[y0:y1, x0:x1]
    locked = lab_lock(o, src_roi, mask_hard)
    base[y0:y1, x0:x1] = (o * (1 - m) + locked * m).round().astype(np.uint8)


AVC = {}   # bot -> (avatar, {eye: eye-dict}) ; per-bot cache, filled outside timing (prod caches per bot)


def avatar_iris(base, pb, bot, st):
    av, ea_all = AVC[bot]
    n = 0
    for k in "lr":
        eb = eye(pb, k)
        roi = roi_of(eb, base.shape)
        x0, y0, x1, y1 = roi
        mb = vis_mask(eb, roi)
        s_ = st.setdefault(k, {})
        if mb.sum() < 10:
            s_["skipped"] = "base eye closed"; continue
        best = None
        for ka in (k, "r" if k == "l" else "l"):
            ea = ea_all[ka]
            dg = float(np.abs(ea["gaze"] - eb["gaze"]).max())
            s = eb["r"] / ea["r"]
            M = cv2.getRotationMatrix2D(tuple(ea["c"]), -(eb["ang"] - ea["ang"]), s)
            M[:, 2] += eb["c"] - ea["c"] - (x0, y0)
            warped = cv2.warpAffine(av, M, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REFLECT)
            ma_w = cv2.warpAffine(ea["mask"], M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            cover = float((mb * (ma_w > 0.5)).sum() / mb.sum())
            cand = {"src": ka, "gaze_d": round(dg, 3), "cover": round(cover, 3), "scale": round(float(s), 3)}
            if cover >= MIN_COVER and dg <= MAX_GAZE:
                best = (cand, warped, mb * (ma_w > 0.5)); break
            s_.setdefault("rejected", []).append(cand)
        if best is None:
            s_["skipped"] = "occluded/gaze"; continue
        s_.update(best[0])
        blend(base, best[1], best[2].astype(np.float32), roi)
        n += 1
    return n


# ------------------------------------------------------------------ pipeline
ST = {}


def fine(k, t):
    if not FINE:
        return t
    torch.cuda.synchronize(); n = time.perf_counter(); ST.setdefault('_' + k, []).append((n - t) * 1000); return n


def tick(k, t):
    torch.cuda.synchronize(); n = time.perf_counter(); ST.setdefault(k, []).append((n - t) * 1000); return n


def process(jpg_bytes, bot, stats):
    t = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    x_in = torch.from_numpy(img).to(dev, non_blocking=True)                 # HWC BGR uint8
    t = tick('decode', t)
    helper.clean_all(); helper.input_img = img; helper.is_gray = False   # read_image() minus its is_gray scan (unused here)
    t = fine('d_read', t)
    helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    t = tick('detect', t)
    H, W = img.shape[:2]
    rgb01 = x_in.permute(2, 0, 1)[None].flip(1).float() / 255.               # (1,3,H,W) RGB
    if BG_BICUBIC:
        canvas = (F.interpolate(rgb01, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1) * 255).round()
    else:
        canvas = esrgan(rgb01)
    t = tick('bg_x2', t)
    nf = len(helper.det_faces)
    if nf:
        helper.align_warp_face()
        faces = np.stack(helper.cropped_faces)                              # N,512,512,3 BGR
        ft = torch.from_numpy(faces).to(dev).permute(0, 3, 1, 2).flip(1).float() / 255.
        ft = (ft - 0.5) / 0.5
        out = CF_G(ft)
        restored = ((out.float().clamp(-1, 1) + 1) / 2 * 255).round()       # tensor2img rounding
        invs = [cv2.invertAffineTransform(A) for A in helper.affine_matrices]   # face512 -> input
        boxes = [np.array(b[:4]) * 2 for b in helper.det_faces]
        bi = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
        fut = None
        if IRIS_ON:   # MediaPipe on the restored aligned 512 face of the main face, on a CPU thread, overlapped with GPU
            f512 = restored[bi].flip(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            fut = POOL.submit(lm, f512)
        if IRIS_ON or FACEFP:
            ST.setdefault("_scale512", []).append(float(np.sqrt(abs(np.linalg.det(invs[bi][:, :2])))) * 2)
        t = tick('codeformer', t)
        faces_up = []
        for i in range(nf):
            if FACEFP:   # x2 at the canvas footprint: 512 face -> (footprint/2) area-resize -> ESRGAN x2
                sc = float(np.sqrt(abs(np.linalg.det(invs[i][:, :2])))) * 2       # face512 px -> canvas px
                S = int(min(512, max(128, round(512 * sc / 2 / 64) * 64)))   # 64-px buckets (7 graphs)
                src = restored[i:i + 1] if S == 512 else F.interpolate(restored[i:i + 1], size=(S, S), mode='area').round()
                faces_up.append(esrgan(src / 255.))
            else:
                faces_up.append(esrgan(restored[i:i + 1] / 255.))
        t = tick('face_x2', t)
        for i in range(nf):
            fsz = faces_up[i].shape[-1]
            inv_aff = invs[i].copy(); inv_aff[:, :2] *= 1024 / fsz; inv_aff[:, 2] *= 2   # face(fsz) -> canvas (helper's math at fsz=1024)
            paste(canvas, faces_up[i], inv_aff, restored[i:i + 1] if FACEFP else None)
        canvas = canvas.floor_()                                             # astype(uint8) truncation
        t = tick('paste', t)
        canvas = skin_face(canvas, boxes)
        t = tick('skin', t)
    res = canvas[0].flip(0).permute(1, 2, 0).clamp_(0, 255).to(torch.uint8).cpu().numpy()  # HWC BGR
    t = tick('download', t)
    if IRIS_ON and nf:
        st = stats.setdefault(bot, {})
        ti = time.perf_counter()
        p512 = fut.result()
        if p512 is not None:     # aligned-face landmarks -> canvas: x2 of the inverse affine
            A2 = invs[bi] * 2
            pb = p512 @ A2[:, :2].T + A2[:, 2]
        else:
            pb = lm_crop(res, boxes[bi]); st["base_lm"] = "2x crop"
        ti = fine('i_mediapipe_wait', ti)
        if pb is None:   # fall back to the input crop x2 (the x2 restore keeps the geometry)
            pi = lm_crop(img, boxes[bi] / 2)
            pb = None if pi is None else pi * 2; st["base_lm"] = "input x2"
        if pb is None:
            st["err"] = "no landmarks"; st["n"] = 0
        elif AVC.get(bot) is None:
            st["err"] = "no face on avatar"; st["n"] = 0
        else:
            st["n"] = avatar_iris(res, pb, bot, st)
            fine('i_transfer', ti)
        t = tick('iris', t)
    return res, nf


imgs = sorted(glob.glob(f'{IN}/*.jpg'))
if IRIS_ON:   # per-bot avatar landmark cache (prod: computed once per bot)
    lm(np.zeros((256, 256, 3), np.uint8))
    for p in imgs:
        bot = os.path.splitext(os.path.basename(p))[0]
        av = cv2.imread(f'{AV}/{bot}.jpg')
        pa = None if av is None else lm(av)
        if pa is None:
            AVC[bot] = None; continue
        ea = {k: eye(pa, k) for k in "lr"}
        for k in "lr":
            ea[k]["mask"] = vis_mask(ea[k], (0, 0, av.shape[1], av.shape[0]))
        AVC[bot] = (av, ea)

data = {os.path.splitext(os.path.basename(p))[0]: open(p, 'rb').read() for p in imgs}
# warmup (excluded): every image once (kernels / allocator for each face size), then reset
for b, d in data.items():
    process(d, b, {})
torch.cuda.synchronize()
ST.clear(); torch.cuda.reset_peak_memory_stats()
per, fails, stats, results = {}, [], {}, {}
REP = int(os.environ.get('REP', '3'))
for rep in range(REP):
    for bot, d in data.items():
        torch.cuda.synchronize(); ts = time.perf_counter()
        res, nf = process(d, bot, stats if rep == 0 else {})
        ms = (time.perf_counter() - ts) * 1000
        per.setdefault(bot, []).append(ms)
        if rep == 0:
            results[bot] = res
            if nf == 0:
                fails.append(f'{bot}:noface')
os.makedirs(f'{OUT}/{VARIANT}', exist_ok=True)
if SAVE:
    for bot, res in results.items():
        cv2.imwrite(f'{OUT}/{VARIANT}/{bot}.png', res)
allv = np.concatenate([np.array(v) for v in per.values()])
perm = {b: round(float(np.median(v)), 1) for b, v in per.items()}
rec = {'method': VARIANT,
       'per_image_ms': perm, 'p50_ms': round(float(np.percentile(allv, 50)), 1),
       'p95_ms': round(float(np.percentile(allv, 95)), 1), 'cold_load_s': round(cold_load_s, 2),
       'gpu': '3090 GPU3', 'vram_peak_mb': None,
       'torch_peak_alloc_mb': round(torch.cuda.max_memory_allocated() / 2**20),
       'torch_peak_reserved_mb': round(torch.cuda.max_memory_reserved() / 2**20),
       'stage_ms_p50': {k: round(float(np.median(x)), 1) for k, x in ST.items()},
       'output_hw': list(next(iter(results.values())).shape[:2]), 'failures': fails,
       'opts': {'cuda_graphs': GRAPH, 'cf_fp16': CF_HALF, 'parse_fp16': PARSE_HALF, 'face_x2_at_footprint': FACEFP, 'det_fp16': DET_HALF, 'channels_last': CL, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'reps': REP},
       'notes': ('warm single process; latency = jpeg bytes decode -> final BGR uint8 in host memory (no disk write); '
                 f'{REP} passes over 26 imgs after a full warm pass, p50/p95 over all {len(allv)} runs, per_image_ms = per-bot median. '
                 'RetinaFace r50 (resize 640) once; CodeFormer w1.0 ' + ('fp16 autocast' if CF_HALF else 'fp32') +
                 '; RealESRGAN_x2plus fp16 untiled on the 512 restored face' +
                 ('; bg = GPU bicubic x2' if BG_BICUBIC else ' and on the full frame (bg)') +
                 '; GPU ROI paste (grid_sample inverse affine, erosion, Gaussian, ParseNet mask = helper constants); '
                 'SkinDiffDetail-Lite fp16 on bbox+40% with feathered ellipse' +
                 ('; avatar iris transfer (iris_variants avatar rules) with MediaPipe on the 2x face crop only, avatar '
                  'landmarks cached per bot' if IRIS_ON else ''))}
if IRIS_ON:
    rec['eyes_replaced'] = {b: s.get('n', 0) for b, s in stats.items()}
    rec['iris_stats'] = {b: {k: v for k, v in s.items()} for b, s in stats.items()}
json.dump(rec, open(f'{LAT}/{VARIANT}.json', 'w'), indent=1, default=float)
print('DONE', VARIANT, 'p50', rec['p50_ms'], 'p95', rec['p95_ms'], rec['stage_ms_p50'], 'fails', fails,
      'eyes', sum(rec.get('eyes_replaced', {}).values()) if IRIS_ON else '-')
