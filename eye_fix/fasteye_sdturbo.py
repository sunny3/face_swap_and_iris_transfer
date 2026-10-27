# Fast eye detailer: RetinaFace (facelib, GPU) -> eyelid mask (MediaPipe on the face crop, CPU | RetinaFace-kps ellipses)
# -> square eye-band crop -> 512 -> sd-turbo img2img (official diffusers example call, params patched) -> paste inside mask.
# Runs in docker on alaska GPU 3; see fasteye_run.sh.  Inference = official sd-turbo img2img example:
#   pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16)
#   pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0)
import os, sys, glob, json, time, math, argparse
os.chdir('/w/CodeFormer')  # facelib weights resolve relative to CodeFormer
import numpy as np, cv2, torch
from diffusers import AutoPipelineForImage2Image, AutoencoderKL
from facelib.detection import init_detection_model

p = argparse.ArgumentParser()
p.add_argument('--strength', type=float, default=0.35)
p.add_argument('--steps', type=int, default=1, help='actual denoising steps executed')
p.add_argument('--mask', default='mp', choices=['mp', 'kps'])
p.add_argument('--res', type=int, default=512)
p.add_argument('--variant', required=True)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

PROMPT = ("close-up of a woman's eyes, natural realistic eyes, perfectly round irises with detailed iris texture, "
          "round dark pupils, natural catchlights, sharp eyelashes")
TURBO = glob.glob('/b/hf_ir/hub/models--stabilityai--sd-turbo/snapshots/*')[0]  # unet/text_encoder/tokenizer/scheduler
VAE = '/b/osd_models/sd21/vae'  # SD2.1 VAE (snapshot above has no vae folder; sd-turbo uses the SD2.1 VAE)
LIDS = {"l": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "r": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]}
GROW, SIG = 1.3, 3
BASE = '/b/out/codeformer_w1.0_x2_skindiff'
OUT = f'/b/out/{a.variant}'
dev = 'cuda'

t0 = time.time()
pipe = AutoPipelineForImage2Image.from_pretrained(TURBO, vae=AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16),
                                                  torch_dtype=torch.float16, safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True)
with torch.no_grad():
    pe, _ = pipe.encode_prompt(PROMPT, dev, 1, False)
pipe.text_encoder = None  # prompt is fixed: embeds cached, encoder freed
torch.cuda.empty_cache()
det = init_detection_model('retinaface_resnet50', half=False, device=dev)
mp_lm = None
if a.mask == 'mp':
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    mp_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='/b/models_mp/face_landmarker.task'), num_faces=1))
n_inf = math.ceil(a.steps / a.strength - 1e-9)  # diffusers img2img runs int(n_inf * strength) steps
while int(n_inf * a.strength) < a.steps:
    n_inf += 1
torch.cuda.synchronize()
cold = time.time() - t0


def detect(img):
    h, w = img.shape[:2]
    s = 640 / max(h, w)
    with torch.no_grad():
        b = det.detect_faces(cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR), 0.97)
    if len(b) == 0:
        return None
    b = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))] / s  # largest face
    return b[:4], b[5:15].reshape(5, 2)


def box_from(cx, cy, span, w, h):
    side = int(max(2.2 * span, 192))
    x0 = int(np.clip(cx - side / 2, 0, max(0, w - side)))
    y0 = int(np.clip(cy - side / 2, 0, max(0, h - side)))
    return x0, y0, min(side, w - x0, h - y0)


def mask_mp(img, bb):
    h, w = img.shape[:2]
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    s = int(1.4 * max(bb[2] - bb[0], bb[3] - bb[1]))
    fx0, fy0 = int(np.clip(cx - s / 2, 0, w - 1)), int(np.clip(cy - s / 2, 0, h - 1))
    crop = img[fy0:fy0 + s, fx0:fx0 + s]
    ch, cw = crop.shape[:2]
    sm = cv2.resize(crop, (256, round(256 * ch / cw)), interpolation=cv2.INTER_AREA)
    r = mp_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(sm[:, :, ::-1])))
    if not r.face_landmarks:
        return None
    pts = np.array([(q.x * cw + fx0, q.y * ch + fy0) for q in r.face_landmarks[0]])
    e = pts[LIDS["l"] + LIDS["r"]]
    x0, y0, side = box_from(*e.mean(0), e[:, 0].max() - e[:, 0].min(), w, h)
    m = np.zeros((side, side), np.uint8)
    for k in LIDS:
        poly = pts[LIDS[k]]; c = poly.mean(0)
        cv2.fillPoly(m, [np.round(c + (poly - c) * GROW - (x0, y0)).astype(np.int32)], 255)
    return (x0, y0, side), cv2.GaussianBlur(m, (0, 0), SIG)


def mask_kps(img, kps):
    h, w = img.shape[:2]
    le, re_ = kps[0], kps[1]
    iod = float(np.linalg.norm(re_ - le))
    ang = math.degrees(math.atan2(re_[1] - le[1], re_[0] - le[0]))
    x0, y0, side = box_from(*((le + re_) / 2), 1.47 * iod, w, h)
    m = np.zeros((side, side), np.uint8)
    for c in (le, re_):
        cv2.ellipse(m, (int(c[0] - x0), int(c[1] - y0)), (int(0.30 * iod), int(0.13 * iod)), ang, 0, 360, 255, -1)
    return (x0, y0, side), cv2.GaussianBlur(m, (0, 0), SIG)


def run(img, seed):
    st = {}
    torch.cuda.synchronize(); t = time.perf_counter()
    d = detect(img)
    torch.cuda.synchronize(); st['det'] = time.perf_counter() - t; t = time.perf_counter()
    if d is None:
        return img, None, None, st
    mk = mask_mp(img, d[0]) if a.mask == 'mp' else None
    used = 'mp'
    if mk is None:
        mk, used = mask_kps(img, d[1]), 'kps'
    (x0, y0, side), m = mk
    st['mask'] = time.perf_counter() - t; t = time.perf_counter()
    crop = img[y0:y0 + side, x0:x0 + side]
    x = torch.from_numpy(cv2.resize(crop, (a.res, a.res), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].copy()).to(dev)
    x = x.permute(2, 0, 1)[None].half() / 255
    torch.cuda.synchronize(); st['crop'] = time.perf_counter() - t; t = time.perf_counter()
    with torch.no_grad():
        y = pipe(prompt_embeds=pe, image=x, num_inference_steps=n_inf, strength=a.strength, guidance_scale=0.0,
                 output_type='pt', generator=torch.Generator(dev).manual_seed(seed)).images
    torch.cuda.synchronize(); st['diff'] = time.perf_counter() - t; t = time.perf_counter()
    y = torch.nn.functional.interpolate(y.float(), (side, side), mode='bicubic', align_corners=False).clamp(0, 1)
    y = (y[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()[:, :, ::-1]
    al = (m.astype(np.float32) / 255)[:, :, None]
    out = img.copy()
    out[y0:y0 + side, x0:x0 + side] = np.round(crop * (1 - al) + y * al).astype(np.uint8)
    full_m = np.zeros(img.shape[:2], np.uint8); full_m[y0:y0 + side, x0:x0 + side] = m
    st['paste'] = time.perf_counter() - t
    return out, full_m, used, st


paths = sorted(glob.glob(f'{BASE}/*.png'))
os.makedirs(OUT, exist_ok=True)
run(cv2.imread(paths[0]), 0); run(cv2.imread(paths[1]), 0)  # warmup
torch.cuda.reset_peak_memory_stats()
per, stages, diffs, used_m = {}, [], {}, {}
for pth in paths:
    bot = os.path.splitext(os.path.basename(pth))[0]
    img = cv2.imread(pth, cv2.IMREAD_COLOR)
    torch.cuda.synchronize(); ts = time.perf_counter()
    out, fm, used, st = run(img, a.seed)
    per[bot] = round((time.perf_counter() - ts) * 1000, 1)
    stages.append(st); used_m[bot] = used
    cv2.imwrite(f'{OUT}/{bot}.png', out)
    d = np.abs(out.astype(np.int16) - img.astype(np.int16)).mean(2)
    rec = {'in_mask': None, 'outside': round(float(d.mean()), 4), 'changed_px_outside': 0}
    if fm is not None:
        rec = {'in_mask': round(float(d[fm > 127].mean()), 3), 'in_mask_any': round(float(d[fm > 0].mean()), 3),
               'outside': round(float(d[fm == 0].mean()), 4), 'changed_px_outside': int((d[fm == 0] > 0).sum())}
        ref = f'/b/eyemask2x/{bot}.png'
        if os.path.exists(ref):
            r = cv2.imread(ref, 0) > 127; u = fm > 127
            rec['iou_vs_eyemask2x'] = round(float((r & u).sum() / max(1, (r | u).sum())), 3)
    diffs[bot] = rec
    print(bot, per[bot], 'ms', used, {k: round(v * 1000, 1) for k, v in st.items()}, rec, flush=True)
v = np.array(list(per.values()))
sk = ['det', 'mask', 'crop', 'diff', 'paste']
ok = [d for d in diffs.values() if d['in_mask'] is not None]
res = {'method': f'sd-turbo img2img fp16 {a.res}px eye-band crop, strength {a.strength}, {a.steps} step(s) '
                 f'(num_inference_steps={n_inf}), guidance 0, mask={a.mask}, prompt: {PROMPT}',
       'per_image_ms': per, 'p50_ms': round(float(np.percentile(v, 50)), 1), 'p95_ms': round(float(np.percentile(v, 95)), 1),
       'cold_load_s': round(cold, 2), 'gpu': '3090 GPU3', 'vram_peak_mb': round(torch.cuda.max_memory_allocated() / 2**20),
       'stage_ms_p50': {k: round(float(np.median([s[k] for s in stages if k in s])) * 1000, 1) for k in sk},
       'stage_ms_p95': {k: round(float(np.percentile([s[k] for s in stages if k in s], 95)) * 1000, 1) for k in sk},
       'mask_used': used_m,
       'pixel_diff': {'mean_in_mask': round(float(np.mean([d['in_mask'] for d in ok])), 3),
                      'mean_outside': round(float(np.mean([d['outside'] for d in ok])), 4),
                      'changed_px_outside_total': int(sum(d['changed_px_outside'] for d in ok)),
                      'mean_iou_vs_eyemask2x': round(float(np.mean([d['iou_vs_eyemask2x'] for d in ok if 'iou_vs_eyemask2x' in d])), 3),
                      'per_image': diffs},
       'notes': 'warm end-to-end per image = RetinaFace det (640px, GPU) + eyelid mask + crop/resize + diffusion (VAE enc + UNet + VAE dec) + '
                'paste; image read/write excluded. mask=mp: MediaPipe FaceLandmarker (CPU, XNNPACK) on a 256px face crop, lids polygon x1.3, '
                'Gaussian sigma 3 (same recipe as eyemask2x); falls back to kps ellipse if MediaPipe misses. mask=kps: two ellipses '
                '(0.30 x 0.13 IOD) at RetinaFace eye kps. Prompt embeds cached at load (text encoder freed). VAE = SD2.1 VAE '
                '(osd_models/sd21/vae). in_mask = mean |out-base| (0-255) where mask>127; outside = where mask==0. GPU 3 shared.'}
print(json.dumps({k: res[k] for k in ('p50_ms', 'p95_ms', 'cold_load_s', 'vram_peak_mb', 'stage_ms_p50')}),
      json.dumps({k: v for k, v in res['pixel_diff'].items() if k != 'per_image'}))
os.makedirs('/b/lat', exist_ok=True)
json.dump(res, open(f'/b/lat/{a.variant}.json', 'w'), indent=1)
