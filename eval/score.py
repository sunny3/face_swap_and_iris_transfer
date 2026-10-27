# Per method x image: identity (ArcFace cos to the bot avatar, buffalo_l + independent glintr100), face quality
# (pyiqa topiq_nr-face / hyperiqa on the 512 aligned face), eye sharpness (Laplacian var in the aligned eye boxes),
# background damage (mean abs diff vs input outside the dilated face box). Runs in docker on alaska GPU 3.
import json, os, cv2, numpy as np, pyiqa, torch
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import insightface

B = Path("/b"); OUT = B / "out"
fa = FaceAnalysis(name="buffalo_l", root="/b/insightface", providers=["CUDAExecutionProvider"]); fa.prepare(ctx_id=0, det_size=(640, 640))
glint = insightface.model_zoo.get_model("/b/insightface/models/antelopev2/glintr100.onnx", providers=["CUDAExecutionProvider"]); glint.prepare(ctx_id=0)
gl = lambda img, f: (lambda e: e / np.linalg.norm(e))(glint.get(img, f))
iqa = {n: pyiqa.create_metric(n, device="cuda") for n in ("topiq_nr-face", "hyperiqa")}


def face(img):
    fs = fa.get(img)
    return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if fs else None


def to_t(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).cuda()


def eye_sharp(al):  # aligned 512: arcface template eyes at ~(192,245),(320,245)
    g = cv2.cvtColor(al, cv2.COLOR_BGR2GRAY)
    return float(np.mean([cv2.Laplacian(g[215:275, x - 45:x + 45], cv2.CV_64F).var() for x in (192, 320)]))


refs = {}
for p in (B / "avatars").iterdir():
    a = cv2.imread(str(p)); f = face(a)
    if f is not None:
        refs[p.stem] = (f.normed_embedding, gl(a, f))
raw = {p.stem: cv2.imread(str(p)) for p in sorted((B / "in").iterdir())}
dirs = {"before": B / "in", **{d.name: d for d in sorted(OUT.iterdir()) if d.is_dir()}}
rows = []
for m, d in dirs.items():
    for p in sorted(d.iterdir()):
        if p.stem not in raw or p.stem not in refs:
            continue
        img = cv2.imread(str(p)); r = raw[p.stem]
        f = face(img)
        if f is None:
            rows.append({"method": m, "img": p.stem, "face": False}); continue
        al = face_align.norm_crop(img, f.kps, image_size=512)
        sc = r.shape[1] / img.shape[1]
        small = img if sc == 1 else cv2.resize(img, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = (f.bbox * sc).astype(int); w, h = x1 - x0, y1 - y0
        mask = np.ones(r.shape[:2], bool); mask[max(0, y0 - h):y1 + h, max(0, x0 - w):x1 + w] = False
        bg = float(np.abs(small.astype(int) - r.astype(int))[mask].mean()) if mask.any() else 0.0
        x0, y0, x1, y1 = f.bbox.astype(int); s = int(max(x1 - x0, y1 - y0) * 0.85)  # face crop for the UI
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        (B / "crops" / m).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(B / "crops" / m / f"{p.stem}.jpg"), cv2.resize(img[max(0, cy - s):cy + s, max(0, cx - s):cx + s], (448, 448), interpolation=cv2.INTER_LANCZOS4), [cv2.IMWRITE_JPEG_QUALITY, 93])
        (B / "crops_eyes" / m).mkdir(parents=True, exist_ok=True)  # eye band from a 1024 aligned face (keeps x2 detail)
        al2 = cv2.warpAffine(img, face_align.estimate_norm(f.kps, 1024), (1024, 1024), flags=cv2.INTER_LANCZOS4)  # eyes land at ~(370,413),(652,413)
        cv2.imwrite(str(B / "crops_eyes" / m / f"{p.stem}.jpg"), al2[318:508, 250:772], [cv2.IMWRITE_JPEG_QUALITY, 94])
        with torch.no_grad():
            q = {n: float(fn(to_t(al))) for n, fn in iqa.items()}
        e, g = refs[p.stem]
        rows.append({"method": m, "img": p.stem, "face": True, "id_sim": float(np.dot(e, f.normed_embedding)),
                     "id_glint": float(np.dot(g, gl(img, f))), "eye_sharp": eye_sharp(al), "bg_diff": bg,
                     "size": f"{img.shape[1]}x{img.shape[0]}", **q})
json.dump(rows, open(B / "scores.json", "w"), indent=1)
by = {}
for r in rows:
    by.setdefault(r["method"], []).append(r)
print(f"{'method':30s} faces  id_sim id_glint topiq_face hyperiqa eye_sharp bg_diff")
for m, rs in sorted(by.items(), key=lambda kv: -np.mean([r["topiq_nr-face"] for r in kv[1] if r["face"]] or [0])):
    ok = [r for r in rs if r["face"]]
    if not ok:
        print(f"{m:30s} 0/{len(rs)}"); continue
    mean = lambda k: np.mean([r[k] for r in ok])
    print(f"{m:30s} {len(ok)}/{len(rs)}  {mean('id_sim'):.3f}  {mean('id_glint'):.3f}    {mean('topiq_nr-face'):.3f}     {mean('hyperiqa'):.3f}   {mean('eye_sharp'):7.1f}  {mean('bg_diff'):.2f}")
