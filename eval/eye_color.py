"""Iris colour check: mean CIELAB inside the iris ring (r*0.4..r*0.95, inside the lid opening, top-8% L highlights
dropped) for prod (input x2), base (CodeFormer x2 + skin), fast eye fix, and the bot avatar. Same landmarks for the
three render stages (input landmarks x2); the avatar gets its own. Prints dE2000 vs prod / vs avatar and dL."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import iris_variants as V
from skimage.color import deltaE_ciede2000

R = Path(__file__).resolve().parent
ST = {"prod": lambda b: cv2.resize(cv2.imread(str(R / "in" / f"{b}.jpg")), (1024, 1536), interpolation=cv2.INTER_CUBIC),
      "base": lambda b: cv2.imread(str(R / "work" / "base_x2sd" / f"{b}.png")),
      "fasteye": lambda b: cv2.imread(str(R / "out" / "fasteye_sdturbo_s0.35" / f"{b}.jpg")),
      "clock": lambda b: cv2.imread(str(R / "out" / "fasteye_sdturbo_s0.35_clock" / f"{b}.jpg"))}


def ring_lab(img, e):
    h, w = img.shape[:2]; yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - e["c"][0], yy - e["c"][1])
    lid = np.zeros((h, w), np.uint8); cv2.fillPoly(lid, [np.round(e["lid"]).astype(np.int32)], 1)
    m = (d >= e["r"] * 0.4) & (d <= e["r"] * 0.95) & (lid > 0)
    if m.sum() < 12:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    lab[:, 0] *= 100 / 255; lab[:, 1:] -= 128
    lab = lab[lab[:, 0] <= np.percentile(lab[:, 0], 92)]  # drop catchlights
    return lab.mean(0)


rows = []
for p in sorted((R / "in").glob("*.jpg")):
    b = p.stem
    pi = V.lm(cv2.imread(str(p)))
    pa = V.lm(cv2.imread(str(R / "avatars" / f"{b}.jpg")))
    if pi is None:
        rows.append({"bot": b, "skip": "no landmarks on prod"}); continue
    pi = pi * 2
    av = cv2.imread(str(R / "avatars" / f"{b}.jpg"))
    for k in ("l", "r"):
        e = V.eye(pi, k); r = {"bot": b, "eye": k}
        for s, f in ST.items():
            r[s] = ring_lab(f(b), e)
        r["avatar"] = ring_lab(av, V.eye(pa, k)) if pa is not None else None
        rows.append(r)

de = lambda a, c: float(deltaE_ciede2000(a[None, None], c[None, None])[0, 0]) if a is not None and c is not None else None
out = []
for r in rows:
    if "skip" in r:
        out.append(r); continue
    o = {"bot": r["bot"], "eye": r["eye"]}
    for s in ("base", "fasteye", "clock"):
        o[f"dE_{s}_vs_prod"] = de(r[s], r["prod"])
        o[f"dL_{s}_vs_prod"] = None if r[s] is None or r["prod"] is None else float(r[s][0] - r["prod"][0])
    for s in ("prod", "fasteye", "clock"):
        o[f"dE_{s}_vs_avatar"] = de(r[s], r["avatar"])
    o["lab"] = {s: None if r[s] is None else [round(float(x), 1) for x in r[s]] for s in ("prod", "base", "fasteye", "clock", "avatar")}
    out.append(o)
json.dump(out, open(R / "eye_color.json", "w"), indent=1)
ok = [o for o in out if "skip" not in o and o["dE_fasteye_vs_prod"] is not None]
med = lambda k: np.median([o[k] for o in ok if o[k] is not None])
p90 = lambda k: np.percentile([o[k] for o in ok if o[k] is not None], 90)
print(f"eyes measured: {len(ok)}  skipped bots: {[o['bot'] for o in out if 'skip' in o]}")
for k in ("dE_base_vs_prod", "dE_fasteye_vs_prod", "dE_clock_vs_prod", "dL_base_vs_prod", "dL_fasteye_vs_prod", "dL_clock_vs_prod", "dE_prod_vs_avatar", "dE_fasteye_vs_avatar", "dE_clock_vs_avatar"):
    print(f"{k:22s} median {med(k):6.2f}   p90 {p90(k):6.2f}")
big = sorted(ok, key=lambda o: -o["dE_fasteye_vs_prod"])[:6]
for o in big:
    print(o["bot"], o["eye"], f"dE {o['dE_fasteye_vs_prod']:.1f} dL {o['dL_fasteye_vs_prod']:+.1f}", o["lab"]["prod"], "->", o["lab"]["fasteye"], "avatar", o["lab"]["avatar"])
