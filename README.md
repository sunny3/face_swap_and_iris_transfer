# Face & eye enhancement after face swap

Post-processing for bot pictures after the InsightFace swap (inswapper_128 + GFPGAN 1.4, the current prod output).
It adds realistic skin detail and fixes AI-corrupted eyes while keeping the bot's identity, and it runs in **about
0.55 s per picture on an RTX 3090**.

```
prod output (swap + GFPGAN, 512x768)
  └─ 1. face_enhance/fast_pipe.py      CodeFormer x2 + skin detail (+ avatar iris)   ~333 ms   → 1024x1536
       └─ 2. eye_fix/fasteye_sdturbo.py   1-step sd-turbo eye inpaint               ~216 ms   → 1024x1536
```

Both stages are timed separately, as warm processes on alaska GPU 3 (shared). They have **not yet been chained in one
process**, so ~0.55 s is the sum of the two measurements, not one end-to-end timing.

## Results (26 context pics, `examples/`)

| | identity to avatar ↑ | face quality ↑ | latency (3090, warm) |
|---|---|---|---|
| prod (swap + GFPGAN) | 0.828 | 0.595 | – |
| old slow chain: CodeFormer x2 + skin detail | 0.771 | 0.815 | 1646 ms |
| **1. `fast_cf_x2sd_iris`** (this repo) | 0.770 | 0.815 | **333 ms** p50 / 389 p95 |
| **1 + 2. + `fasteye_sdturbo_s0.35`** (this repo) | 0.763 | 0.809 | **+216 ms** p50 / 246 p95 |

- **Identity:** ArcFace (buffalo_l) cosine to the bot avatar.
- **Face quality:** pyiqa `topiq_nr-face` on the 512 aligned face.
- **Example images:** `examples/eyes_before_after_*of4.jpg` show the eye band for all 26 bots, prod vs final. They
  are the pictures posted to #computer-vision on 2026-10-25.
- **Pixels outside the eyes:** stage 2 leaves them byte-identical (0 px changed outside the eye mask).
- **Eye colour:** stage 2 makes irises slightly lighter. The median iris-colour difference to prod is ΔE2000 5.3,
  with lightness +3.1. This was reviewed and accepted (2026-10-25).
  - Optional colour lock: shift each iris's Lab mean back to prod, ~3 ms. That brings it to ΔE 1.5. The code is in
    `eval/eye_color.py` (measurement); the lock itself lives in `~/draw_illustrations/facefix/eye_colorlock.py`.

## Stage 1 — `face_enhance/fast_pipe.py`

Reproduces CodeFormer `-w 1.0 -s 2 --bg_upsampler realesrgan --face_upsample`, then SkinDiffDetail on the face, then
avatar iris transfer. All of it runs in **one warm GPU process**. The model code and weights are the official ones;
only orchestration and precision changed.

1. **Detect** with RetinaFace r50 (CodeFormer `facelib`, resize 640) once, reused by every stage. The anchor grid is
   cached per input size (the original rebuilds it in a Python loop on every call: 70 → 16 ms).
2. **Restore** the 512 aligned face with CodeFormer, `w = 1.0`, fp16 autocast, under a CUDA graph. The codebook
   lookup is replaced with an identical-output copy, because the original builds a CPU tensor, which can't be graph
   captured.
3. **Upscale x2** with RealESRGAN_x2plus fp16, untiled: on the 512 restored face (1024 face) and on the full frame
   for the background.
   - Variant `*_bgbicubic` uses a GPU bicubic background instead: 226 ms, with the same face scores.
4. **Paste back** on the GPU, inside the face ROI only. It re-implements `FaceRestoreHelper.paste_faces_to_input_image`
   with the same constants: grid_sample inverse affine, erosion, Gaussian feather, ParseNet face mask (fp16). This
   took 1041 ms on CPU and takes 22 ms here.
5. **Skin detail:** `1x-ITF-SkinDiffDetail-Lite-v1` (spandrel, fp16) on the face bbox + 40%, pasted with a feathered
   ellipse.
6. **Avatar iris** (variants containing `iris`):
   - MediaPipe FaceLandmarker runs on the face crop in a CPU thread while the GPU works. The avatar's landmarks are
     cached per bot.
   - The avatar's clean iris disc is warped (similarity transform) into each eye, Lab-locked to the render's iris,
     and pasted only inside the iris ∩ eyelid opening.
   - An eye is skipped if the avatar's iris covers < 85 % of the visible iris, or if the gaze differs by
     > 0.12 eye-widths. On the bench 30 of 52 eyes were replaced.

| stage p50 (ms) | decode | detect | bg x2 | CodeFormer | face x2 | paste | skin | iris |
|---|---|---|---|---|---|---|---|---|
| `fast_cf_x2sd_iris` | 3 | 16 | 105 | 47 | 72 | 22 | 56 | 4 |

- VRAM: about 3.6 GB.
- Cold load: about 3.5 s.
- Rejected: the `facefp` option (face x2 at the on-canvas size) costs about 25 % face quality.

Run on alaska:

```bash
bash run_pipe.sh fast_cf_x2sd_iris 1 bench cl graph parse16
```

- **Arguments:** `<variant> <save 0|1> [opts]`. Variant name flags: `iris`, `bgbicubic`, `facefp`.
- **Options:** `bench`, `cl` (channels_last), `graph` (CUDA graphs), `parse16`, `cf32`, `det16`.
- **Environment:** `REP` = number of timing passes (default 3). `FINE=1` gives a finer per-stage breakdown.
- **Reads:** `/b/in/<bot>.jpg` and `/b/avatars/<bot>.jpg`.
- **Writes:** `/b/out/<variant>/<bot>.png` and `/b/lat/<variant>.json` (latency, per-stage times, iris stats).

## Stage 2 — `eye_fix/fasteye_sdturbo.py`

A one-step diffusion eye inpaint. It is the official diffusers sd-turbo img2img call with the parameters patched.

1. **Detect** with RetinaFace (640 px, GPU) and take the largest face.
2. **Eye mask** (`--mask mp`, the default and the chosen one):
   - MediaPipe FaceLandmarker runs on a 256 px face crop.
   - Both eyelid polygons are grown ×1.3 and blurred with a Gaussian (σ 3).
   - If MediaPipe misses, it falls back to `--mask kps`: two ellipses (0.30 × 0.13 IOD) at RetinaFace's eye points.
     That mask is ~30 ms faster but looser.
3. **Crop** a square eye band around both eyes (≥ 2.2 × eye span, ≥ 192 px) and resize it to 512.
4. **sd-turbo img2img**, fp16:
   - `strength 0.35`, 1 actual step (`num_inference_steps = 3`), `guidance_scale 0`, seed 0.
   - The prompt is fixed and its embeds are cached at load; the text encoder is freed.
   - Prompt: *"close-up of a woman's eyes, natural realistic eyes, perfectly round irises with detailed iris texture,
     round dark pupils, natural catchlights, sharp eyelashes"*.
5. **Paste** the result, resized back, only inside the soft eye mask. Everything else stays byte-identical.

- Stage p50 (ms): detect 35 (shareable with stage 1), MediaPipe mask 36 (CPU), crop 5, diffusion 132, paste 3.
- VRAM: 2.6 GB. Cold load: 12–15 s.
- Other variants tested:

  | variant | latency p50 | identity |
  |---|---|---|
  | `--strength 0.5` | 239 ms | 0.755 |
  | `--strength 0.5 --steps 2` | 263 ms | 0.754 |
  | `--mask kps` | 188 ms | 0.761 |

Run on alaska:

```bash
bash fasteye_run.sh fasteye_sdturbo.py --variant fasteye_sdturbo_s0.35 --strength 0.35 --steps 1 --mask mp
```

**Input:** `BASE` at the top of the script is `/b/out/codeformer_w1.0_x2_skindiff`, the slow-chain output that the bench
used. To chain it after stage 1, set `BASE = '/b/out/fast_cf_x2sd_iris'`.

## Environment (alaska)

Everything runs in `pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime` on `--gpus device=3`. Extra packages are
installed with `pip --target` (torch is never reinstalled). Mounts: `/b = ~/facefix_bench`, `/w = ~/codeformer_run`.

| what | where on alaska | source |
|---|---|---|
| CodeFormer code + `basicsr` + `facelib` | `~/codeformer_run/CodeFormer` (`basicsr/version.py` created by hand) | github.com/sczhou/CodeFormer |
| CodeFormer, RetinaFace, ParseNet, RealESRGAN_x2plus weights | `~/codeformer_run/CodeFormer/weights/{CodeFormer,facelib,realesrgan}` | CodeFormer `scripts/download_pretrained_models.py` + v0.1.0 release |
| CodeFormer deps | `~/codeformer_run/pylib` (pip `--no-deps`) | – |
| SkinDiffDetail + MediaPipe model | `~/facefix_bench/models_fast/{1x-ITF-SkinDiffDetail-Lite-v1.pth, face_landmarker.task}` | OpenModelDB; Google MediaPipe |
| stage-1 deps | `~/facefix_bench/pylib_fast` (spandrel 0.4.2, safetensors 0.8.0, einops 0.8.2) + `pylib_mp` (mediapipe 0.10.31, absl-py, attrs) | PyPI |
| sd-turbo (unet / text enc / tokenizer / scheduler) | `~/facefix_bench/hf_ir/hub/models--stabilityai--sd-turbo` (that snapshot has no VAE) | HF `stabilityai/sd-turbo` |
| SD2.1 VAE | `~/facefix_bench/osd_models/sd21/vae` | HF `sd2-community/stable-diffusion-2-1-base` |
| stage-2 deps | `~/facefix_bench/pylib_osd` (diffusers 0.27.2, transformers 4.44.2, huggingface_hub 0.25.2, tokenizers 0.19.1) + `pylib_mp`; `face_landmarker.task` in `~/facefix_bench/models_mp/` | PyPI |

The scripts on alaska are `~/facefix_bench/scripts_fast/fast_pipe.py` and `~/facefix_bench/fasteye_scripts/fasteye_sdturbo.py`,
identical (md5) to the copies here. The run scripts take `flock ~/facefix_bench/gpu3.lock` so that timings aren't
shared with other jobs.

## Evaluation — `eval/`

- **`score.py`** (`score.sh` runs it in docker on GPU 3) scores every `/b/out/<method>/` against the prod inputs.
  - Metrics: ArcFace identity to the avatar (buffalo_l + glintr100), pyiqa `topiq_nr-face` + `hyperiqa`, eye
    Laplacian sharpness, and background change.
  - It also writes the face and eye crops used by the comparison pages.
  - Eye sharpness **cannot rank eye fixes**: cleaner irises score lower. Judge eyes visually.
- **`eye_color.py`** measures iris-ring CIELAB and ΔE2000: prod vs after vs avatar (needs `~/draw_illustrations/facefix`
  helpers, which use MediaPipe).

The comparison pages are in `~/draw_illustrations/facefix/` and are served by `server.py` on
http://127.0.0.1:8765/facefix/:
- `facefix.html`: all 30+ methods
- `eyes.html`: eye detailers
- `fasteye.html`: fast eye fix only
- `eyes_crops.html`: eye crops, prod vs after

## What was tried and rejected (same 26 pics)

- **GPEN-BFR 512/1024/2048 and RestoreFormer++:** barely above prod quality.
- **OSDFace x2:** 0.77 / 0.79 at 1.3 s, below CodeFormer.
- **InstantRestore** (avatar as reference): 0.76 / 0.58.
- **Krea-2 ZenFaceDetailer on the whole face** (denoise 0.20–0.28): **destroys identity** (0.27–0.41).
- **Krea-2 eye detailers** (Impact-Pack YOLO eyes, or a MediaPipe eye mask, 8 steps): the same effect as stage 2,
  but 6–45 s.
- **Gemini 3 Pro iris paste-back:** keeps identity, but 16 s and $0.14 per picture.

## Known limits

- MediaPipe on the full frame misses small faces (1 of 26, 2418616). Stage 2 runs it on a face crop, which found that
  face too. If it still misses, stage 2 falls back to the kps ellipse mask.
- Avatar iris transfer skips eyes where the avatar's iris is covered or looks elsewhere (22 of 52 eyes skipped).
- Small faces: the eyes are only ~35–70 px wide even at 1024×1536, which is why the eye band is upscaled to 512
  before diffusion.
- Latency was measured on a **shared** 3090 (another ~13 GB process on GPU 3), so the numbers are slightly
  pessimistic.
