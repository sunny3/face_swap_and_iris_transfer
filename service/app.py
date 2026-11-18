"""HTTP service: stage 1 (face_enhance/fast_pipe.py, variant fast_cf_x2sd_iris) + stage 2
(eye_fix/fasteye_sdturbo.py, strength 0.35, 1 step, mask mp) chained in one warm process.

POST /enhance  {image_base64, avatar_base64? | avatar_sha1?}  + X-API-Token
  image_base64   the swapped render (swap + GFPGAN, e.g. 512x768)
  avatar_base64  the swap donor (bot avatar / gold face) -> iris source; empty = no iris transfer
  avatar_sha1    sha1 of an avatar sent before (cached, LRU 512): no re-upload; 409 if it fell out of the cache
-> {image_base64 (JPEG, 2x the input), faces, eyes_replaced, eye_mask, timings_ms}
GET /health

The models are not thread-safe (CUDA graphs, FaceRestoreHelper state): every request runs under one lock.
Run with service/run_service.sh (alaska, docker, cwd=/w/CodeFormer).
"""
import base64
import hashlib
import os
import sys
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'face_enhance'), os.path.join(HERE, '..', 'eye_fix')]
TOKEN = os.environ['FACEFIX_TOKEN']
JPEG_Q = int(os.environ.get('FACEFIX_JPEG_QUALITY', '95'))

# both scripts parse sys.argv at import: give each its README bench arguments
sys.argv = ['fast_pipe.py', 'fast_cf_x2sd_iris', '0', 'cl', 'graph', 'parse16']
import fast_pipe as s1  # noqa: E402
sys.argv = ['fasteye_sdturbo.py', '--variant', 'service', '--strength', '0.35', '--steps', '1', '--mask', 'mp']
import fasteye_sdturbo as s2  # noqa: E402

LOCK = threading.Lock()
AVATARS = OrderedDict()   # sha1(avatar bytes) -> AVC key; s1.AVC holds the landmarks (per-bot cache in the bench)
MAX_AVATARS = 512


def _avatar_key(raw):
    """Fill s1.AVC for this avatar once (same code as the bench's per-bot cache) and return its key."""
    key = hashlib.sha1(raw).hexdigest()
    if key in AVATARS:
        AVATARS.move_to_end(key)
        return key
    av = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    pa = None if av is None else s1.lm(av)
    if pa is None:
        s1.AVC[key] = None
    else:
        ea = {k: s1.eye(pa, k) for k in "lr"}
        for k in "lr":
            ea[k]["mask"] = s1.vis_mask(ea[k], (0, 0, av.shape[1], av.shape[0]))
        s1.AVC[key] = (av, ea)
    AVATARS[key] = True
    if len(AVATARS) > MAX_AVATARS:
        s1.AVC.pop(AVATARS.popitem(last=False)[0], None)
    return key


def _b64(s, what):
    try:
        return base64.b64decode(s.split(',', 1)[-1], validate=False)
    except Exception:
        raise HTTPException(400, f'{what} is not base64')


def enhance(img_bytes, avatar_bytes, key=None):
    t0 = time.perf_counter()
    stats = {}
    if key is not None:
        AVATARS.move_to_end(key)
    elif avatar_bytes:
        key = _avatar_key(avatar_bytes)
    if key is None:   # stage 1 without iris: no AVC entry -> "no face on avatar" branch, nothing replaced
        key = '_none'; s1.AVC[key] = None
    res, nf = s1.process(img_bytes, key, stats)
    t1 = time.perf_counter()
    out, _, mask_used, _ = s2.run(res, 0)
    t2 = time.perf_counter()
    return out, {'faces': nf, 'eyes_replaced': stats.get(key, {}).get('n', 0), 'eye_mask': mask_used,
                 'timings_ms': {'stage1': round((t1 - t0) * 1000, 1), 'stage2': round((t2 - t1) * 1000, 1)}}


# warm both stages (CUDA graphs / allocator) on a blank frame
with LOCK:
    _z = cv2.imencode('.jpg', np.zeros((768, 512, 3), np.uint8))[1].tobytes()
    s1.lm(np.zeros((256, 256, 3), np.uint8))
    enhance(_z, b'')

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class EnhanceRequest(BaseModel):
    image_base64: str
    avatar_base64: str = ''
    avatar_sha1: str = ''    # sha1 of the avatar bytes sent earlier: skips re-uploading it; 409 when not cached


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/enhance')
def enhance_route(req: EnhanceRequest, x_api_token: str = Header(default='')):
    if x_api_token != TOKEN:
        raise HTTPException(401, 'bad token')
    img = _b64(req.image_base64, 'image_base64')
    avatar = _b64(req.avatar_base64, 'avatar_base64') if req.avatar_base64 else b''
    if cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) is None:
        raise HTTPException(400, 'image_base64 is not an image')
    t = time.perf_counter()
    with LOCK:
        wait = time.perf_counter() - t
        if not avatar and req.avatar_sha1:
            if req.avatar_sha1 not in AVATARS:
                raise HTTPException(409, 'avatar not cached: send avatar_base64')
            out, info = enhance(img, None, req.avatar_sha1)
        else:
            out, info = enhance(img, avatar)
    info['timings_ms']['queue'] = round(wait * 1000, 1)
    ok, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return {'image_base64': base64.b64encode(jpg.tobytes()).decode(), **info}
