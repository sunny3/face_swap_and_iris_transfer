#!/bin/bash
# usage: run_service.sh [gpu=0] [port=9152]   (alaska; weights/libs from the bench dirs, see README "Environment")
# Needs ~/facefix_service/.env with FACEFIX_TOKEN=... and fastapi+uvicorn in ~/facefix_bench/pylib_svc:
#   docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v ~/facefix_bench:/b pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
#     pip install --no-deps --target /b/pylib_svc fastapi==0.115.6 starlette==0.41.3 uvicorn==0.32.1 h11==0.14.0 click \
#     anyio==4.7.0 sniffio pydantic==2.10.4 pydantic-core==2.27.2 annotated-types typing-extensions
# (newer fastapi imports opentelemetry, which the image lacks)
gpu=${1:-0}; port=${2:-9152}
SRC=$(cd "$(dirname "$0")/.." && pwd)
docker rm -f facefix_service 2>/dev/null
docker run -d --name facefix_service --restart unless-stopped --gpus device=$gpu \
  -u $(id -u):$(id -g) -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e HF_HOME=/b/hf_ir -e HF_HUB_OFFLINE=1 -e PYTHONWARNINGS=ignore \
  -e OMP_NUM_THREADS=2 --cpus 4 --memory 12g --env-file ~/facefix_service/.env \
  -p $port:8000 -v ~/facefix_bench:/b:ro -v ~/codeformer_run:/w:ro -v "$SRC":/svc:ro -w /w/CodeFormer \
  -e PYTHONPATH=/b/pylib_svc:/b/pylib_osd:/b/pylib_mp:/w/pylib:/w/CodeFormer:/b/pylib_fast \
  pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  python -m uvicorn --app-dir /svc/service app:app --host 0.0.0.0 --port 8000 --workers 1
