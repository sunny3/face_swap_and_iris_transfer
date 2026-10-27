#!/bin/bash
# usage: fasteye_run.sh <script.py> <args...>   (alaska GPU 3 under the shared lock)
S=$1; shift
cd ~/facefix_bench
flock ~/facefix_bench/gpu3.lock docker run --rm --gpus device=3 --network bridge -e HF_HOME=/b/hf_ir -e HF_HUB_OFFLINE=1 -e PYTHONWARNINGS=ignore \
  -v ~/facefix_bench:/b -v ~/codeformer_run:/w pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  bash -c "PYTHONPATH=/b/pylib_osd:/b/pylib_mp:/w/pylib:/w/CodeFormer:/b/OSDFace python /b/fasteye_scripts/$S $* 2>&1 | grep -v -i -E 'warn|deprecat|W0000|INFO: Created'"
