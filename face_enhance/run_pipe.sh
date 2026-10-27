#!/bin/bash
# usage: run_pipe.sh <variant> [save=1] [opts...]   (alaska; GPU 3; flock)
v=$1; save=${2:-1}; shift 2; opts="$*"
B=~/facefix_bench; cn=fastpipe_$$
flock $B/gpu3.lock docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -e USER=grylkov -e LOGNAME=grylkov -e REP=${REP:-3} -e FINE=${FINE:-0} --name $cn --gpus device=3 --network bridge -v $B:/b -v ~/codeformer_run:/w \
  -e PYTHONPATH=/w/pylib:/w/CodeFormer:/b/pylib_fast:/b/pylib_mp -w /w/CodeFormer pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  python /b/scripts_fast/fast_pipe.py "$v" "$save" $opts > $B/scripts_fast/log_$v.txt 2>&1 &
dpid=$!; peak=0
while kill -0 $dpid 2>/dev/null; do
  pids=$(docker top $cn -eo pid 2>/dev/null | tail -n +2 | tr '\n' '|')
  if [ -n "$pids" ]; then
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -E "^(${pids%|})," | awk -F', ' '{s+=$2} END{print s+0}')
    [ "$m" -gt "$peak" ] && peak=$m
  fi
  sleep 0.5
done
python3 - "$B/lat/$v.json" "$peak" <<'PY'
import json,sys
p,pk=sys.argv[1],int(sys.argv[2])
try:
    d=json.load(open(p)); d['vram_peak_mb']=pk if pk else None; json.dump(d,open(p,'w'),indent=1)
    print(d['method'],'p50',d['p50_ms'],'p95',d['p95_ms'],'cold',d['cold_load_s'],'vram',pk,d['stage_ms_p50'],d['failures'])
except Exception as e: print('NOJSON',p,e)
PY
grep -v -i -E "warn|W0000|I0000" $B/scripts_fast/log_$v.txt | tail -4
