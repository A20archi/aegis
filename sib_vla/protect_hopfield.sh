#!/bin/bash
# Continuously guards Hopfield: if GPU free < DANGER for 2 consecutive checks, kill ONLY my
# boost eval procs (never Hopfield). Ensures the OOM-killer never fires.
set -uo pipefail; cd "$(dirname "$0")"
DANGER=${DANGER:-2800}; low=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${free:-99999}" -lt "$DANGER" ]; then
    low=$((low+1))
    if [ "$low" -ge 2 ]; then
      for pid in $(ps -eo pid,args | grep 'eval_libero_v.py --config configs/_boost' | grep -v grep | awk '{print $1}'); do kill -9 "$pid" 2>/dev/null; done
      echo "[$(date +%T)] PROTECT: free=${free}MiB < ${DANGER} -> killed MY eval to spare Hopfield" >> protect_hopfield.out
      low=0
    fi
  else low=0; fi
  sleep 12
done
