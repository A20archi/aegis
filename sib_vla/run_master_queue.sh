#!/bin/bash
# MASTER QUEUE — waits for the GPU to FREE UP (external jobs to clear), then runs
# Stage 1 -> Stage 2 -> Stage 3 in sequence. Each stage self-parallelises with a
# live-VRAM-sized, OOM-safe scheduler (run_queue_core.sh). NanoVLA (Stage 4) is
# intentionally NOT queued (no released checkpoint; handled separately).
set -uo pipefail
cd "$(dirname "$0")"
START_FREE=${START_FREE:-68000}   # MiB free that signals external jobs have cleared
STABLE=${STABLE:-3}               # consecutive checks above START_FREE before starting
POLL=${POLL:-60}
START_AFTER=${START_AFTER:-2026-06-20 18:00}   # do NOT execute before Saturday evening
target=$(date -d "$START_AFTER" +%s 2>/dev/null || echo 0)

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

echo "[$(date '+%F %T')] MASTER armed — will start ONLY after [$START_AFTER] AND GPU free >= ${START_FREE}MiB x${STABLE}"
echo "[$(date '+%F %T')] protecting external jobs: hopfield_grpo + hamlet_paper_bc_libero (never touched)"

# --- time gate: hold until Saturday evening ---
while [ "$(date +%s)" -lt "$target" ]; do
  echo "[$(date '+%F %T')] before start window [$START_AFTER] — holding"
  sleep 900
done
echo "[$(date '+%F %T')] start window reached — now waiting for GPU to free"

ok=0
while true; do
  f=$(free_mib); f=${f:-0}
  if [ "$f" -ge "$START_FREE" ]; then
    ok=$((ok+1)); echo "[$(date +%T)] free=${f}MiB OK ($ok/$STABLE)"
    [ "$ok" -ge "$STABLE" ] && break
  else
    [ "$ok" -ne 0 ] && echo "[$(date +%T)] free=${f}MiB < ${START_FREE}, reset"
    ok=0
  fi
  sleep "$POLL"
done

export HARDCAP=1   # ONE job at a time (serial), per "execute one by one"
echo "[$(date +%T)] GPU freed -> starting pipeline (SERIAL, one job at a time)"

# --- host-RAM watchdog: last-resort guard so the system OOM-killer never fires.
# If available host RAM drops below RAM_CRIT for 2 consecutive checks, kill only
# MY newest eval process (matches our eval scripts) — NEVER external jobs. ---
RAM_CRIT=${RAM_CRIT:-25000}   # MiB available; below this twice => shed my newest job
(
  low=0
  while true; do
    avail=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}'); avail=${avail:-999999}
    if [ "$avail" -lt "$RAM_CRIT" ]; then
      low=$((low+1))
      if [ "$low" -ge 2 ]; then
        pid=$(pgrep -f 'eval_libero_v.py --config configs/_(lv|abl|boost)|libero_plus_aegis_eval.py' | tail -1)
        [ -n "$pid" ] && { kill -9 "$pid" 2>/dev/null; echo "[$(date '+%F %T')] RAM-GUARD: avail=${avail}MiB < ${RAM_CRIT} -> killed MY eval $pid (external untouched)"; }
        low=0
      fi
    else low=0; fi
    sleep 10
  done
) & RAMGUARD=$!
echo "[$(date +%T)] host-RAM watchdog armed (PID $RAMGUARD, crit=${RAM_CRIT}MiB)"
trap 'kill $RAMGUARD 2>/dev/null' EXIT
bash run_stage1_liberov_objgoal.sh ; echo "[$(date +%T)] >>> stage1 returned"
bash run_stage2_liberoplus.sh      ; echo "[$(date +%T)] >>> stage2 returned"
bash run_stage3_ablations.sh       ; echo "[$(date +%T)] >>> stage3 returned"
echo "[$(date +%T)] ===== MASTER QUEUE COMPLETE (stages 1-3) ====="
