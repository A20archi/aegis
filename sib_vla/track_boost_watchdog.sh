#!/bin/bash
# Pings on: GPU free < FLOOR (act before breaking 10GB rule), a suite completing, or heartbeat.
set -uo pipefail; cd "$(dirname "$0")"
t0=$(date +%s); HB=${HB:-1500}; FLOOR=${FLOOR:-11000}; DONE0=${DONE0:-0}
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  done=$(ls results/boost_base/*/libero_v/baseline/eval_clean.json 2>/dev/null | wc -l)
  [ "${free:-99999}" -lt "$FLOOR" ] && { echo "WATCHDOG: GPU free=${free}MiB < ${FLOOR} -> approaching 10GB rule, SHED a suite $(date +%T)"; break; }
  [ "$done" -gt "$DONE0" ] && { echo "BOOST: $done/4 suites done $(date +%T)"; break; }
  pgrep -fc run_boost_eval_safe.sh >/dev/null 2>&1 || { echo "BOOST runner gone ($done/4) $(date +%T)"; break; }
  [ $(( $(date +%s)-t0 )) -ge $HB ] && { echo "BOOST hb: free=${free}MiB, $done/4 done $(date +%T)"; break; }
  sleep 30
done
