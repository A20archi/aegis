#!/bin/bash
# Low-frequency watcher: pings when guard STARTS a suite (GPU freed), a suite completes, or 30-min heartbeat.
set -uo pipefail; cd "$(dirname "$0")"
t0=$(date +%s); HB=${HB:-1800}
while true; do
  done=$(ls results/boost_base/*/libero_v/baseline/eval_clean.json 2>/dev/null | wc -l)
  [ "$done" -ge 1 ] && { echo "BOOST: $done suite(s) DONE $(date +%T)"; break; }
  if grep -q "START" results/boost_eval_safe.log 2>/dev/null; then echo "BOOST: GPU freed -> eval STARTED $(date +%T)"; tail -2 results/boost_eval_safe.log; break; fi
  pgrep -fc run_boost_eval_safe.sh >/dev/null 2>&1 || { echo "BOOST guard died (relaunch needed) $(date +%T)"; break; }
  [ $(( $(date +%s)-t0 )) -ge $HB ] && { echo "BOOST still waiting: free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)MiB (need 24000) $(date +%T)"; break; }
  sleep 120
done
