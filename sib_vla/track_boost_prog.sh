#!/bin/bash
set -uo pipefail; cd "$(dirname "$0")"
t0=$(date +%s); HB=${HB:-1500}; LT0=${LT0:-0}
while true; do
  done=$(ls results/boost_base/*/libero_v/baseline/eval_clean.json 2>/dev/null | wc -l)
  [ "$done" -ge 1 ] && { echo "BOOST: $done suite(s) DONE $(date +%T)"; break; }
  lt=$(grep -cE 'clean t[0-9]+:' results/boost_base/long/baseline.log 2>/dev/null || echo 0)
  [ "$lt" -ge 3 ] && [ "$lt" -gt "$LT0" ] && { echo "BOOST: Long $lt/10 tasks (projectable) $(date +%T)"; break; }
  pgrep -fc run_boost_eval_safe.sh >/dev/null 2>&1 || { echo "BOOST guard gone (Long $lt/10) $(date +%T)"; tail -2 results/boost_eval_safe.log; cat protect_hopfield.out 2>/dev/null; break; }
  [ $(( $(date +%s)-t0 )) -ge $HB ] && { echo "BOOST hb: Long $lt/10, free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)MiB $(date +%T)"; break; }
  sleep 60
done
