#!/bin/bash
# Pings when any boost suite finishes (or all 4), or heartbeat. Reports partials.
set -uo pipefail; cd "$(dirname "$0")"
t0=$(date +%s); HB=${HB:-1800}; DONE0=${DONE0:-0}
while true; do
  done=$(ls results/boost_base/*/libero_v/baseline/eval_clean.json 2>/dev/null | wc -l)
  [ "$done" -ge 4 ] && { echo "BOOST EVAL ALL 4 DONE $(date +%T)"; break; }
  [ "$done" -gt "$DONE0" ] && { echo "BOOST EVAL: $done/4 suites done $(date +%T)"; break; }
  pgrep -fc run_boost_eval.sh >/dev/null 2>&1 || { echo "BOOST EVAL runner gone ($done/4) $(date +%T)"; break; }
  [ $(( $(date +%s)-t0 )) -ge $HB ] && { echo "BOOST EVAL hb ($done/4) $(date +%T)"; break; }
  sleep 60
done
