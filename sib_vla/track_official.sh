#!/bin/bash
# Pings when all 4 official suites finish (or runner dies, or 30-min heartbeat).
set -uo pipefail; cd "$(dirname "$0")"
t0=$(date +%s); HB=${HB:-1800}
done_json(){ ls results/official_base/$1/libero_v/baseline/eval_clean.json 2>/dev/null; }
while true; do
  d=0; for s in spatial object goal long; do [ -f "$(done_json $s)" ] && d=$((d+1)); done
  runner=$(pgrep -fc 'run_official_parallel.sh' 2>/dev/null || echo 0)
  if [ "$d" -ge 4 ]; then echo "OFFICIAL ALL 4 DONE $(date +%T)"; break; fi
  if [ "$runner" -eq 0 ]; then echo "OFFICIAL runner gone ($d/4 done) $(date +%T)"; break; fi
  if [ $(( $(date +%s)-t0 )) -ge $HB ]; then echo "OFFICIAL heartbeat ($d/4) $(date +%T)"; break; fi
  sleep 45
done
python3 - <<'PY'
import json,os
tot=[]
print("suite     official  (n)")
for n in ['spatial','object','goal','long']:
    p=f'results/official_base/{n}/libero_v/baseline/eval_clean.json'
    if os.path.exists(p):
        d=json.load(open(p)); sr=d['success_rate']*100; tot.append((n,sr))
        print(f'  {n:8s} {sr:5.1f}   ({d.get("n_episodes")})')
    else:
        print(f'  {n:8s}   ...')
if len(tot)==4:
    avg=sum(s for _,s in tot)/4
    print(f'  AVERAGE  {avg:5.1f}   (paper 87.3 | our-repro 83.5)')
PY
