#!/bin/bash
# Tracker for the gaussian-noise degradation sweep. base+TE vs AEGIS across sigma.
# Exits (pings) on each new completion / 30-min heartbeat / all done / runner gone.
set -uo pipefail
cd "$(dirname "$0")"
DONE0=${DONE0:-0}; HEARTBEAT=${HEARTBEAT:-1800}; OD=results/robust_spatial
t0=$(date +%s)
# levels we expect this sweep to fill (0.12 already done from headline)
LEVELS="0 1 2 3 4 7"
write(){
python3 - > "$OD/NOISE.md" 2>/dev/null <<'PY'
import json,os,time
OD='results/robust_spatial'
sig={'0':0.05,'1':0.12,'2':0.20,'3':0.30,'4':0.50,'7':1.00}
order=['0','1','2','3','4','7']
def sr(arm,l):
    p=f'{OD}/libero_v/{arm}/eval_gaussian_noise_{l}.json'
    if os.path.exists(p):
        d=json.load(open(p)); ci=[round(x*100) for x in d.get('success_wilson95',[0,0])]
        return d['success_rate']*100, ci
    return None,None
print('# Gaussian-noise degradation sweep — Spatial, n=200/level (base+TE vs AEGIS)\n')
print(f'_updated {time.strftime("%H:%M:%S")}_\n')
print('| sigma | base+TE | AEGIS | Δ |')
print('|---|---|---|---|')
for l in order:
    b,bc=sr('baseline',l); a,ac=sr('aegis',l)
    bs=f'{b:.1f}' if b is not None else '…'; as_=f'{a:.1f}' if a is not None else '…'
    d=f'**{a-b:+.1f}**' if (b is not None and a is not None) else '—'
    print(f'| {sig[l]:.2f} | {bs} | {as_} | {d} |')
PY
}
while true; do
  write
  done=$(for l in $LEVELS; do for arm in baseline aegis; do [ -f "$OD/libero_v/$arm/eval_gaussian_noise_${l}.json" ] && echo x; done; done | wc -l)
  runner=$(pgrep -fc 'run_robust_parallel.sh' 2>/dev/null || echo 0)
  if [ "$done" -ge 12 ]; then echo "NOISE: ALL DONE $(date +%T)"; cat "$OD/NOISE.md"; break; fi
  if [ "$done" -gt "$DONE0" ]; then echo "NOISE: new completion -> $done $(date +%T)"; cat "$OD/NOISE.md"; break; fi
  if [ "$runner" -eq 0 ] && [ "$done" -lt 12 ]; then echo "NOISE: runner gone ($done done) $(date +%T)"; cat "$OD/NOISE.md"; break; fi
  if [ $(( $(date +%s)-t0 )) -ge $HEARTBEAT ]; then echo "NOISE heartbeat ($done) $(date +%T)"; cat "$OD/NOISE.md"; break; fi
  sleep 60
done
