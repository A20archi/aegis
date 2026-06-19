#!/bin/bash
# Tracker for the robustness headline. Writes results/robust_spatial/ROBUST.md and EXITS
# (pings agent) on each new condition/arm completion, or 30-min heartbeat, or all 12 done.
# Re-arm: DONE0=<current_done_count> nohup bash track_robust.sh &
set -uo pipefail
cd "$(dirname "$0")"
DONE0=${DONE0:-0}; HEARTBEAT=${HEARTBEAT:-1800}; OD=results/robust_spatial
t0=$(date +%s)

write(){
python3 - > "$OD/ROBUST.md" 2>/dev/null <<'PY'
import json,os,glob,time
OD='results/robust_spatial'
conds=['viewpoint_medium','lighting_1','texture_1','viewpoint_large','motion_blur_1','gaussian_noise_1']
WIN={'viewpoint_medium':'systematic (RIB)','lighting_1':'systematic (RIB)','texture_1':'systematic (RIB)',
     'viewpoint_large':'systematic (RIB)','motion_blur_1':'blur (RASF-blind)','gaussian_noise_1':'stochastic (TE ctrl)'}
def sr(arm,c):
    p=f'{OD}/libero_v/{arm}/eval_{c}.json'
    if os.path.exists(p):
        d=json.load(open(p)); ci=d.get('success_wilson95',[None,None])
        return d['success_rate']*100,d.get('n_episodes'),ci
    return None,None,None
print('# LIBERO-V ROBUSTNESS — Spatial, n=1, ep=220, n=200/cond  (base+TE vs AEGIS)\n')
print(f'_updated {time.strftime("%Y-%m-%d %H:%M:%S")}_\n')
print('| condition | type | base+TE | AEGIS | Δ | status |')
print('|---|---|---|---|---|---|')
wins=0;done=0
for c in conds:
    b,bn,_=sr('baseline',c); a,an,_=sr('aegis',c)
    bs=f'{b:.1f}' if b is not None else '…'; as_=f'{a:.1f}' if a is not None else '…'
    if b is not None and a is not None:
        d=a-b; dl=f'**{d:+.1f}**'; st='✅'; done+=1
        if d>0: wins+=1
    else: dl='—'; st='▶' if (b is not None or a is not None) else '⏳'
    print(f'| {c} | {WIN[c]} | {bs} | {as_} | {dl} | {st} |')
print(f'\n**{done}/6 conditions complete · AEGIS wins {wins}/{done}**')
PY
}

while true; do
  write
  done=$(ls $OD/libero_v/*/eval_*.json 2>/dev/null | grep -v eval_clean | wc -l)
  # stall = the RUNNER SCRIPT itself is gone (not just GPU procs, which blip between conditions)
  runner=$(pgrep -fc 'run_robust_parallel.sh' 2>/dev/null || echo 0)
  if [ "$done" -ge 12 ]; then echo "ROBUST: ALL 12 DONE $(date +%T)"; cat "$OD/ROBUST.md"; break; fi
  if [ "$done" -gt "$DONE0" ]; then echo "ROBUST: new completion -> $done/12 $(date +%T)"; cat "$OD/ROBUST.md"; break; fi
  if [ "$runner" -eq 0 ] && [ "$done" -lt 12 ]; then echo "ROBUST: WARNING runner gone, $done/12 (stall?) $(date +%T)"; cat "$OD/ROBUST.md"; break; fi
  if [ $(( $(date +%s)-t0 )) -ge $HEARTBEAT ]; then echo "ROBUST heartbeat ($done/12) $(date +%T)"; cat "$OD/ROBUST.md"; break; fi
  sleep 60
done
