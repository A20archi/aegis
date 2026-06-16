#!/bin/bash
# Auto-tracker for the 4-suite eval. Writes results/allsuites/TRACKER.md live and EXITS
# (pinging the agent) whenever a new suite-arm completes, or every 40 min, or when all 8 done.
# Re-arm with: BASELINE=<current_done_count> nohup bash track_suites.sh &
set -uo pipefail
cd "$(dirname "$0")"
BASELINE=${BASELINE:-0}
HEARTBEAT=${HEARTBEAT:-2400}
TRACK=results/allsuites/TRACKER.md
t0=$(date +%s)

write_tracker(){
python3 - > "$TRACK" 2>/dev/null <<'PY'
import json,os,time
order=['long','spatial','object','goal']
eplen={'long':520,'spatial':220,'object':280,'goal':300}
def sr(n,a):
    p=f'results/allsuites/{n}/libero_v/{a}/eval_clean.json'
    if os.path.exists(p):
        d=json.load(open(p)); ci=d.get('success_wilson95',[None,None])
        return d['success_rate']*100, d.get('n_episodes'), ci
    return None,None,None
print('# 4-SUITE TRACKER — 86 base, n=1+TE, PROPER LIBERO protocol\n')
print(f'_updated {time.strftime("%Y-%m-%d %H:%M:%S")}_  (Long = KEY suite)\n')
print('| suite | max-steps | base+TE | AEGIS | status |')
print('|---|---|---|---|---|')
bv=[];av=[]
for n in order:
    b,bn,_=sr(n,'baseline'); a,an,_=sr(n,'aegis')
    if b is not None: bv.append(b)
    if a is not None: av.append(a)
    bs=f'**{b:.1f}** (n={bn})' if b is not None else '…'
    az=f'**{a:.1f}** (n={an})' if a is not None else '…'
    st='✅ done' if (b is not None and a is not None) else ('▶ running' if (b is not None or a is not None) else '⏳ queued')
    print(f'| {n}{" 🔑" if n=="long" else ""} | {eplen[n]} | {bs} | {az} | {st} |')
mean=lambda v: (sum(v)/len(v)) if v else float("nan")
print(f'\n**Running avg** — base+TE {mean(bv):.1f} · AEGIS {mean(av):.1f}   (paper target **87.3**)')
print(f'\nProgress: base+TE {len(bv)}/4 · AEGIS {len(av)}/4 suites')
PY
}

while true; do
  write_tracker
  done=$(ls results/allsuites/*/libero_v/*/eval_clean.json 2>/dev/null | wc -l)
  if [ "$done" -gt "$BASELINE" ]; then echo "TRACKER: new completion -> $done/8 at $(date +%T)"; cat "$TRACK"; break; fi
  if [ "$done" -ge 8 ]; then echo "TRACKER: ALL 8 DONE at $(date +%T)"; cat "$TRACK"; break; fi
  if [ $(( $(date +%s)-t0 )) -ge $HEARTBEAT ]; then echo "TRACKER heartbeat ($done/8) at $(date +%T)"; cat "$TRACK"; break; fi
  sleep 120
done
