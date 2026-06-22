#!/bin/bash
# RASF ROBUSTNESS SWEEP -- the headline result vs the 99.6% clean-SR papers.
#
# Story: every LIBERO number (incl. VLA-Adapter 99.6%) assumes a PERFECT actuator.
# Under realistic action-execution noise, SR degrades. RASF is a denoising-trained,
# identity-at-init action filter -> it should RETAIN SR where the bare policy collapses.
#
# Apples-to-apples, RASF isolated (no temporal ensembling, n=1):
#   arm BASE  = bare policy          (omit --weights -> identity module)
#   arm RASF  = policy + RASF filter (--weights rasf_on86.pt)
# Noise is injected PRE-filter (sib.corruptions.perturb_actions), so RASF must clean it.
#
# Metric per arm: SR(noise) and retention R = SR(noise)/SR(clean). RASF wins on R.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
CFG=configs/rasf_on86.yaml
W=results/rasf_on86/rasf_on86.pt
OUT=results/rasf_on86
LEVELS="0.0 0.10 0.20 0.30"

echo "######## RASF ROBUSTNESS SWEEP (action-execution noise) ########"
for nz in $LEVELS; do
  tag_nz=$(echo "$nz" | tr -d '.')
  echo "==== noise=$nz  ARM=BASE (bare policy) ===="
  python scripts/eval.py --config $CFG --n-action-steps 1 --action-noise $nz \
    --tag rob_base_n${tag_nz} 2>&1 | tee $OUT/log_rob_base_n${tag_nz}.txt

  echo "==== noise=$nz  ARM=RASF (+ filter) ===="
  python scripts/eval.py --config $CFG --weights $W --n-action-steps 1 --action-noise $nz \
    --tag rob_rasf_n${tag_nz} 2>&1 | tee $OUT/log_rob_rasf_n${tag_nz}.txt
done

echo "==== ROBUSTNESS SUMMARY (retention curve) ===="
python3 -c "
import json, glob, re
rows={}  # (arm,noise) -> (sr, lo, hi)
for f in sorted(glob.glob('$OUT/eval_rob_*.json')):
    r=json.load(open(f)); m=re.match(r'rob_(base|rasf)_n(\d+)', r['name'])
    if not m: continue
    arm=m.group(1); nz=int(m.group(2))/100.0
    rows[(arm,nz)]=(r['success_rate'], *r['success_wilson95'])
levels=sorted({k[1] for k in rows})
base0=rows.get(('base',0.0),(None,))[0]; rasf0=rows.get(('rasf',0.0),(None,))[0]
print(f'{\"noise\":>6} | {\"BASE SR\":>16} {\"ret\":>5} | {\"RASF SR\":>16} {\"ret\":>5} | {\"Δ(RASF-BASE)\":>12}')
for nz in levels:
    b=rows.get(('base',nz)); s=rows.get(('rasf',nz))
    if not b or not s: continue
    br = b[0]/base0 if base0 else float('nan')
    sr_ = s[0]/rasf0 if rasf0 else float('nan')
    bstr=f'{b[0]*100:5.1f}% [{b[1]*100:4.1f},{b[2]*100:4.1f}]'
    sstr=f'{s[0]*100:5.1f}% [{s[1]*100:4.1f},{s[2]*100:4.1f}]'
    print(f'{nz:6.2f} | {bstr:>16} {br*100:4.0f}% | {sstr:>16} {sr_*100:4.0f}% | {(s[0]-b[0])*100:+11.1f}pp')
print()
print('Headline = the gap Δ at high noise: bare policy collapses, RASF retains.')
"
echo "######## RASF ROBUSTNESS DONE ########"
