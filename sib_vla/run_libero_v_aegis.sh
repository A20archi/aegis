#!/bin/bash
# LIBERO-V robustness headline: AEGIS (RIB perception + RASF action + temporal
# ensembling) vs the SmolVLA+TE baseline (the 86% config), across the 4 axes.
#   baseline = SmolVLA + TE          (no RIB, no RASF)
#   aegis    = RIB + RASF + TE
# Both identical except the two AEGIS modules -> the gap is AEGIS's robustness.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
CFG=configs/ib_on86.yaml
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
EP=${1:-10}                       # episodes per task (x10 tasks per condition)
# 6 conditions spanning all 4 axes (viewpoint x2 incl. the hard 'large')
ONLY="viewpoint_medium,viewpoint_large,lighting_1,texture_1,motion_blur_1,gaussian_noise_1"
TE="--forge-ensemble --ensemble-coeff 0.01"

[ -f "$RIB" ] || { echo "FATAL: RIB checkpoint missing ($RIB) — train it first"; exit 1; }

REC="--record --videos-per-task 2"        # save 2 rollout videos/task/condition (per arm)

echo "######## BASELINE: SmolVLA + TE ########"
python scripts/eval_libero_v.py --config $CFG --method baseline $TE $REC \
  --n-action-steps 1 --episodes $EP --only "$ONLY" 2>&1 | tee results/ib_on86/log_libv_baseline.txt

echo "######## AEGIS: RIB + RASF + TE ########"
python scripts/eval_libero_v.py --config $CFG --method aegis \
  --rib-weights $RIB --rasf-weights $RASF $TE $REC \
  --n-action-steps 1 --episodes $EP --only "$ONLY" 2>&1 | tee results/ib_on86/log_libv_aegis.txt

echo "######## AEGIS vs BASELINE — robustness retention ########"
python3 -c "
import json, glob
def load(m):
    d={}
    for f in glob.glob(f'results/ib_on86/libero_v/{m}/eval_*.json'):
        r=json.load(open(f)); d[r['condition']]=(r['success_rate']*100, r['n_episodes'], r['success_wilson95'])
    return d
b=load('baseline'); a=load('aegis')
conds=sorted(c for c in b if c in a)
print(f'{\"condition\":18s} {\"baseline+TE\":>14} {\"AEGIS\":>14} {\"gain\":>8}')
gains=[]
for c in conds:
    bsr,_,bci=b[c]; asr,_,aci=a[c]; g=asr-bsr; gains.append(g)
    print(f'{c:18s} {bsr:6.1f}% [{bci[0]*100:3.0f},{bci[1]*100:3.0f}] {asr:6.1f}% [{aci[0]*100:3.0f},{aci[1]*100:3.0f}] {g:+7.1f}pp')
if gains:
    bm=sum(b[c][0] for c in conds)/len(conds); am=sum(a[c][0] for c in conds)/len(conds)
    print(f'{\"AVERAGE\":18s} {bm:6.1f}%          {am:6.1f}%          {am-bm:+7.1f}pp')
    print(f'\\n>>> AEGIS robustness retention gain = {am-bm:+.1f}pp averaged over {len(conds)} conditions / 4 axes')
"
echo "######## DONE ########"
