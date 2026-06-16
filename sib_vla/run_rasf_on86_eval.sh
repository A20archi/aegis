#!/bin/bash
# Eval the redesigned RASF filter on the generalist, two ways:
#   A) RASF + temporal ensembling -> full action stack, vs 86%
#   B) RASF pure (n=1)            -> RASF's own contribution, vs 79%
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
CFG=configs/rasf_on86.yaml
W=results/rasf_on86/rasf_on86.pt

echo "==== [A] RASF + temporal ensembling  (vs 86.0%) ===="
python scripts/eval.py --config $CFG --weights $W \
  --n-action-steps 1 --forge-ensemble --ensemble-coeff 0.01 --lever3-clamp 1.0 \
  --tag rasf_on86_tempens 2>&1 | tee results/rasf_on86/log_eval_tempens.txt

echo "==== [B] RASF pure, n=1  (vs 79.0%) ===="
python scripts/eval.py --config $CFG --weights $W \
  --n-action-steps 1 --tag rasf_on86_pure 2>&1 | tee results/rasf_on86/log_eval_pure.txt

echo "==== RASF-ON-86 SUMMARY ===="
python3 -c "
import json, glob
ref={'rasf_on86_tempens':('RASF+TempEns','86.0%'),'rasf_on86_pure':('RASF pure n=1','79.0%')}
for f in sorted(glob.glob('results/rasf_on86/eval_rasf_on86_*.json')):
    r=json.load(open(f)); name=r['name']; p=r['success_rate']; lo,hi=r['success_wilson95']
    lbl,base=ref.get(name,(name,'?'))
    print(f'  {lbl:14s} SR={p*100:5.1f}%  [{lo*100:.1f},{hi*100:.1f}]  jerk={r[\"rms_jerk_mean\"]:.3f}   vs {base}')
"
