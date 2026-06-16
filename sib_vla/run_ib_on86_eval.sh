#!/bin/bash
# Eval the IB-Adapter (trained on the 79% generalist) two ways:
#   A) IB + temporal ensembling  -> "IB on top of the 86%" (full visual+action stack)
#   B) IB pure (n=1, no ensemble) -> isolates IB's own contribution vs 79%
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
CFG=configs/ib_on86.yaml
W=results/ib_on86/ib_on86.pt

echo "==== [A] IB-Adapter + temporal ensembling  (vs 86.0%) ===="
python scripts/eval.py --config $CFG --ib-weights $W --n-heads 8 \
  --n-action-steps 1 --forge-ensemble --ensemble-coeff 0.01 --lever3-clamp 1.0 \
  --tag ib_on86_tempens 2>&1 | tee results/ib_on86/log_eval_tempens.txt

echo "==== [B] IB-Adapter pure, n=1  (vs 79.0%) ===="
python scripts/eval.py --config $CFG --ib-weights $W --n-heads 8 \
  --n-action-steps 1 --tag ib_on86_pure 2>&1 | tee results/ib_on86/log_eval_pure.txt

echo "==== IB-ON-86 SUMMARY ===="
python3 -c "
import json, glob
ref={'ib_on86_tempens':('IB+TempEns','86.0% (TempEns only)'),'ib_on86_pure':('IB pure n=1','79.0% (generalist)')}
for f in sorted(glob.glob('results/ib_on86/eval_ib_on86_*.json')):
    r=json.load(open(f)); name=r['name']; p=r['success_rate']; lo,hi=r['success_wilson95']
    lbl,base=ref.get(name,(name,'?'))
    print(f'  {lbl:12s} SR={p*100:5.1f}%  [{lo*100:.1f},{hi*100:.1f}]  jerk={r[\"rms_jerk_mean\"]:.3f}   vs {base}')
"
