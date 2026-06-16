#!/bin/bash
# Layer the SIB action bottleneck on the 86% generalist base, measure marginal gain.
#   Phase 1: estimate_lambda (cache chunks + per-band variance) ~45m
#   Phase 2: train SIB module (beta=1e-4) ~30m
#   Phase 3: eval SIB + temporal ensemble  (vs 86%)   n=1
#   Phase 4: eval SIB pure (no ensemble)    (vs 79%)   n=1
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
CFG=configs/sib_on86.yaml
W=results/sib_on86/sib_on86_b1e4.pt
mkdir -p results/sib_on86

echo "==== [1/4] estimate_lambda ===="
python scripts/estimate_lambda.py --config $CFG 2>&1 | tee results/sib_on86/log_lambda.txt

echo "==== [2/4] train SIB module (beta=1e-4) ===="
python scripts/train.py --config $CFG --beta 1e-4 --tag sib_on86_b1e4 2>&1 | tee results/sib_on86/log_train.txt

echo "==== [3/4] eval: SIB + temporal ensemble (compare to 86.0%) ===="
python scripts/eval.py --config $CFG --weights $W \
  --n-action-steps 1 --forge-ensemble --ensemble-coeff 0.01 --lever3-clamp 1.0 \
  --tag sib_on86_forge_n1 2>&1 | tee results/sib_on86/log_eval_forge.txt

echo "==== [4/4] eval: SIB pure, no ensemble (compare to 79.0%) ===="
python scripts/eval.py --config $CFG --weights $W \
  --n-action-steps 1 --tag sib_on86_pure_n1 2>&1 | tee results/sib_on86/log_eval_pure.txt

echo "==== SIB-ON-86 SUMMARY ===="
python3 -c "
import json, glob
ref={'sib_on86_forge_n1':('SIB+TempEns','86.0% (TempEns only)'),'sib_on86_pure_n1':('SIB pure','79.0% (no ens)')}
for f in sorted(glob.glob('results/sib_on86/eval_sib_on86_*_n1.json')):
    r=json.load(open(f)); name=r['name']; p=r['success_rate']; lo,hi=r['success_wilson95']
    lbl,base=ref.get(name,(name,'?'))
    print(f'  {lbl:14s} SR={p*100:5.1f}%  [{lo*100:.1f},{hi*100:.1f}]  jerk={r[\"rms_jerk_mean\"]:.3f}   vs {base}')
"
