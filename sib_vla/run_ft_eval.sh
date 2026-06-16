#!/bin/bash
# Eval spatial-FT checkpoints on LIBERO-Spatial at n_action_steps=10.
# Usage: ./run_ft_eval.sh 4000 2000 3000 1000
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
FT=outputs/smolvla_spatial_ft/checkpoints
for STEP in "$@"; do
  CKPT="$FT/$(printf '%06d' "$STEP")/pretrained_model"
  echo "==== eval FT checkpoint step=$STEP  n=10 ===="
  python3 -c "import re,pathlib; p=pathlib.Path('configs/repro_eval.yaml'); t=p.read_text(); t=re.sub(r'^checkpoint:.*$','checkpoint: $CKPT',t,flags=re.M); p.write_text(t)"
  python scripts/eval.py --config configs/repro_eval.yaml \
    --n-action-steps 10 --tag "spatial_ft_${STEP}_n10" \
    2>&1 | tee "results/repro/log_ft_eval_${STEP}.txt"
done
echo "==== FT EVAL SUMMARY (vs generalist 79.0%, baseline 72.0%) ===="
python3 -c "
import json, glob
rows=[]
for f in sorted(glob.glob('results/repro/eval_spatial_ft_*_n10.json')):
    r=json.load(open(f)); rows.append((r['name'],r['success_rate'],r['success_wilson95'],r['n_episodes']))
for name,p,(lo,hi),n in rows:
    print(f'  {name:24s} SR={p*100:5.1f}%  [{lo*100:.1f},{hi*100:.1f}]  n={n}  vs79: {(p-0.79)*100:+.1f}pp')
"
