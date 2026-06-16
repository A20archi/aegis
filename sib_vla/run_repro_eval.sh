#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
for N in 10 1; do
  echo "==== eval n_action_steps=$N ===="
  python scripts/eval.py --config configs/repro_eval.yaml \
    --n-action-steps "$N" --tag "repro_n$N" 2>&1 | tee "results/repro/log_eval_n$N.txt"
done
echo "==== SUMMARY vs vanilla 72.0% ===="
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("results/repro/eval_repro_n*.json")):
    r=json.load(open(f)); p=r["success_rate"]; lo,hi=r["success_wilson95"]; n=r["n_episodes"]
    print(f"  {r['name']:10s} SR={p*100:5.1f}%  [{lo*100:.1f},{hi*100:.1f}]  n={n}  vs72: {(p-0.72)*100:+.1f}pp")
PY
