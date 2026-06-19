#!/bin/bash
# DECISIVE TEST: evaluate the OFFICIAL paper checkpoint (HuggingFaceVLA/smolvla_libero)
# that we have on disk but never evaluated. All our numbers used our OWN repro
# (smolvla_spatial_repro), which under-reproduces. This tells us the true base.
#   baseline+TE, n_action_steps=1 (matches paper config), per-suite max-steps.
# SUITES env: space-sep names (default all 4). EP: trials/task (capped to 20 => n=200).
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
OFFICIAL=/home/user/Desktop/aryan/aryan/ckpt_official_smolvla_libero
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-50}
SUITES=${SUITES:-"spatial long object goal"}

declare -A SU=( [spatial]=libero_spatial [long]=libero_10 [object]=libero_object [goal]=libero_goal )
declare -A EL=( [spatial]=220 [long]=520 [object]=280 [goal]=300 )

for name in $SUITES; do
  od=results/official_base/${name}; mkdir -p "$od"
  if [ -f "$od/libero_v/baseline/eval_clean.json" ]; then echo "[$(date +%T)] skip $name (done)"; continue; fi
  cat > configs/_official_${name}.yaml <<EOF
inherit: base.yaml
checkpoint: $OFFICIAL
output_dir: $od
suite: ${SU[$name]}
episode_length: ${EL[$name]}
n_action_steps: 1
record:
  enabled: false
EOF
  echo "[$(date +%T)] START official/$name (EP=$EP)"
  $PY scripts/eval_libero_v.py --config "configs/_official_${name}.yaml" --method baseline $TE \
    --n-action-steps 1 --episodes "$EP" --only clean >"$od/baseline.log" 2>&1
  sr=$($PY -c "import json;print(json.load(open('$od/libero_v/baseline/eval_clean.json'))['success_rate'])" 2>/dev/null || echo "ERR")
  echo "[$(date +%T)] DONE  official/$name  SR=$sr"
done
echo "[$(date +%T)] === official-base summary ==="
$PY - <<'PY'
import json,os
for n in ['spatial','long','object','goal']:
    p=f'results/official_base/{n}/libero_v/baseline/eval_clean.json'
    if os.path.exists(p):
        d=json.load(open(p)); print(f'  {n:8s} {d["success_rate"]*100:5.1f}  (n={d.get("n_episodes")})')
PY
