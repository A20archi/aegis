#!/bin/bash
# Eval the LoRA-boosted base on all 4 suites (baseline+TE, n=1, n=200, per-suite max-steps).
# Parallel CONC, Long-first. Restartable. Run AFTER finetune merges its checkpoint.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
BOOST=results/lora_boost/smolvla_lora_boost
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-50}; CONC=${CONC:-4}
names=(long spatial object goal)
suites=(libero_10 libero_spatial libero_object libero_goal)
eplen=(520 220 280 300)
for i in "${!names[@]}"; do
  od=results/boost_base/${names[$i]}; mkdir -p "$od"
  cat > configs/_boost_${names[$i]}.yaml <<YAML
inherit: base.yaml
checkpoint: $BOOST
output_dir: $od
suite: ${suites[$i]}
episode_length: ${eplen[$i]}
n_action_steps: 1
record:
  enabled: false
YAML
done
run_one(){ local name=$1 od=results/boost_base/$1
  [ -f "$od/libero_v/baseline/eval_clean.json" ] && { echo "skip $name"; return; }
  echo "[$(date +%T)] START $name"
  $PY scripts/eval_libero_v.py --config "configs/_boost_$name.yaml" --method baseline $TE \
    --n-action-steps 1 --episodes "$EP" --only clean >"$od/baseline.log" 2>&1
  echo "[$(date +%T)] DONE $name SR=$($PY -c "import json;print('%.1f'%(json.load(open('$od/libero_v/baseline/eval_clean.json'))['success_rate']*100))" 2>/dev/null)"
}
n=0; for name in "${names[@]}"; do run_one "$name" & n=$((n+1)); [ $((n%CONC)) -eq 0 ] && wait -n 2>/dev/null; done; wait
echo "=== BOOST 4-suite vs 85.25 base ==="
$PY - <<'PY'
import json,os
old={'spatial':85.5,'object':97.5,'goal':93.5,'long':64.5}; tot=[]
for n in ['spatial','object','goal','long']:
    p=f'results/boost_base/{n}/libero_v/baseline/eval_clean.json'
    if os.path.exists(p):
        sr=json.load(open(p))['success_rate']*100; tot.append(sr)
        print(f'  {n:8s} {sr:5.1f}  (was {old[n]}, {sr-old[n]:+.1f})')
if len(tot)==4: print(f'  AVG {sum(tot)/4:.1f}  (was 85.25 | paper 87.3 | goal 90)')
PY
