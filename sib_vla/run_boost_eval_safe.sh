#!/bin/bash
# OOM-PROOF boosted eval: runs suites SERIALLY, and waits for >= NEED MiB free GPU
# before starting each one. Cannot oversubscribe the GPU regardless of external jobs.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
BOOST=results/lora_boost/smolvla_lora_boost
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-50}; NEED=${NEED:-9000}    # require 9GB free before launching a suite
names=(long spatial object goal)
suites=(libero_10 libero_spatial libero_object libero_goal)
eplen=(520 220 280 300)
wait_mem(){
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge "$NEED" ] && { echo "[$(date +%T)] GPU free=${free}MiB >= ${NEED} -> go"; break; }
    echo "[$(date +%T)] GPU free=${free}MiB < ${NEED}, waiting..."; sleep 60
  done
}
for i in "${!names[@]}"; do
  name=${names[$i]}; od=results/boost_base/$name; mkdir -p "$od"
  [ -f "$od/libero_v/baseline/eval_clean.json" ] && { echo "skip $name (done)"; continue; }
  cat > configs/_boost_${name}.yaml <<YAML
inherit: base.yaml
checkpoint: $BOOST
output_dir: $od
suite: ${suites[$i]}
episode_length: ${eplen[$i]}
n_action_steps: 1
record:
  enabled: false
YAML
  wait_mem
  echo "[$(date +%T)] START $name (serial, OOM-guarded)"
  $PY scripts/eval_libero_v.py --config "configs/_boost_$name.yaml" --method baseline $TE \
    --n-action-steps 1 --episodes "$EP" --only clean >"$od/baseline.log" 2>&1
  sr=$($PY -c "import json;print('%.1f'%(json.load(open('$od/libero_v/baseline/eval_clean.json'))['success_rate']*100))" 2>/dev/null || echo ERR)
  echo "[$(date +%T)] DONE $name SR=$sr"
done
echo "[$(date +%T)] === BOOST 4-suite (OOM-guarded) complete ==="
$PY - <<'PY'
import json,os
old={'spatial':85.5,'object':97.5,'goal':93.5,'long':64.5}; tot=[]
for n in ['spatial','object','goal','long']:
    p=f'results/boost_base/{n}/libero_v/baseline/eval_clean.json'
    if os.path.exists(p):
        sr=json.load(open(p))['success_rate']*100; tot.append(sr); print(f'  {n:8s} {sr:5.1f} (was {old[n]}, {sr-old[n]:+.1f})')
if len(tot)==4: print(f'  AVG {sum(tot)/4:.2f} (was 85.25 | paper 87.3)')
PY
