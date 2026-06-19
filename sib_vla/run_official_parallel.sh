#!/bin/bash
# OFFICIAL checkpoint, paper-faithful, all 4 suites in PARALLEL (Long-first so total
# wall-clock ~= the longest suite). baseline+TE, n_action_steps=1, n=200, per-suite max-steps.
# Restartable: skips any suite whose eval_clean.json already exists.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
OFFICIAL=/home/user/Desktop/aryan/aryan/ckpt_official_smolvla_libero
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-50}                 # capped to n_envs(20) => n=200/suite
CONC=${CONC:-4}

# Long-first (longest, 520 steps) so it launches in wave 1 and bounds total time.
names=(long spatial object goal)
suites=(libero_10 libero_spatial libero_object libero_goal)
eplen=(520 220 280 300)

for i in "${!names[@]}"; do
  od=results/official_base/${names[$i]}; mkdir -p "$od"
  cat > configs/_official_${names[$i]}.yaml <<EOF
inherit: base.yaml
checkpoint: $OFFICIAL
output_dir: $od
suite: ${suites[$i]}
episode_length: ${eplen[$i]}
n_action_steps: 1
record:
  enabled: false
EOF
done

run_one(){
  local name=$1 cfg=configs/_official_$1.yaml od=results/official_base/$1
  [ -f "$od/libero_v/baseline/eval_clean.json" ] && { echo "[$(date +%T)] skip $name (done)"; return; }
  echo "[$(date +%T)] START $name"
  $PY scripts/eval_libero_v.py --config "$cfg" --method baseline $TE \
    --n-action-steps 1 --episodes "$EP" --only clean >"$od/baseline.log" 2>&1
  local sr=$($PY -c "import json;print('%.1f'%(json.load(open('$od/libero_v/baseline/eval_clean.json'))['success_rate']*100))" 2>/dev/null || echo ERR)
  echo "[$(date +%T)] DONE  $name  SR=$sr"
}

echo "[$(date +%T)] === official paper-faithful 4-suite: EP=$EP CONC=$CONC (baseline+TE) ==="
n=0
for name in "${names[@]}"; do
  run_one "$name" &
  n=$((n+1)); [ $((n % CONC)) -eq 0 ] && wait -n 2>/dev/null
done
wait
echo "[$(date +%T)] === OFFICIAL 4-SUITE DONE -> summary ==="
$PY - <<'PY'
import json,os
tot=[]
for n in ['spatial','object','goal','long']:
    p=f'results/official_base/{n}/libero_v/baseline/eval_clean.json'
    if os.path.exists(p):
        d=json.load(open(p)); sr=d['success_rate']*100; tot.append(sr)
        print(f'  {n:8s} {sr:5.1f}  (n={d.get("n_episodes")})')
if len(tot)==4: print(f'  AVERAGE  {sum(tot)/4:5.1f}   (paper 87.3 | our-repro 83.5)')
PY
