#!/usr/bin/env bash
# Train RIB on each frozen colleague-ACT suite, 2 concurrent. Identity-init, corruption-aug.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONPATH="$PWD:$(dirname $PWD)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/miniconda3/envs/lerobot/bin/python
STEPS=${STEPS:-6000}; CAP=${CAP:-2}
declare -A DS=( [Spatial]=libero_spatial_image [Object]=libero_object_image [Goal]=libero_goal_image [Long]=libero_10_image )
mkdir -p ../results/aegis_act_v2
pids=()
for S in Object Goal Spatial Long; do
  out=../results/aegis_act_v2/$S; mkdir -p "$out"
  echo "[launch] RIB train $S -> $out/rib.pt"
  $PY -u finetune_rib_act_v2.py --ckpt-dir ../act_ckpts/$S/act/30000 \
      --dataset lerobot/${DS[$S]} --out "$out/rib.pt" --steps $STEPS \
      --num-workers 5 --ckpt-every 3000 > "$out/train.log" 2>&1 &
  pids+=($!)
  # throttle to CAP concurrent
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do sleep 15; done
done
wait
echo "[done] all RIB training finished"
for S in Object Goal Spatial Long; do
  echo "=== $S ==="; grep -E "DONE|final_fusion" ../results/aegis_act_v2/$S/train.log | tail -1
done
