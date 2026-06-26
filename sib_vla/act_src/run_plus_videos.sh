#!/usr/bin/env bash
# Paired base-vs-AEGIS perturbed rollout videos: 1 task/category x 7 cats x 4 suites x 2 arms.
# Runs AFTER the main sweep to avoid GPU contention. seed 42.
set -uo pipefail
cd "$(dirname "$0")"; ACT=$PWD; SIB=$(dirname "$ACT")
LROOT=/home/user/Desktop/vla_projects/LIBERO-plus; LPENV=/home/user/miniconda3/envs/lerobot_lplus
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=0 \
  PYTHONPATH="$LROOT:$ACT:$SIB" LIBERO_CONFIG_PATH=/home/user/Desktop/vla_projects/.libero_lplus \
  MAGICK_HOME=$LPENV LD_LIBRARY_PATH=$LPENV/lib
PY=$LPENV/bin/python
declare -A DS=( [Spatial]=libero_spatial_image [Object]=libero_object_image [Goal]=libero_goal_image [Long]=libero_10_image )
declare -A SU=( [Spatial]=libero_spatial [Object]=libero_object [Goal]=libero_goal [Long]=libero_10 )
declare -A PMS=( [Spatial]=300 [Object]=300 [Goal]=300 [Long]=520 )
VID=$SIB/results/act_plus_v2/_videos
mkdir -p "$VID"
for S in Spatial Object Goal Long; do for arm in base aegis; do
  rw=""; [ "$arm" = "aegis" ] && rw="--rib-weights $SIB/results/aegis_act_v2/$S/rib.pt"
  echo "[vid] $S $arm"
  $PY -u $ACT/plus_eval_aegis.py --suite ${SU[$S]} --dataset lerobot/${DS[$S]} \
    --base-ckpt $SIB/act_ckpts/$S/act/30000 --arm $arm $rw --seed 42 --n-per-cat 1 \
    --max-steps ${PMS[$S]} --record-dir $VID/$S --videos-per-cat 1 \
    --out $SIB/results/act_plus_v2/_vidtmp/$S > $VID/${S}_${arm}.log 2>&1
done; done
echo "[vid] DONE -> $VID  ($(find $VID -name '*.mp4'|wc -l) clips)"
