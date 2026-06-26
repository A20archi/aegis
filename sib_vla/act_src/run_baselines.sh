#!/usr/bin/env bash
# Competitor baselines on LIBERO-Plus: TTA + BN-adapt on frozen base, seed 42, all 4 suites.
# Compares Δ-over-base against AEGIS's +5.1. (RobustVLA/BYOVLA = cite-only, external code.)
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
HARDCAP=${HARDCAP:-4}; SEED=42; NPC=${NPC:-12}
JOBS=$(mktemp)
for bl in tta bn; do for S in Spatial Object Goal Long; do
  od=$SIB/results/act_baselines/$bl/$S
  echo -e "$od/run.log\t$PY -u $ACT/plus_eval_aegis.py --suite ${SU[$S]} --dataset lerobot/${DS[$S]} --base-ckpt $SIB/act_ckpts/$S/act/30000 --arm base --baseline $bl --seed $SEED --n-per-cat $NPC --max-steps ${PMS[$S]} --out $od" >> "$JOBS"
done; done
echo "[baselines] $(wc -l < "$JOBS") jobs (TTA+BN x 4 suites, seed42, HARDCAP=$HARDCAP)"
while IFS=$'\t' read -r log cmd; do
  mkdir -p "$(dirname "$log")"; ( eval "$cmd" > "$log" 2>&1; echo "[done] $log" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$HARDCAP" ]; do sleep 10; done
done < "$JOBS"; wait; rm -f "$JOBS"
echo "[baselines] DONE -> results/act_baselines/"
