#!/bin/bash
# LIBERO-V robustness headline, PARALLEL (box is free) — Spatial, proper paper protocol:
# n_action_steps=1, episode_length=220, n=200/condition (20/task x10), base+TE vs AEGIS(RIB+RASF+TE).
# Win-axes first (systematic visual shifts where RIB helps); stochastic noise last (TE's job, control).
# 6 conditions x 2 arms = 12 jobs at CONC concurrency. Restartable (skips done condition/arm).
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
CFG=configs/_robust_spatial.yaml
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-20}
CONC=${CONC:-4}
OD=results/robust_spatial
# win-axes first: viewpoint/lighting/texture (RIB shines) -> blur/noise controls last
CONDS=${CONDS:-"viewpoint_medium lighting_1 texture_1 viewpoint_large motion_blur_1 gaussian_noise_1"}

run_one(){
  local cond=$1 arm=$2
  local jf="$OD/libero_v/$arm/eval_${cond}.json"
  [ -f "$jf" ] && { echo "[$(date +%T)] skip $cond/$arm (done)"; return; }
  echo "[$(date +%T)] START $cond/$arm"
  if [ "$arm" = baseline ]; then
    $PY scripts/eval_libero_v.py --config "$CFG" --method baseline $TE \
      --n-action-steps 1 --episodes "$EP" --only "$cond" --record --videos-per-task 2 \
      >"$OD/${cond}_${arm}.log" 2>&1
  else
    $PY scripts/eval_libero_v.py --config "$CFG" --method aegis \
      --rib-weights "$RIB" --rasf-weights "$RASF" $TE \
      --n-action-steps 1 --episodes "$EP" --only "$cond" --record --videos-per-task 2 \
      >"$OD/${cond}_${arm}.log" 2>&1
  fi
  echo "[$(date +%T)] DONE  $cond/$arm"
}

mkdir -p "$OD/libero_v/baseline" "$OD/libero_v/aegis"
echo "[$(date +%T)] === robustness parallel: EP=$EP CONC=$CONC | conds: $CONDS ==="
n=0
for cond in $CONDS; do for arm in baseline aegis; do
  run_one "$cond" "$arm" &
  n=$((n+1)); [ $((n % CONC)) -eq 0 ] && wait -n 2>/dev/null
done; done
wait
echo "[$(date +%T)] === ALL ROBUSTNESS DONE ==="
