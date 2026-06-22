#!/usr/bin/env bash
# LIBERO-Plus-native PAIRWISE videos on the LOCAL A100 (no Modal needed).
# Records baseline AND aegis on the SAME suite/seed -> matched side-by-side clips per
# perturbation category. EVAL-ONLY (reuses existing checkpoints; no training).
#
# Local env is already complete (verified): LIBERO-plus repo + 6.4GB assets present,
# lerobot 0.4.3 + CUDA, imageio-ffmpeg, checkpoints. Just run this.
#
#   SUITE=libero_object SEED=42 VPC=1 bash run_liberoplus_video_local.sh
#   CATS="Camera Light" bash run_liberoplus_video_local.sh     # only some categories
#
# Polite to hopfield/hamlet: runs niced, the two arms SEQUENTIALLY (not parallel), small
# VRAM. It DOES share GPU compute, so run when you're OK with that (or the A100 is free).
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_REPO=${LIBERO_PLUS_REPO:-/home/user/Desktop/vla_projects/LIBERO-plus}
export PYTHONPATH="$(pwd):$LIBERO_PLUS_REPO:${PYTHONPATH:-}"
PY=${PY:-/home/user/anaconda3/bin/python}

CKPT=${CKPT:-outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model}
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
SUITE=${SUITE:-libero_object}
SEED=${SEED:-42}
PERCAT=${PERCAT:-1}            # tasks/category to run; recorded ones are the contrast clips
VPC=${VPC:-1}                  # videos recorded per category
declare -A MS=( [libero_spatial]=220 [libero_object]=280 [libero_goal]=300 [libero_10]=520 )
MAXSTEPS=${MS[$SUITE]:-280}
RECDIR=results/liberoplus_video/$SUITE/videos
CATFLAG=(); [ -n "${CATS:-}" ] && CATFLAG=(--cats $CATS)

echo "[lpv $(date +%T)] suite=$SUITE seed=$SEED per_cat=$PERCAT vids/cat=$VPC -> $RECDIR"
echo "[lpv] sanity: assets present? $( [ -f $LIBERO_PLUS_REPO/libero/libero/assets/scenes/libero_floor_base_style.xml ] && echo YES || echo 'NO -> see README assets.zip step' )"

run_arm(){  # $1=method  $2=extra-args
  local m="$1"; shift
  echo "[lpv $(date +%T)] === $m ==="
  nice -n 15 $PY scripts/libero_plus_aegis_eval.py \
    --method "$m" --ckpt "$CKPT" --suite "$SUITE" --per-cat "$PERCAT" --seed "$SEED" \
    --max-steps "$MAXSTEPS" --record-dir "$RECDIR" --videos-per-cat "$VPC" \
    --out "$RECDIR/${m}_metrics.json" "${CATFLAG[@]}" "$@"
}

# SAME seed across both arms => init states match => true pairwise side-by-sides.
run_arm baseline
run_arm aegis --rib-weights "$RIB" --rasf-weights "$RASF"

echo "[lpv $(date +%T)] DONE. mp4s under $RECDIR/<category>/{baseline,aegis}_task<id>.mp4"
ls -R "$RECDIR" 2>/dev/null | grep -E "\.mp4$|:" | head -40
