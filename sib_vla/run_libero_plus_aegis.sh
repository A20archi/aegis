#!/bin/bash
# AEGIS vs baseline on LIBERO-Plus. Uses the isolated lerobot_lplus env + LIBERO-Plus config.
# Usage: METHOD=aegis SUITE=libero_object PERCAT=3 bash run_libero_plus_aegis.sh
set -uo pipefail
REPO=/home/user/Desktop/vla_projects/LIBERO-plus
CFG=/home/user/Desktop/vla_projects/.libero_lplus
ENV=/home/user/miniconda3/envs/lerobot_lplus
SIB=/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla
METHOD=${METHOD:-aegis}; SUITE=${SUITE:-libero_object}; PERCAT=${PERCAT:-3}
RIB=$SIB/results/ib_on86/rib_on86.pt
RASF=$SIB/results/rasf_on86/rasf_on86.pt
CKPT=${CKPT:-outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model}
EXTRA=""; [ "$METHOD" = aegis ] && EXTRA="--rib-weights $RIB --rasf-weights $RASF"
cd "$SIB"
env PYTHONPATH=$REPO:$SIB LIBERO_CONFIG_PATH=$CFG MAGICK_HOME=$ENV LD_LIBRARY_PATH=$ENV/lib \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  $ENV/bin/python scripts/libero_plus_aegis_eval.py \
    --method $METHOD --ckpt "$CKPT" $EXTRA --suite $SUITE --per-cat $PERCAT "${@}"
