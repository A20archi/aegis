#!/bin/bash
# MULTI-SEED — re-run the headline LIBERO-V Object+Goal grid for extra seeds so we can
# report mean ± CI (seed 42 is the primary, already on disk). Uses eval's --seed override,
# which isolates each seed under results/.../seed<N>/ (resume-safe). OOM-safe via the queue.
#   SEEDS="123 456" EP=20 bash run_multiseed.sh
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./run_queue_core.sh
PY=${PY:-/home/user/anaconda3/bin/python}
BASE=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-20}
SEEDS=${SEEDS:-"123 456"}
CONDS="gaussian_noise_1 motion_blur_1 lighting_1 texture_1 viewpoint_medium viewpoint_large"
declare -A SUITE=( [object]=libero_object [goal]=libero_goal )
declare -A ELEN=( [object]=280 [goal]=300 )

JL=$(mktemp)
for sname in object goal; do
  od=results/liberov_objgoal/$sname
  cfg=configs/_lv_${sname}.yaml
  cat > "$cfg" <<YAML
inherit: base.yaml
checkpoint: $BASE
output_dir: $od
suite: ${SUITE[$sname]}
episode_length: ${ELEN[$sname]}
n_action_steps: 1
record:
  enabled: false
YAML
  for seed in $SEEDS; do
    for c in $CONDS; do
      echo -e "$od/seed${seed}_baseline_${c}.log\t$PY scripts/eval_libero_v.py --config $cfg --method baseline $TE --n-action-steps 1 --episodes $EP --tasks 0,1,2,3,4,5,6,7,8,9 --only $c --seed $seed" >> "$JL"
      echo -e "$od/seed${seed}_aegis_${c}.log\t$PY scripts/eval_libero_v.py --config $cfg --method aegis --rib-weights $RIB --rasf-weights $RASF $TE --n-action-steps 1 --episodes $EP --tasks 0,1,2,3,4,5,6,7,8,9 --only $c --seed $seed" >> "$JL"
    done
  done
done
run_stage "LIBERO-V multi-seed [$SEEDS]" "$JL"
rm -f "$JL"
echo "[$(date +%T)] MULTI-SEED COMPLETE — results under results/liberov_objgoal/*/seed<N>/"
