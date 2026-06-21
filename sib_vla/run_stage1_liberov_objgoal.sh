#!/bin/bash
# STAGE 1 — LIBERO-V robustness on Object + Goal (cross-suite generalization of
# the Spatial robustness story). 6 corruption axes x {baseline, aegis} x {object,
# goal} = 24 evals, n=200 each (20 trials x 10 tasks), n_action_steps=1.
# OOM-safe: parallelism sized to live free VRAM via run_queue_core.sh.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./run_queue_core.sh
PY=/home/user/anaconda3/bin/python
BASE=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-20}
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
  for c in $CONDS; do
    # baseline arm
    echo -e "$od/lv_baseline_${c}.log\t$PY scripts/eval_libero_v.py --config $cfg --method baseline $TE --n-action-steps 1 --episodes $EP --tasks 0,1,2,3,4,5,6,7,8,9 --only $c" >> "$JL"
    # aegis arm
    echo -e "$od/lv_aegis_${c}.log\t$PY scripts/eval_libero_v.py --config $cfg --method aegis --rib-weights $RIB --rasf-weights $RASF $TE --n-action-steps 1 --episodes $EP --tasks 0,1,2,3,4,5,6,7,8,9 --only $c" >> "$JL"
  done
done

run_stage "LIBERO-V object+goal" "$JL"
rm -f "$JL"

# summary
$PY - <<'PY'
import json,glob,os
for sname in ['object','goal']:
    print(f"=== {sname} ===")
    for m in ['baseline','aegis']:
        for f in sorted(glob.glob(f'results/liberov_objgoal/{sname}/libero_v/{m}/eval_*.json')):
            try:
                d=json.load(open(f)); print(f"  {m:8s} {d['condition']:18s} {d['success_rate']*100:5.1f}  n={d['n_episodes']}")
            except Exception: pass
PY
echo "[$(date +%T)] STAGE 1 COMPLETE"
