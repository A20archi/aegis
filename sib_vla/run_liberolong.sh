#!/bin/bash
# LIBERO-LONG robustness — the SAME 6 LIBERO-V corruption axes × {baseline, aegis} on the
# long-horizon suite libero_10 (episode_length 520; truncating to 300 silently kills long SR).
# 12 evals, n=200 (20 trials × 10 tasks), n_action_steps=1. OOM-safe via run_queue_core.sh.
#   EP=20 bash run_liberolong.sh
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
SEEDS=${SEEDS:-""}                       # set e.g. "123 456" for multi-seed long; "" = primary
CONDS="gaussian_noise_1 motion_blur_1 lighting_1 texture_1 viewpoint_medium viewpoint_large"

od=results/liberov_long
cfg=configs/_lv_long.yaml
cat > "$cfg" <<YAML
inherit: base.yaml
checkpoint: $BASE
output_dir: $od
suite: libero_10
episode_length: 520
n_action_steps: 1
record:
  enabled: false
YAML

JL=$(mktemp)
emit(){  # $1=arm-extra-args  $2=tag  $3=seedflag
  local args="$1" tag="$2" sflag="$3"
  for c in $CONDS; do
    echo -e "$od/${tag}_${c}.log\t$PY scripts/eval_libero_v.py --config $cfg $args $TE --n-action-steps 1 --episodes $EP --tasks 0,1,2,3,4,5,6,7,8,9 --only $c $sflag" >> "$JL"
  done
}
if [ -z "$SEEDS" ]; then
  emit "--method baseline" "baseline" ""
  emit "--method aegis --rib-weights $RIB --rasf-weights $RASF" "aegis" ""
else
  for s in $SEEDS; do
    emit "--method baseline" "seed${s}_baseline" "--seed $s"
    emit "--method aegis --rib-weights $RIB --rasf-weights $RASF" "seed${s}_aegis" "--seed $s"
  done
fi

run_stage "LIBERO-Long robustness (libero_10, 520-step)" "$JL"
rm -f "$JL"

$PY - <<'PY'
import json,glob
print("=== LIBERO-Long (libero_10) ===")
for m in ['baseline','aegis']:
    for f in sorted(glob.glob(f'results/liberov_long/libero_v/{m}/eval_*.json')):
        try:
            d=json.load(open(f)); print(f"  {m:8s} {d['condition']:18s} {d['success_rate']*100:5.1f}  n={d['n_episodes']}")
        except Exception: pass
PY
echo "[$(date +%T)] LIBERO-LONG COMPLETE — results/liberov_long/"
