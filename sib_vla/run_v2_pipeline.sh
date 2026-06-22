#!/bin/bash
# v2-base -> AEGIS@92 pipeline. Generates configs on the fly from a checkpoint path,
# then runs each stage. Use as the stronger base (outputs/smolvla_spatial_v2) lands.
#
#   ./run_v2_pipeline.sh evalbase <ckpt_dir>   # base+TE clean SR (peak-pick / Tue early-read)
#   ./run_v2_pipeline.sh modules  <ckpt_dir>   # retrain RIB + RASF ON the new base
#   ./run_v2_pipeline.sh aegis    <ckpt_dir>   # AEGIS clean (new base + new RIB+RASF + TE)
#   ./run_v2_pipeline.sh all      <ckpt_dir>   # evalbase -> modules -> aegis
#
# <ckpt_dir> e.g. outputs/smolvla_spatial_v2/checkpoints/last/pretrained_model
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
STAGE="${1:?stage: evalbase|modules|aegis|all}"
CKPT="${2:?path to v2 pretrained_model dir}"
[ -d "$CKPT" ] || { echo "FATAL: ckpt dir not found: $CKPT"; exit 1; }

TAG=v2
RIBCFG=configs/_rib_${TAG}.yaml      # also used for base-clean eval (same checkpoint)
RASFCFG=configs/_rasf_${TAG}.yaml
RIBOUT=results/ib_on_${TAG}
RASFOUT=results/rasf_on_${TAG}
RIBPT=$RIBOUT/rib_${TAG}.pt
RASFPT=$RASFOUT/rasf_${TAG}.pt
TE="--forge-ensemble --ensemble-coeff 0.01"

gen_configs() {
  mkdir -p $RIBOUT $RASFOUT
  cat > $RIBCFG <<EOF
inherit: base.yaml
checkpoint: $CKPT
output_dir: $RIBOUT
n_action_steps: 1
record:
  enabled: false
EOF
  cat > $RASFCFG <<EOF
inherit: base.yaml
checkpoint: $CKPT
output_dir: $RASFOUT
module: rasf
n_action_steps: 1
rasf_gain_floor: 0.05
rasf_gate_max: 0.95
record:
  enabled: false
EOF
}

evalbase() {
  echo "### [v2] base+TE CLEAN SR  ckpt=$CKPT  $(date +%T)"
  python scripts/eval_libero_v.py --config $RIBCFG --method baseline $TE \
    --n-action-steps 1 --episodes 20 --only clean 2>&1 | grep -iE "== clean|Error|Traceback"
  python3 -c "import json;r=json.load(open('$RIBOUT/libero_v/baseline/eval_clean.json'));print(f'>>> base+TE clean = {r[\"success_rate\"]*100:.1f}% [{r[\"success_wilson95\"][0]*100:.0f},{r[\"success_wilson95\"][1]*100:.0f}] n={r[\"n_episodes\"]}  (was 86.0)')"
}

modules() {
  echo "### [v2] retrain RIB on new base  $(date +%T)"
  python scripts/finetune_rib.py --config $RIBCFG --steps 12000 --lr-rib 3e-4 --lr-head 2e-5 \
    --batch-size 32 --d-z 512 --n-heads 8 --beta 1e-4 --free-bits 0.05 --corrupt-frac 0.6 --tag rib_${TAG}
  echo "### [v2] retrain RASF on new base  $(date +%T)"
  python scripts/train_rasf.py --config $RASFCFG --steps 8000 --tag rasf_${TAG}
  echo "### [v2] modules saved: $RIBPT  $RASFPT"
}

aegis() {
  [ -f "$RIBPT" ]  || { echo "FATAL: $RIBPT missing — run 'modules' first"; exit 1; }
  [ -f "$RASFPT" ] || { echo "FATAL: $RASFPT missing — run 'modules' first"; exit 1; }
  echo "### [v2] AEGIS CLEAN (base+RIB+RASF+TE)  $(date +%T)"
  python scripts/eval_libero_v.py --config $RIBCFG --method aegis \
    --rib-weights $RIBPT --rasf-weights $RASFPT $TE \
    --n-action-steps 1 --episodes 20 --only clean 2>&1 | grep -iE "== clean|inject|Error|Traceback"
  python3 -c "import json;r=json.load(open('$RIBOUT/libero_v/aegis/eval_clean.json'));print(f'>>> AEGIS clean = {r[\"success_rate\"]*100:.1f}% [{r[\"success_wilson95\"][0]*100:.0f},{r[\"success_wilson95\"][1]*100:.0f}] n={r[\"n_episodes\"]}  (TARGET 92, was 87.5)')"
}

gen_configs
case "$STAGE" in
  evalbase) evalbase;;
  modules)  modules;;
  aegis)    aegis;;
  all)      evalbase; modules; aegis;;
  *) echo "unknown stage: $STAGE"; exit 1;;
esac
echo "### [v2] stage '$STAGE' done $(date +%T)"
