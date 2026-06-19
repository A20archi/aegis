#!/bin/bash
# READY-TO-FIRE Long-only finetune pipeline. Fires ONLY if Long comes short of 73.7.
# LIBERO is conventionally trained per-suite, so a Long-specific adapter is faithful:
# Long suite -> this Long-finetuned ckpt; other suites keep the boosted base.
#   1) LoRA finetune SmolVLA on the 379 libero_10 (Long) episodes only
#   2) patch merged ckpt (inject type=smolvla + copy 4 processor files from base)
#   3) eval Long n=200 n=1: boostedBase+TE (baseline) then +RIB+RASF (aegis)
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
BASE=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
OUTDIR=results/lora_long
MERGED=$OUTDIR/smolvla_lora_long
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
STEPS=${STEPS:-3000}; RANK=${RANK:-32}; ALPHA=${ALPHA:-64}; LR=${LR:-2e-4}
EP=${EP:-50}; NEED=${NEED:-15500}
TE="--forge-ensemble --ensemble-coeff 0.01"

wait_mem(){ while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)
  [ "${free:-0}" -ge "$NEED" ] && { echo "[$(date +%T)] free=${free} >= $NEED -> go"; break; }
  echo "[$(date +%T)] free=${free} < $NEED, wait"; sleep 60; done; }

echo "[$(date +%T)] === STEP 1: Long-only LoRA finetune ($STEPS steps) ==="
wait_mem
$PY scripts/finetune_lora.py --config configs/_lora_boost.yaml \
  --episodes-json results/long_episode_idx.json \
  --steps "$STEPS" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" --batch-size 32 \
  --out-name smolvla_lora_long
[ -f "$MERGED/model.safetensors" ] || { echo "FATAL: finetune produced no merged ckpt"; exit 1; }

echo "[$(date +%T)] === STEP 2: patch merged ckpt ==="
$PY - "$BASE" "$MERGED" <<'PY'
import json,sys,shutil,os
base,merged=sys.argv[1],sys.argv[2]
cfg=json.load(open(f"{merged}/config.json"))
if "type" not in cfg:
    bt=json.load(open(f"{base}/config.json")).get("type","smolvla")
    cfg["type"]=bt; json.dump(cfg,open(f"{merged}/config.json","w"),indent=2)
    print("  injected type:",bt)
for f in ["policy_postprocessor.json","policy_preprocessor.json",
          "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
          "policy_preprocessor_step_5_normalizer_processor.safetensors"]:
    if not os.path.exists(f"{merged}/{f}") and os.path.exists(f"{base}/{f}"):
        shutil.copy(f"{base}/{f}",f"{merged}/{f}"); print("  copied",f)
print("  patch done")
PY

echo "[$(date +%T)] === STEP 3: eval Long n=200 n=1 ==="
od=results/long_ft/long; mkdir -p "$od"
cat > configs/_longft_long.yaml <<YAML
inherit: base.yaml
checkpoint: $MERGED
output_dir: $od
suite: libero_10
episode_length: 520
n_action_steps: 1
record:
  enabled: false
YAML
wait_mem
echo "[$(date +%T)] Long baseline (boostedLong + TE)"
$PY scripts/eval_libero_v.py --config configs/_longft_long.yaml --method baseline $TE \
  --n-action-steps 1 --episodes "$EP" --only clean >"$od/baseline.log" 2>&1
bsr=$($PY -c "import json;print('%.1f'%(json.load(open('$od/libero_v/baseline/eval_clean.json'))['success_rate']*100))" 2>/dev/null||echo ERR)
wait_mem
echo "[$(date +%T)] Long AEGIS (boostedLong + RIB+RASF + TE)"
$PY scripts/eval_libero_v.py --config configs/_longft_long.yaml --method aegis \
  --rib-weights "$RIB" --rasf-weights "$RASF" $TE \
  --n-action-steps 1 --episodes "$EP" --only clean >"$od/aegis.log" 2>&1
asr=$($PY -c "import json;print('%.1f'%(json.load(open('$od/libero_v/aegis/eval_clean.json'))['success_rate']*100))" 2>/dev/null||echo ERR)
echo "[$(date +%T)] === LONG FINETUNE RESULT: baseline=$bsr  AEGIS=$asr  (threshold 73.7 to beat 87.3) ==="
