#!/bin/bash
# Stronger base retrain to chase higher clean SR (current 020000 -> 86% w/TE, 87.5% AEGIS).
# SAME proven recipe (frozen VLM + train_expert_only, lr cosine) as run_repro_pipeline.sh,
# scaled up on the lever most likely to help a non-compute-limited ceiling: BIGGER BATCH
# (256 -> 512, better gradient estimates) + moderately more steps. save_freq=4000 so we
# eval intermediate checkpoints and pick the SR peak (guards against overfitting the small
# ~500-demo dataset). Architecture flags left at repro defaults -> RIB/RASF stay drop-in.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # recover fragmentation (batch 512 OOM'd by 378MB)
OUT=outputs/smolvla_spatial_v2
STEPS="${1:-30000}"
BATCH="${2:-384}"
LR="${3:-1e-4}"
WARMUP=1500
mkdir -p results/repro_v2
echo "[retrain] OUT=$OUT steps=$STEPS batch=$BATCH lr=$LR warmup=$WARMUP  $(date '+%F %T')"
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=$LR \
  --policy.scheduler_warmup_steps=$WARMUP \
  --policy.scheduler_decay_steps=$STEPS \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --output_dir=$OUT \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=4000 \
  --eval_freq=100000000 \
  --log_freq=100 \
  --num_workers=32 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  2>&1 | tee results/repro_v2/log_train.txt
echo "[retrain] DONE $(date '+%F %T')  -> $OUT/checkpoints/"
