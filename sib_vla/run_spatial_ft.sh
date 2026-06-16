#!/bin/bash
# Continue-finetune the 79% generalist on LIBERO-Spatial only (specialization push to 80+).
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
SRC=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
OUT=outputs/smolvla_spatial_ft
STEPS="${1:-4000}"

echo "==== [FT] spatial specialization from 79% ckpt: $STEPS steps, lr 5e-5, batch 128 ===="
lerobot-train \
  --policy.path=$SRC \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --rename_map='{"observation.images.wrist_image":"observation.images.image2"}' \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps=$STEPS \
  --output_dir=$OUT \
  --batch_size=128 \
  --steps=$STEPS \
  --save_freq=1000 \
  --eval_freq=100000000 \
  --log_freq=100 \
  --num_workers=24 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  2>&1 | tee results/repro/log_spatial_ft.txt

# patch the "type" discriminator into every saved checkpoint config.json
python3 scripts/patch_ckpt_type.py "$OUT"

echo "==== [FT] training done; checkpoints: ===="
ls "$OUT/checkpoints/" | grep -E '^[0-9]+$'
