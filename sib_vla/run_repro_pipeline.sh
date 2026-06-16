#!/bin/bash
# Proper SmolVLA reproduction on LIBERO (doc §2b recipe), then eval on Spatial.
#
# Phase 1: retrain from lerobot/smolvla_base — SmolVLM2 VLM loaded + FROZEN,
#          action expert trained (official train_expert_only recipe), but with
#          the doc's CAUSE-2 fix: large effective batch (256) instead of 4-32.
# Phase 2: eval the result on LIBERO-Spatial across an n_action_steps sweep,
#          report best vs the 72.0% vanilla baseline.

set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl

OUT=outputs/smolvla_spatial_repro
STEPS="${1:-20000}"
BATCH="${2:-256}"
mkdir -p results/repro

echo "================================================================"
echo "[Phase 1] Retrain from smolvla_base  steps=$STEPS  batch=$BATCH"
echo "          frozen VLM + trained expert (~100M params), lr=1e-4 cosine"
echo "================================================================"

lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
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
  2>&1 | tee results/repro/log_train.txt

# Resolve the saved checkpoint (lerobot writes checkpoints/last -> latest step)
CKPT="$OUT/checkpoints/last/pretrained_model"
if [ ! -d "$CKPT" ]; then
  LASTSTEP=$(ls -1 "$OUT/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
  CKPT="$OUT/checkpoints/$LASTSTEP/pretrained_model"
fi
echo "[Phase 1] Done. Checkpoint: $CKPT"

# Point the eval config at the resolved checkpoint
python3 - "$CKPT" <<'PY'
import sys, re, pathlib
ckpt = sys.argv[1]
p = pathlib.Path("configs/repro_eval.yaml")
t = p.read_text()
t = re.sub(r'^checkpoint:.*$', f'checkpoint: {ckpt}', t, flags=re.M)
p.write_text(t)
print("[cfg] repro_eval.yaml checkpoint ->", ckpt)
PY

echo "================================================================"
echo "[Phase 2] Eval on LIBERO-Spatial  (n_action_steps sweep)"
echo "================================================================"

for N in 10 1; do
  echo "---- eval n_action_steps=$N ----"
  python scripts/eval.py --config configs/repro_eval.yaml \
    --n-action-steps "$N" --tag "repro_n$N" \
    2>&1 | tee "results/repro/log_eval_n$N.txt"
done

echo "================================================================"
echo "[Pipeline] DONE — results vs vanilla baseline 72.0%"
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("results/repro/eval_repro_n*.json")):
    r = json.load(open(f))
    p = r["success_rate"]; lo, hi = r["success_wilson95"]; n = r["n_episodes"]
    tag = r["name"]
    delta = (p - 0.72) * 100
    print(f"  {tag:12s} SR={p*100:5.1f}%  Wilson95=[{lo*100:.1f},{hi*100:.1f}]  n={n}  vs72%: {delta:+.1f}pp")
PY
echo "================================================================"
