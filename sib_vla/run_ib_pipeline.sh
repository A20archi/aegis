#!/bin/bash
# IB-Adapter pipeline: Phase 1 fine-tune (~1 h) → Phase 2 eval (~2 h)
# SmolVLA + Fused IB-Adapter (StableVLA, Fu et al. 2026) on LIBERO-Spatial

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── config ────────────────────────────────────────────────────────────────
CFG="configs/ib_adapter.yaml"
STEPS="${1:-10000}"         # override via $1, e.g. ./run_ib_pipeline.sh 5000
TAG="smolvla_ib"
HEADS=8
LR_IB=2e-4
LR_HEAD=2e-5
BATCH=32
WEIGHTS="results/ib_adapter/${TAG}.pt"

# ── Phase 1 : fine-tune IB-Adapter + action head ─────────────────────────
echo "================================================================"
echo "[Phase 1] Fine-tuning IB-Adapter for ${STEPS} steps"
echo "================================================================"

mkdir -p results/ib_adapter

python scripts/finetune_ib.py \
    --config "$CFG" \
    --steps   "$STEPS" \
    --lr-ib   "$LR_IB" \
    --lr-head "$LR_HEAD" \
    --batch-size "$BATCH" \
    --n-heads "$HEADS" \
    --tag     "$TAG" \
    2>&1 | tee results/ib_adapter/log_finetune.txt

echo ""
echo "[Phase 1] Done. Weights at: $WEIGHTS"

# ── Phase 2 : evaluate on LIBERO-Spatial ─────────────────────────────────
echo "================================================================"
echo "[Phase 2] Evaluating SmolVLA + IB-Adapter  (n=1, LIBERO-Spatial)"
echo "================================================================"

python scripts/eval_ib.py \
    --config  "$CFG" \
    --weights "$WEIGHTS" \
    --n-heads "$HEADS" \
    --tag     "ib_adapter_n1" \
    2>&1 | tee results/ib_adapter/log_eval.txt

RESULT_FILE="results/ib_adapter/eval_ib_adapter_n1.json"
echo ""
echo "================================================================"
echo "[Pipeline] DONE"
if [ -f "$RESULT_FILE" ]; then
    python3 -c "
import json, sys
r = json.load(open('$RESULT_FILE'))
p  = r['success_rate']
lo, hi = r['success_wilson95']
n  = r['n_episodes']
print(f'  SmolVLA+IB-Adapter (n=1)  SR = {p:.3f} ({p*100:.1f}%)  '
      f'Wilson95=[{lo:.3f},{hi:.3f}]  n={n}')
"
fi
echo "================================================================"
