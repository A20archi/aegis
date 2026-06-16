#!/usr/bin/env bash
# Full Forge+SIB pipeline:
#   Phase 1: Forge fine-tune SmolVLA (L1 EMA + L3 norm + L4 aug)  ~2h
#   Phase 2: estimate_lambda on forge model                         ~9 min
#   Phase 3: train SIB on forge model                              ~2 min
#   Phase 4: eval forge+SIB at n=1                                 ~2.5h
set -euo pipefail
cd /home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla

LOG=results/log_forge_pipeline.txt
mkdir -p results/forge_ft
echo "" | tee -a "$LOG"
echo "=== FORGE PIPELINE START  $(date) ===" | tee -a "$LOG"

# -----------------------------------------------------------------------
# Phase 1: Forge fine-tune
# -----------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== [1/4] Forge fine-tune (L1+L3+L4) ===" | tee -a "$LOG"
python -u scripts/finetune_forge.py \
    --config configs/sib.yaml \
    --steps 10000 \
    --lr-head 2e-5 \
    --lr-backbone 2e-6 \
    --batch-size 32 \
    --ema-decay 0.9999 \
    --aug-prob 0.3 \
    --lever3-clamp 0.98 \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------
# Phase 2: estimate_lambda on forge model
# -----------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== [2/4] estimate_lambda on forge model ===" | tee -a "$LOG"
python -u scripts/estimate_lambda.py \
    --config configs/forge_ft.yaml \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------
# Phase 3: train SIB on forge model
# -----------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== [3/4] train SIB on forge model (beta=1e-4) ===" | tee -a "$LOG"
python -u scripts/train.py \
    --config configs/forge_ft.yaml \
    --beta 1e-4 \
    --tag forge_sib_b1e-4 \
    2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------
# Phase 4: eval forge+SIB at n=1
# -----------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== [4/4] eval forge+SIB at n=1 (preserve_energy) ===" | tee -a "$LOG"
python -u scripts/eval.py \
    --config configs/forge_ft_n1.yaml \
    --weights results/forge_ft/forge_sib_b1e-4.pt \
    --tag forge_sib_n1_pe \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== FORGE PIPELINE DONE  $(date) ===" | tee -a "$LOG"

python3 -c "
import json, glob
for f in sorted(glob.glob('results/eval_*.json')):
    d = json.load(open(f))
    print(f\"{d['name']:40s}  SR={d['success_rate']:.3f}  n={d['n_episodes']}\")
" 2>&1 | tee -a "$LOG"
