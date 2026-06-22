#!/bin/bash
# NanoVLA-S baseline training (language-conditioned ACT + frozen BERT token) on LIBERO.
# Queues behind the SmolVLA base retrain (single GPU). After it converges, validate vs
# paper Table 1 (Spatial 81.6 / Object 93.6 / Goal 89.6 / Long 49.8 / Avg 78.7), THEN
# port AEGIS (RIB @ encoder_img_feat_input_proj, RASF @ action chunk) — the next-week work.
#   bash run_nanovla_baseline.sh smoke           # 3-step real-data smoke (sanity)
#   bash run_nanovla_baseline.sh [steps] [batch] [lr]
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${1:-}" = "smoke" ]; then
  echo "[nanovla] 3-step real-data smoke ..."
  exec python scripts/finetune_nanovla.py --smoke
fi
STEPS="${1:-100000}"; BATCH="${2:-64}"; LR="${3:-1e-4}"; OUT=results/nanovla_s
echo "[nanovla] baseline train steps=$STEPS batch=$BATCH lr=$LR -> $OUT  $(date '+%F %T')"
python scripts/finetune_nanovla.py --repo-id HuggingFaceVLA/libero \
  --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --out "$OUT" \
  2>&1 | tee results/log_nanovla_baseline.txt
echo "[nanovla] DONE $(date '+%F %T')  -> $OUT/nanovla_s.pt"
