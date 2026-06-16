#!/bin/bash
# ============================================================================
# LIBERO-V (Visual) 4-axis robustness sweep on the repro checkpoint, for
# vanilla + SmolVLA+SIB (pure) + SmolVLA+IB-Adapter. Self-contained: resolves
# the repro ckpt itself and gates on a sim self-test before any GPU rollout.
#
# Invoked by run_after_baseline.sh ONLY if armed (results/orchestrator/.libero_v_armed).
# Manual:  cd sib_vla && MUJOCO_GL=egl bash run_libero_v.sh [episodes_per_task]
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
LVCFG=configs/libero_v.yaml
LOG=results/orchestrator
mkdir -p "$LOG" results/libero_v
EPS="${1:-10}"

# --- resolve repro checkpoint + patch into libero_v.yaml -------------------
CKPT="outputs/smolvla_spatial_repro/checkpoints/last/pretrained_model"
if [ ! -d "$CKPT" ]; then
  LS=$(ls -1 outputs/smolvla_spatial_repro/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  CKPT="outputs/smolvla_spatial_repro/checkpoints/$LS/pretrained_model"
fi
if [ ! -f "$CKPT/config.json" ]; then echo "[libero_v] FATAL: repro ckpt missing ($CKPT)"; exit 1; fi
python3 - "$LVCFG" "$CKPT" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
p.write_text(re.sub(r'^checkpoint:.*$', f'checkpoint: {sys.argv[2]}', t, flags=re.M))
print("[libero_v] checkpoint ->", sys.argv[2])
PY

# --- self-test gate (env-only; no GPU policy) -----------------------------
echo "[libero_v] running sim self-test ..."
python3 -c "
import sib.libero_v as lv, sys
r = lv.self_test(save_dir='results/libero_v/selftest')
print('[libero_v] selftest deltas:', {k: round(v,2) for k,v in r.items()})
sys.exit(0 if all(v>1.0 for v in r.values()) else 1)
" || { echo '[libero_v] FATAL: sim self-test failed (perturbations not rendering). Aborting.'; exit 1; }

# --- the sweep: vanilla, SIB(pure), IB ------------------------------------
run() {  # tag method extra-args...
  local tag="$1"; shift
  echo "================ LIBERO-V: $tag ================"
  python scripts/eval_libero_v.py --config "$LVCFG" --n-action-steps 1 --episodes "$EPS" "$@" \
    2>&1 | tee "$LOG/log_libero_v_$tag.txt"
}
run vanilla --method vanilla
[ -f results/sib_repro/sib_repro.pt ] && run sib --method sib --weights results/sib_repro/sib_repro.pt \
  || echo "[libero_v] skip SIB (results/sib_repro/sib_repro.pt absent)"
[ -f results/ib_repro/smolvla_ib_repro.pt ] && run ib --method ib --weights results/ib_repro/smolvla_ib_repro.pt --n-heads 8 \
  || echo "[libero_v] skip IB (results/ib_repro/smolvla_ib_repro.pt absent)"

echo "[libero_v] DONE. Per-method summaries: results/libero_v/<method>/summary.json"
