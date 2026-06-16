#!/bin/bash
# ============================================================================
# Auto-orchestrator: waits for the proper baseline (run_repro_pipeline.sh) to
# finish, then runs the two requested method families on the SAME (new, bs=256
# repro) checkpoint and reports each vs the new baseline:
#
#   1. SmolVLA + SIB   -- both variants (user choice):
#        (a) pure  : spectral bottleneck on the FROZEN repro head
#        (b) +LoRA : LoRA-adapt the repro head, then SIB on top  (SR can move)
#   2. SmolVLA + IB-Adapter (Fused, StableVLA) in the visual connector
#
# All method evals are at n_action_steps=1 (the checkpoint optimum + the
# apples-to-apples comparison vs the baseline's n=1 eval). SIB-pure also gets an
# n=25 eval for the jerk/smoothness story.
#
# Gating: refuses to burn GPU unless the repro checkpoint exists AND its n=1
# baseline SR is sane (>0.40). Each stage is fault-isolated; a failure in one
# method does not abort the others. Status -> results/orchestrator/STATUS.txt.
#
# LIBERO-V (4-axis robustness) runs LAST, but ONLY if it has been armed
# (run_libero_v.sh exists AND results/orchestrator/.libero_v_armed present).
#
# Launch (detached, survives the session):
#   cd sib_vla && nohup bash run_after_baseline.sh > results/orchestrator/nohup.out 2>&1 &
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl

ORCH=results/orchestrator
mkdir -p "$ORCH" results/sib_repro results/sib_lora_repro results/ib_repro
STATUS="$ORCH/STATUS.txt"
SUMMARY="$ORCH/SUMMARY.txt"
BASE_N1_JSON="results/repro/eval_repro_n1.json"
REPRO_CKPT_DEFAULT="outputs/smolvla_spatial_repro/checkpoints/last/pretrained_model"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [orch] $*" | tee -a "$STATUS"; }

log "orchestrator started (pid $$). Target: SIB(pure+LoRA) + IB-Adapter on repro ckpt, n=1."

# ---------------------------------------------------------------------------
# 1. WAIT for the baseline pipeline to finish.
#    Completion signal = results/repro/eval_repro_n1.json (the pipeline's LAST
#    artefact, written after the n=10 then n=1 evals). While run_repro_pipeline.sh
#    is alive we keep waiting; if it dies without the json -> fatal.
# ---------------------------------------------------------------------------
log "waiting for baseline ($BASE_N1_JSON) ..."
while [ ! -f "$BASE_N1_JSON" ]; do
  if ! pgrep -af "run_repro_pipeline.sh" >/dev/null 2>&1; then
    sleep 45   # grace for final flush
    if [ -f "$BASE_N1_JSON" ]; then break; fi
    log "FATAL: run_repro_pipeline.sh is gone but $BASE_N1_JSON never appeared."
    log "       baseline likely crashed. NOT launching method runs. Inspect results/repro/."
    echo "FAILED: baseline did not complete" > "$SUMMARY"
    exit 1
  fi
  sleep 300
done
log "baseline pipeline complete."

# ---------------------------------------------------------------------------
# 2. Resolve the repro checkpoint + sanity-gate.
# ---------------------------------------------------------------------------
CKPT="$REPRO_CKPT_DEFAULT"
if [ ! -d "$CKPT" ]; then
  LASTSTEP=$(ls -1 outputs/smolvla_spatial_repro/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  CKPT="outputs/smolvla_spatial_repro/checkpoints/$LASTSTEP/pretrained_model"
fi
if [ ! -f "$CKPT/config.json" ]; then
  log "FATAL: resolved checkpoint '$CKPT' has no config.json. Aborting."
  echo "FAILED: repro checkpoint missing/invalid ($CKPT)" > "$SUMMARY"; exit 1
fi
log "repro checkpoint: $CKPT"

BASE_SR_N1=$(python3 -c "import json;print(json.load(open('$BASE_N1_JSON'))['success_rate'])" 2>/dev/null || echo "nan")
BASE_SR_N10=$(python3 -c "import json;print(json.load(open('results/repro/eval_repro_n10.json'))['success_rate'])" 2>/dev/null || echo "nan")
log "baseline SR: n=1 ${BASE_SR_N1}  n=10 ${BASE_SR_N10}"

python3 -c "import sys;v=float('$BASE_SR_N1');sys.exit(0 if v>0.40 else 1)" 2>/dev/null || {
  log "FATAL: baseline n=1 SR ${BASE_SR_N1} <= 0.40 (collapsed). NOT burning GPU on methods."
  echo "FAILED: baseline SR collapsed (${BASE_SR_N1})" > "$SUMMARY"; exit 1
}

# ---------------------------------------------------------------------------
# 3. Patch the resolved checkpoint into the repro-pointed configs.
#    (sib_lora_repro.yaml points at the LoRA-merged model, produced below.)
# ---------------------------------------------------------------------------
for cfg in configs/sib_repro.yaml configs/lora_repro.yaml configs/ib_adapter_repro.yaml; do
  python3 - "$cfg" "$CKPT" <<'PY'
import sys, re, pathlib
cfg, ckpt = sys.argv[1], sys.argv[2]
p = pathlib.Path(cfg); t = p.read_text()
t = re.sub(r'^checkpoint:.*$', f'checkpoint: {ckpt}', t, flags=re.M)
p.write_text(t)
PY
  log "patched checkpoint -> $cfg"
done

# Helper: run a stage, tee its log, never abort the whole pipeline on failure.
run_stage() {  # name logfile cmd...
  local name="$1" logf="$2"; shift 2
  log "START  $name"
  if "$@" > "$logf" 2>&1; then
    log "DONE   $name"
  else
    log "FAILED $name (rc=$?) -- see $logf ; continuing to next stage"
  fi
}

# ===========================================================================
# STAGE A -- SmolVLA + SIB (pure, frozen repro head)
#   estimate_lambda (build cache from repro head) -> train bottleneck -> eval
# ===========================================================================
run_stage "SIB-pure: estimate_lambda" "$ORCH/log_sibpure_lambda.txt" \
  python scripts/estimate_lambda.py --config configs/sib_repro.yaml
run_stage "SIB-pure: train"           "$ORCH/log_sibpure_train.txt" \
  python scripts/train.py --config configs/sib_repro.yaml --beta 1e-3 --tag sib_repro
run_stage "SIB-pure: eval n=1"        "$ORCH/log_sibpure_eval_n1.txt" \
  python scripts/eval.py --config configs/sib_repro.yaml \
    --weights results/sib_repro/sib_repro.pt --n-action-steps 1 --tag sib_repro_n1
run_stage "SIB-pure: eval n=25 (jerk)" "$ORCH/log_sibpure_eval_n25.txt" \
  python scripts/eval.py --config configs/sib_repro.yaml \
    --weights results/sib_repro/sib_repro.pt --n-action-steps 25 --tag sib_repro_n25

# ===========================================================================
# STAGE B -- SmolVLA + SIB + LoRA (LoRA-adapt repro head, then SIB)
#   finetune_lora -> estimate_lambda (on lora model) -> train -> eval n=1
# ===========================================================================
run_stage "SIB+LoRA: finetune_lora"   "$ORCH/log_siblora_lora.txt" \
  python scripts/finetune_lora.py --config configs/lora_repro.yaml \
    --steps 4000 --rank 16 --alpha 32 --lr 2e-4 --batch-size 8 --out-name smolvla_lora
if [ -f results/sib_lora_repro/smolvla_lora/config.json ]; then
  run_stage "SIB+LoRA: estimate_lambda" "$ORCH/log_siblora_lambda.txt" \
    python scripts/estimate_lambda.py --config configs/sib_lora_repro.yaml
  run_stage "SIB+LoRA: train"           "$ORCH/log_siblora_train.txt" \
    python scripts/train.py --config configs/sib_lora_repro.yaml --beta 1e-4 --tag sib_lora_repro
  run_stage "SIB+LoRA: eval n=1"        "$ORCH/log_siblora_eval_n1.txt" \
    python scripts/eval.py --config configs/sib_lora_repro.yaml \
      --weights results/sib_lora_repro/sib_lora_repro.pt --n-action-steps 1 --tag sib_lora_repro_n1
else
  log "FAILED SIB+LoRA: merged LoRA model absent -- skipping SIB-on-LoRA sub-stages"
fi

# ===========================================================================
# STAGE C -- SmolVLA + IB-Adapter (Fused, in visual connector)
#   finetune_ib -> eval n=1
# ===========================================================================
run_stage "IB-Adapter: finetune"      "$ORCH/log_ib_finetune.txt" \
  python scripts/finetune_ib.py --config configs/ib_adapter_repro.yaml \
    --steps 10000 --lr-ib 2e-4 --lr-head 2e-5 --batch-size 32 --n-heads 8 --tag smolvla_ib_repro
run_stage "IB-Adapter: eval n=1"      "$ORCH/log_ib_eval_n1.txt" \
  python scripts/eval_ib.py --config configs/ib_adapter_repro.yaml \
    --weights results/ib_repro/smolvla_ib_repro.pt --n-heads 8 --tag ib_repro_n1

# ===========================================================================
# 4. SUMMARY -- every method vs the new baseline (n=1), +5pp check.
# ===========================================================================
log "writing summary"
python3 - "$BASE_SR_N1" "$BASE_SR_N10" > "$SUMMARY" 2>&1 <<'PY'
import json, os, sys
base_n1  = float(sys.argv[1])
base_n10 = float(sys.argv[2]) if sys.argv[2] not in ("nan","") else float("nan")
def load(p):
    try: return json.load(open(p))
    except Exception: return None
rows = [
  ("baseline (repro) n=1",  "results/repro/eval_repro_n1.json"),
  ("baseline (repro) n=10", "results/repro/eval_repro_n10.json"),
  ("SmolVLA+SIB (pure) n=1",        "results/sib_repro/eval_sib_repro_n1.json"),
  ("SmolVLA+SIB (pure) n=25",       "results/sib_repro/eval_sib_repro_n25.json"),
  ("SmolVLA+SIB (+LoRA) n=1",       "results/sib_lora_repro/eval_sib_lora_repro_n1.json"),
  ("SmolVLA+IB-Adapter n=1",        "results/ib_repro/eval_ib_repro_n1.json"),
]
print("="*78)
print(" RESULTS vs new repro baseline (n=1 = %.1f%%)   [goal: method >= baseline+5pp]" % (base_n1*100))
print("="*78)
print(f"{'method':32s} {'SR%':>6s} {'Wilson95':>16s} {'n':>5s} {'Δvs base':>9s} {'jerk':>7s}  5pp?")
for label, path in rows:
    r = load(path)
    if r is None:
        print(f"{label:32s} {'--':>6s}  (missing: {path})"); continue
    sr = r['success_rate']; lo,hi = r['success_wilson95']; n=r['n_episodes']
    jerk = r.get('rms_jerk_mean', float('nan'))
    d = (sr - base_n1)*100
    flag = "" if label.startswith("baseline") else ("YES" if d>=5.0 else "no")
    print(f"{label:32s} {sr*100:6.1f} [{lo*100:5.1f},{hi*100:5.1f}] {n:5d} {d:+8.1f}pp {jerk:7.3f}  {flag}")
print("="*78)
print("note: SIB-pure is a smoothness method (expect SR~baseline, big jerk drop at n=25).")
print("      SIB+LoRA and IB-Adapter are the SR plays. +5pp over an already-strong")
print("      baseline is a stretch goal, reported honestly above.")
PY
cat "$SUMMARY" | tee -a "$STATUS"

# ===========================================================================
# 5. LIBERO-V robustness (4-axis) -- only if armed.
# ===========================================================================
if [ -f run_libero_v.sh ] && [ -f "$ORCH/.libero_v_armed" ]; then
  log "LIBERO-V armed -> launching run_libero_v.sh"
  run_stage "LIBERO-V (4-axis)" "$ORCH/log_libero_v.txt" bash run_libero_v.sh
else
  log "LIBERO-V not armed (run_libero_v.sh missing or $ORCH/.libero_v_armed absent) -- skipping."
fi

log "ALL DONE."
