#!/bin/bash
# ============================================================================
# diag_long.sh — SURGICAL ablation of the AEGIS Long t3/t4 collapse.
# Step 1: isolate the culprit module (base vs RIB-only vs RASF-only vs full).
# Step 2: sweep de-strength on the culprit to find the gentlest fix that reaches
#         base parity (t3≈85, t4≈35).  t3,t4 only, n=20/task, CONC=1 so it barely
#         touches the main eval's CPU. Each variant -> its own output_dir (no collide).
# Restartable: skips any variant whose eval_clean.json already exists.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
B86=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"
TASKS="3,4"; EP=20

run(){   # name  <method+weight+override args...>
  local name=$1; shift
  local od=results/diag_long/$name
  local meth; meth=$(echo "$* " | grep -oE 'method [a-z]+' | awk '{print $2}')
  [ -f "$od/libero_v/$meth/eval_clean.json" ] && { echo "[$(date +%T)] skip $name (done)"; return; }
  mkdir -p "$od" configs
  cat > configs/_diag_${name}.yaml <<EOF
inherit: base.yaml
checkpoint: $B86
output_dir: $od
suite: libero_10
episode_length: 520
n_action_steps: 1
record:
  enabled: false
EOF
  echo "[$(date +%T)] RUN $name : $*"
  $PY scripts/eval_libero_v.py --config configs/_diag_${name}.yaml "$@" \
      --n-action-steps 1 --episodes $EP --tasks $TASKS --only clean > "$od/run.log" 2>&1
  echo "[$(date +%T)] DONE $name"
}

echo "=== Long t3/t4 ablation (base ref from main run: t3=85 t4=35) ==="
# --- Step 1: isolate the culprit ---
run base        --method baseline $TE
run rib_only    --method aegis --rib-weights $RIB $TE
run rasf_only   --method aegis --rib-weights $RIB --rib-fusion-scale 0 --rasf-weights $RASF $TE
run full        --method aegis --rib-weights $RIB --rasf-weights $RASF $TE
# --- Step 2: de-strength fixes (both modules covered; analysis picks the relevant ones) ---
run fix_gm0.5   --method aegis --rib-weights $RIB --rasf-weights $RASF $TE --rasf-gate-max 0.5
run fix_gm0.3   --method aegis --rib-weights $RIB --rasf-weights $RASF $TE --rasf-gate-max 0.3
run fix_rib0.5  --method aegis --rib-weights $RIB --rasf-weights $RASF $TE --rib-fusion-scale 0.5

echo; echo "=== DIAG SUMMARY (per-task t3/t4 SR) ==="
$PY - <<'PY'
import json,glob,os
order=['base','rib_only','rasf_only','full','fix_gm0.5','fix_gm0.3','fix_rib0.5']
print('%-12s %6s %6s   %s'%('variant','t3','t4','note'))
ref={3:85,4:35}
for n in order:
    js=glob.glob(f'results/diag_long/{n}/libero_v/*/eval_clean.json')
    if not js: print('%-12s %6s %6s'%(n,'-','-')); continue
    d=json.load(open(js[0])); pt={t['task_id']:round(t['success_rate']*100) for t in d.get('per_task',[])}
    t3,t4=pt.get(3),pt.get(4)
    note=''
    if n=='full': note='<- the collapse (expect ~30/10)'
    print('%-12s %6s %6s   %s'%(n,t3,t4,note))
print(f"\nbase reference (main run): t3={ref[3]} t4={ref[4]}  -> parity target")
PY
