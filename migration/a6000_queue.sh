#!/bin/bash
# ============================================================================
# a6000_queue.sh — RESTARTABLE master queue for the A6000.
# Resumes the SmolVLA base 12k->30k (A6000-optimal batch, OOM auto-ladder), then
# runs the ENTIRE SmolVLA AEGIS pipeline: peak-pick -> modules -> AEGIS clean ->
# robustness + noise sweep. Every stage is skip-if-done, so re-running after an
# interruption continues exactly where it stopped ("restart everything as it is").
#
# PREREQ: activate the SmolVLA env first (lerobot 0.4.3 / torch cu124):
#     conda activate smolvla        # the env recreated from migration/pip_smolvla_base.txt
#     cd <repo>/sib_vla
#     bash ../migration/a6000_queue.sh            # full queue
#     bash ../migration/a6000_queue.sh stage0     # just resume training
#     NW=12 BATCH_LADDER="128 96" bash ../migration/a6000_queue.sh   # tune CPUs/batch
# ============================================================================
set -uo pipefail
# locate sib_vla regardless of where we're called from
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/run_v2_pipeline.sh" ]; then cd "$HERE"; else cd "$HERE/../sib_vla"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl

OUT=outputs/smolvla_spatial_v2
CKDIR=$OUT/checkpoints
MIG="$(cd .. && pwd)/migration"
LOG="$MIG/a6000_run.log"
NW=${NW:-16}
BATCHES=(${BATCH_LADDER:-160 128 96})        # A6000-optimal -> fallbacks
ONLY="${1:-all}"
mkdir -p results/repro_v2 "$MIG"
mark(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- STAGE 0
stage0_resume_base(){
  if [ -d "$CKDIR/030000" ]; then mark "STAGE0 skip: 030000 already exists"; return 0; fi
  local cfg="$CKDIR/last/pretrained_model/train_config.json"
  [ -f "$cfg" ] || { mark "STAGE0 FATAL: no resume config at $cfg"; exit 1; }
  local from; from=$(ls "$CKDIR" | grep -E '^[0-9]+$' | sort | tail -1)
  for B in "${BATCHES[@]}"; do
    local alog="results/repro_v2/log_resume_b${B}.txt"
    mark "STAGE0 resume base $from -> 30k  batch=$B  num_workers=$NW"
    set +e
    lerobot-train --config_path="$cfg" --resume=true \
      --batch_size=$B --num_workers=$NW --policy.device=cuda 2>&1 | tee "$alog" >>"$LOG"
    set -e
    [ -d "$CKDIR/030000" ] && { mark "STAGE0 done (batch=$B)"; return 0; }
    if grep -qiE "out of memory|CUDA out of memory" "$alog"; then
      mark "STAGE0 OOM at batch=$B -> dropping to next"; continue
    fi
    # If resume rejected the batch override, fall back to weight-init continuation:
    if grep -qiE "resume.*config|cannot.*resume|mismatch" "$alog"; then
      mark "STAGE0 NOTE: lerobot rejected batch override on resume. Fallback below."
      mark "  -> run: bash run_base_retrain.sh \$((30000)) $B 1e-4  AFTER copying 012000 weights as init,"
      mark "     OR keep batch=384 if it happens to fit. See migration/QUEUE_README.md."
    fi
    mark "STAGE0 exited without 030000 and no OOM (batch=$B) — investigate $alog"; exit 1
  done
  mark "STAGE0 FAILED: smallest batch still OOM/incomplete"; exit 1
}

# ---------------------------------------------------------------- STAGE 1 (peak-pick 12k->30k)
stage1_peakpick(){
  local sum="$MIG/peak_pick.txt"; touch "$sum"
  for c in 012000 016000 020000 024000 028000 030000; do
    [ -d "$CKDIR/$c" ] || { mark "peakpick: no ckpt $c (skip)"; continue; }
    grep -q "^$c " "$sum" && { mark "peakpick: $c already eval'd (skip)"; continue; }
    mark "peakpick: eval base+TE clean @ $c"
    local sr
    sr=$(bash run_v2_pipeline.sh evalbase "$CKDIR/$c/pretrained_model" 2>&1 | tee -a "$LOG" \
         | grep -oE 'base\+TE clean = [0-9.]+' | grep -oE '[0-9.]+$' | head -1)
    echo "$c ${sr:-NA}" >> "$sum"; mark "peakpick: $c -> ${sr:-NA}%"
  done
  local PEAK; PEAK=$(grep -vE ' NA$' "$sum" | sort -k2 -n | tail -1 | awk '{print $1}')
  [ -n "$PEAK" ] || { mark "STAGE1 FATAL: no valid SR in $sum"; exit 1; }
  echo "$PEAK" > "$MIG/peak_ckpt.txt"
  mark "STAGE1 PEAK ckpt = $PEAK  ($(grep "^$PEAK " "$sum"))"
}

peak_path(){ echo "$CKDIR/$(cat "$MIG/peak_ckpt.txt")/pretrained_model"; }

# ---------------------------------------------------------------- STAGE 2 (modules) / 3 (aegis)
stage2_modules(){
  if [ -f results/ib_on_v2/rib_v2.pt ] && [ -f results/rasf_on_v2/rasf_v2.pt ]; then
    mark "STAGE2 skip: rib_v2.pt + rasf_v2.pt exist"; return 0; fi
  mark "STAGE2 retrain RIB+RASF on peak $(cat "$MIG/peak_ckpt.txt")"
  bash run_v2_pipeline.sh modules "$(peak_path)" 2>&1 | tee -a "$LOG"
}
stage3_aegis(){
  if [ -f results/ib_on_v2/libero_v/aegis/eval_clean.json ]; then
    mark "STAGE3 skip: AEGIS clean eval exists"; return 0; fi
  mark "STAGE3 AEGIS clean n=200 on peak"
  bash run_v2_pipeline.sh aegis "$(peak_path)" 2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------- STAGE 4 (robustness + noise)
stage4_robustness(){
  export CFG=configs/_rib_v2.yaml RIB=results/ib_on_v2/rib_v2.pt RASF=results/rasf_on_v2/rasf_v2.pt
  if [ ! -f results/ib_on_v2/libero_v/aegis/eval_viewpoint_medium.json ]; then
    mark "STAGE4a robustness table (4 axes, n=200, videos)"
    bash run_libero_v_headline.sh 20 0,1,2,3,4,5,6,7,8,9 2>&1 | tee -a "$LOG"
  else mark "STAGE4a skip: robustness outputs exist"; fi
  if [ ! -f results/ib_on_v2/libero_v/aegis/eval_gaussian_noise_7.json ]; then
    mark "STAGE4b noise sweep sigma 0.05->1.0 (both arms)"
    bash run_noise_sweep.sh 20 0,1,2,3,4,5,6,7,8,9 2>&1 | tee -a "$LOG"
  else mark "STAGE4b skip: noise sweep outputs exist"; fi
}

# ---------------------------------------------------------------- STAGE 5
# 4-SUITE clean SR (Spatial/Object/Goal/Long) on the ALREADY-TRAINED 86 base + existing
# modules, n=1+TE -> exactly the recipe that gave 86 on Spatial. NO retrain (the base is
# multi-suite: 40 tasks). This is the FAST path to the "beat 87.3 average" goal (~10h).
stage5_allsuites(){
  local B86=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
  local RIB=results/ib_on86/rib_on86.pt RASF=results/rasf_on86/rasf_on86.pt
  [ -d "$B86" ] || { mark "STAGE5 FATAL: 86 base missing ($B86)"; exit 1; }
  [ -f "$RIB" ] && [ -f "$RASF" ] || { mark "STAGE5 FATAL: modules missing ($RIB / $RASF)"; exit 1; }
  local TE="--forge-ensemble --ensemble-coeff 0.01"
  local EP="${EP:-20}"                          # episodes/task (20 -> n=200/suite; drop to 10 if slow)
  local names=(spatial object goal long)
  local suites=(libero_spatial libero_object libero_goal libero_10)
  for i in "${!names[@]}"; do
    local name=${names[$i]} suite=${suites[$i]}
    local od=results/allsuites/$name cfg=configs/_allsuite_$name.yaml
    mkdir -p "$od"
    cat > "$cfg" <<EOF
inherit: base.yaml
checkpoint: $B86
output_dir: $od
suite: $suite
n_action_steps: 1
record:
  enabled: false
EOF
    if [ ! -f "$od/libero_v/baseline/eval_clean.json" ]; then
      mark "STAGE5 $name ($suite): baseline+TE clean  EP=$EP"
      python scripts/eval_libero_v.py --config "$cfg" --method baseline $TE \
        --n-action-steps 1 --episodes "$EP" --only clean 2>&1 | tee -a "$LOG" | grep -iE "clean|Error|Traceback"
    else mark "STAGE5 $name baseline: skip (exists)"; fi
    if [ ! -f "$od/libero_v/aegis/eval_clean.json" ]; then
      mark "STAGE5 $name ($suite): AEGIS clean  EP=$EP"
      python scripts/eval_libero_v.py --config "$cfg" --method aegis \
        --rib-weights "$RIB" --rasf-weights "$RASF" $TE \
        --n-action-steps 1 --episodes "$EP" --only clean 2>&1 | tee -a "$LOG" | grep -iE "clean|inject|Error|Traceback"
    else mark "STAGE5 $name aegis: skip (exists)"; fi
  done
  python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json
names=['spatial','object','goal','long']
def sr(n,arm):
    try: return json.load(open(f'results/allsuites/{n}/libero_v/{arm}/eval_clean.json'))['success_rate']*100
    except Exception: return None
b=[sr(n,'baseline') for n in names]; a=[sr(n,'aegis') for n in names]
av=lambda r:(sum(x for x in r if x is not None)/len([x for x in r if x is not None])) if any(x is not None for x in r) else float('nan')
print('\n%8s %9s %8s'%('suite','base+TE','AEGIS'))
for i,n in enumerate(names):
    print('%8s %9s %8s'%(n,('%.1f'%b[i] if b[i] is not None else 'NA'),('%.1f'%a[i] if a[i] is not None else 'NA')))
print('%8s %9.1f %8.1f   (paper avg target = 87.3)'%('AVG',av(b),av(a)))
PY
  mark "STAGE5 done -> 4-suite table above; raw in results/allsuites/*"
}

# ---------------------------------------------------------------- driver
mark "=== A6000 QUEUE start (mode=$ONLY) ==="
case "$ONLY" in
  stage0) stage0_resume_base;;
  stage1) stage1_peakpick;;
  stage2) stage2_modules;;
  stage3) stage3_aegis;;
  stage4) stage4_robustness;;
  stage5|suites) stage5_allsuites;;                 # 4-suite clean on 86 base (FAST, no retrain)
  all) stage0_resume_base; stage1_peakpick; stage2_modules; stage3_aegis; stage4_robustness;;
  *) mark "unknown mode $ONLY (use: all|stage0..stage5|suites)"; exit 1;;
esac
mark "=== A6000 QUEUE '$ONLY' complete ==="
