#!/bin/bash
# Lean culprit isolation for the AEGIS Long t3/t4 collapse.
# Known from the main run: base+TE = t3:85 t4:35 ;  full AEGIS = t3:30 t4:10.
# We only need the two isolation arms to attribute the collapse:
#   rib_only  (RIB on, RASF off)   vs   rasf_only (RIB off via fusion-scale 0, RASF on)
# n=10/task (collapse signal is huge: 85->30), CONC=1. Each arm -> own output_dir.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
B86=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
RIB=results/ib_on86/rib_on86.pt; RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"; TASKS="3"; EP=10

run(){ local name=$1; shift
  local od=results/diag_long/$name
  [ -f "$od/libero_v/aegis/eval_clean.json" ] && { echo "[$(date +%T)] skip $name (done)"; return; }
  mkdir -p "$od"
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
  $PY scripts/eval_libero_v.py --config configs/_diag_${name}.yaml --method aegis $TE \
      --n-action-steps 1 --episodes $EP --tasks $TASKS --only clean "$@" > "$od/run.log" 2>&1
  echo "[$(date +%T)] DONE $name"
}

run rib_only_n10   --rib-weights $RIB                                          # RASF off
run rasf_only_n10  --rib-weights $RIB --rib-fusion-scale 0 --rasf-weights $RASF # RIB off

echo; echo "=== CULPRIT VERDICT (t3/t4, n=10) ==="
$PY - <<'PY'
import json,glob
print('%-14s %5s %5s'%('arm','t3','t4'))
print('%-14s %5s %5s   (known)'%('base+TE',85,35))
for n in ['rib_only_n10','rasf_only_n10']:
    js=glob.glob(f'results/diag_long/{n}/libero_v/*/eval_clean.json')
    if js:
        d=json.load(open(js[0])); pt={t['task_id']:round(t['success_rate']*100) for t in d.get('per_task',[])}
        print('%-14s %5s %5s'%(n,pt.get(3),pt.get(4)))
    else: print('%-14s %5s %5s'%(n,'-','-'))
print('%-14s %5s %5s   (known)'%('full AEGIS',30,10))
print('\nRead: arm matching ~30/10 = the culprit module; arm matching ~85/35 = harmless.')
PY
