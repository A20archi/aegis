#!/bin/bash
# 4-suite clean eval (base+TE & AEGIS) on the 86 base, run PARALLEL on the A100 alongside
# the Hopfield/hamlet GRPO runs. Concurrency-capped so we never OOM the GPU or starve CPU.
# Restartable: skips any (suite,arm) whose eval_clean.json already exists.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
B86=outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-50}                         # 50 trials/task = proper LIBERO protocol (not casual 20)
CONC=${CONC:-4}
# LONG-FIRST (most important suite, launches in wave 1). Per-suite max-steps = standard
# OpenVLA/LIBERO protocol so it's a fair compare vs the paper's vanilla numbers:
#   Long(libero_10)=520  Spatial=220  Object=280  Goal=300   (Long was truncated at 300!)
names=(long spatial object goal)
suites=(libero_10 libero_spatial libero_object libero_goal)
eplen=(520 220 280 300)

for i in "${!names[@]}"; do
  od=results/allsuites/${names[$i]}; mkdir -p "$od"
  cat > configs/_allsuite_${names[$i]}.yaml <<EOF
inherit: base.yaml
checkpoint: $B86
output_dir: $od
suite: ${suites[$i]}
episode_length: ${eplen[$i]}
n_action_steps: 1
record:
  enabled: false
EOF
done

run_one(){
  local name=$1 arm=$2 cfg=configs/_allsuite_$1.yaml od=results/allsuites/$1
  [ -f "$od/libero_v/$arm/eval_clean.json" ] && { echo "[$(date +%T)] skip $name/$arm (done)"; return; }
  echo "[$(date +%T)] START $name/$arm"
  if [ "$arm" = baseline ]; then
    $PY scripts/eval_libero_v.py --config "$cfg" --method baseline $TE \
      --n-action-steps 1 --episodes "$EP" --only clean --record --videos-per-task 5 >"$od/${arm}.log" 2>&1
  else
    $PY scripts/eval_libero_v.py --config "$cfg" --method aegis \
      --rib-weights "$RIB" --rasf-weights "$RASF" $TE \
      --n-action-steps 1 --episodes "$EP" --only clean --record --videos-per-task 5 >"$od/${arm}.log" 2>&1
  fi
  echo "[$(date +%T)] DONE  $name/$arm"
}

echo "[$(date +%T)] === 4-suite parallel eval: EP=$EP CONC=$CONC (8 jobs) ==="
n=0
for name in "${names[@]}"; do for arm in baseline aegis; do
  run_one "$name" "$arm" &
  n=$((n+1)); [ $((n % CONC)) -eq 0 ] && wait -n 2>/dev/null
done; done
wait
echo "[$(date +%T)] === ALL EVALS DONE -> summary ==="
$PY - <<'PY'
import json
names=['spatial','object','goal','long']
sr=lambda n,a:(json.load(open(f'results/allsuites/{n}/libero_v/{a}/eval_clean.json'))['success_rate']*100) if __import__('os').path.exists(f'results/allsuites/{n}/libero_v/{a}/eval_clean.json') else None
b=[sr(n,'baseline') for n in names]; a=[sr(n,'aegis') for n in names]
av=lambda r:(sum(x for x in r if x is not None)/max(1,len([x for x in r if x is not None])))
print('%8s %9s %8s'%('suite','base+TE','AEGIS'))
for i,n in enumerate(names):
    print('%8s %9s %8s'%(n,('%.1f'%b[i] if b[i] is not None else 'NA'),('%.1f'%a[i] if a[i] is not None else 'NA')))
print('%8s %9.1f %8.1f   (paper target 87.3)'%('AVG',av(b),av(a)))
PY
