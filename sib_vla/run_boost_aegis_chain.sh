#!/bin/bash
# CHAINED: waits for the baseline boost sweep (run_boost_eval_safe.sh) to finish,
# then runs AEGIS (RIB+RASF+TE, trained gates) on the SAME boosted base, all 4
# clean suites, SERIAL + OOM-guarded. Identity-residual guarantee => AEGIS can only
# hold or improve clean SR vs the boosted baseline.
#   baseline (already run) = boostedBase + TE
#   aegis   (this script)  = boostedBase + RIB + RASF + TE
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/user/anaconda3/bin/python
BOOST=results/lora_boost/smolvla_lora_boost
RIB=${RIB:-results/ib_on86/rib_on86.pt}
RASF=${RASF:-results/rasf_on86/rasf_on86.pt}
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-50}; NEED=${NEED:-9000}
names=(long spatial object goal)
suites=(libero_10 libero_spatial libero_object libero_goal)
eplen=(520 220 280 300)

[ -f "$RIB" ]  || { echo "FATAL: RIB ckpt missing ($RIB)";  exit 1; }
[ -f "$RASF" ] || { echo "FATAL: RASF ckpt missing ($RASF)"; exit 1; }

# 1) Block until the baseline boost sweep is gone.
echo "[$(date +%T)] AEGIS-chain armed; waiting for baseline sweep to finish..."
while pgrep -f run_boost_eval_safe.sh >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%T)] baseline sweep finished -> starting AEGIS-on-boosted"

wait_mem(){
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge "$NEED" ] && { echo "[$(date +%T)] GPU free=${free}MiB >= ${NEED} -> go"; break; }
    echo "[$(date +%T)] GPU free=${free}MiB < ${NEED}, waiting..."; sleep 60
  done
}

for i in "${!names[@]}"; do
  name=${names[$i]}; od=results/boost_aegis/$name; mkdir -p "$od"
  [ -f "$od/libero_v/aegis/eval_clean.json" ] && { echo "skip $name (done)"; continue; }
  cat > configs/_boostag_${name}.yaml <<YAML
inherit: base.yaml
checkpoint: $BOOST
output_dir: $od
suite: ${suites[$i]}
episode_length: ${eplen[$i]}
n_action_steps: 1
record:
  enabled: false
YAML
  wait_mem
  echo "[$(date +%T)] START aegis $name (serial, OOM-guarded)"
  $PY scripts/eval_libero_v.py --config "configs/_boostag_$name.yaml" --method aegis \
    --rib-weights "$RIB" --rasf-weights "$RASF" $TE \
    --n-action-steps 1 --episodes "$EP" --only clean >"$od/aegis.log" 2>&1
  sr=$($PY -c "import json;print('%.1f'%(json.load(open('$od/libero_v/aegis/eval_clean.json'))['success_rate']*100))" 2>/dev/null || echo ERR)
  echo "[$(date +%T)] DONE aegis $name SR=$sr"
done

echo "[$(date +%T)] === BOOST+AEGIS 4-suite complete ==="
$PY - <<'PY'
import json,os
paper={'spatial':None,'object':None,'goal':None,'long':None}
def sr(p):
    return json.load(open(p))['success_rate']*100 if os.path.exists(p) else None
rows=[]; bt=[]; at=[]
for n in ['spatial','object','goal','long']:
    b=sr(f'results/boost_base/{n}/libero_v/baseline/eval_clean.json')
    a=sr(f'results/boost_aegis/{n}/libero_v/aegis/eval_clean.json')
    rows.append((n,b,a))
    if b is not None: bt.append(b)
    if a is not None: at.append(a)
print(f'{"suite":8s}{"boostBase":>12}{"boost+AEGIS":>14}{"gap":>8}')
for n,b,a in rows:
    bs=f'{b:5.1f}' if b is not None else '  -- '
    as_=f'{a:5.1f}' if a is not None else '  -- '
    gp=f'{a-b:+5.1f}' if (a is not None and b is not None) else '   - '
    print(f'{n:8s}{bs:>12}{as_:>14}{gp:>8}')
if len(bt)==4: print(f'  boostBase AVG  {sum(bt)/4:.2f}  (paper 87.3)')
if len(at)==4: print(f'  boost+AEGIS AVG {sum(at)/4:.2f}  (paper 87.3)')
PY
