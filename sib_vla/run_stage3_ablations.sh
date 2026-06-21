#!/bin/bash
# STAGE 3 — Per-locus ablations on LIBERO-Spatial (the modules' training
# distribution), 6 corruption axes, n=200. Attributes each axis's gain to the
# right mechanism. baseline=SmolVLA+TE and aegis=full are already measured; here:
#   vanilla    : no TE, no modules                      (floor)
#   rib_only   : aegis + RASF disabled (gate_max=0)     (perception leg alone)
#   rasf_only  : aegis + RIB disabled  (fusion_scale=0) (action leg alone)
# Each arm gets its OWN output_dir so JSONs don't collide.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./run_queue_core.sh
PY=/home/user/anaconda3/bin/python
RIB=results/ib_on86/rib_on86.pt
RASF=results/rasf_on86/rasf_on86.pt
TE="--forge-ensemble --ensemble-coeff 0.01"
EP=${EP:-20}
CONDS="gaussian_noise_1 motion_blur_1 lighting_1 texture_1 viewpoint_medium viewpoint_large"
TASKS=0,1,2,3,4,5,6,7,8,9
# read base checkpoint + suite from the headline spatial config so the arm configs match it
BASECFG=configs/ib_on86.yaml

mk_cfg(){  # $1=arm -> writes configs/_abl_<arm>.yaml pointing output_dir at results/ablations/<arm>
  local arm=$1
  $PY - "$BASECFG" "$arm" <<'PY'
import sys,yaml,os
base=yaml.safe_load(open(sys.argv[1])); arm=sys.argv[2]
base["output_dir"]=f"results/ablations/{arm}"
base.setdefault("n_action_steps",1)
open(f"configs/_abl_{arm}.yaml","w").write(yaml.safe_dump(base,sort_keys=False))
PY
}
for arm in vanilla ribonly rasfonly; do mk_cfg "$arm"; done

JL=$(mktemp)
for c in $CONDS; do
  echo -e "results/ablations/vanilla/${c}.log\t$PY scripts/eval_libero_v.py --config configs/_abl_vanilla.yaml --method vanilla --n-action-steps 1 --episodes $EP --tasks $TASKS --only $c" >> "$JL"
  echo -e "results/ablations/ribonly/${c}.log\t$PY scripts/eval_libero_v.py --config configs/_abl_ribonly.yaml --method aegis --rib-weights $RIB --rasf-weights $RASF --rasf-gate-max 0 $TE --n-action-steps 1 --episodes $EP --tasks $TASKS --only $c" >> "$JL"
  echo -e "results/ablations/rasfonly/${c}.log\t$PY scripts/eval_libero_v.py --config configs/_abl_rasfonly.yaml --method aegis --rib-weights $RIB --rasf-weights $RASF --rib-fusion-scale 0 $TE --n-action-steps 1 --episodes $EP --tasks $TASKS --only $c" >> "$JL"
done
mkdir -p results/ablations/vanilla results/ablations/ribonly results/ablations/rasfonly
run_stage "Ablations (Spatial, 6 axes)" "$JL"
rm -f "$JL"

$PY - <<'PY'
import json,glob
print(f"{'axis':18s}{'vanilla':>9}{'RIB-only':>10}{'RASF-only':>11}")
axes=['gaussian_noise_1','motion_blur_1','lighting_1','texture_1','viewpoint_medium','viewpoint_large']
def sr(arm,c):
    m='vanilla' if arm=='vanilla' else 'aegis'
    g=glob.glob(f'results/ablations/{arm}/libero_v/{m}/eval_{c}.json')
    if not g: return None
    try: return json.load(open(g[0]))['success_rate']*100
    except Exception: return None
for c in axes:
    v=sr('vanilla',c); r=sr('ribonly',c); a=sr('rasfonly',c)
    f=lambda x:f'{x:5.1f}' if x is not None else '  -- '
    print(f"{c:18s}{f(v):>9}{f(r):>10}{f(a):>11}")
PY
echo "[$(date +%T)] STAGE 3 COMPLETE — results/ablations/"
