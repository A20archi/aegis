#!/usr/bin/env bash
# Tier-1 Long bridge: RIB fusion-strength sweep (no retrain). mult=0 == base (identity).
# Finds the strength where Long clean >= base while keeping the sensor/light robustness.
set -uo pipefail
cd "$(dirname "$0")"; ACT=$PWD; SIB=$(dirname "$ACT")
LROOT=/home/user/Desktop/vla_projects/LIBERO-plus
LENV=/home/user/miniconda3/envs/lerobot; LPENV=/home/user/miniconda3/envs/lerobot_lplus
PY=$LENV/bin/python; PYLP=$LPENV/bin/python
RIB=$SIB/results/aegis_act_v2/Long/rib.pt
CK=$SIB/act_ckpts/Long/act/30000
MULTS=${MULTS:-"0.0 0.25 0.5 0.75 1.0"}; HARDCAP=${HARDCAP:-4}; SEED=42
JOBS=$(mktemp)
for m in $MULTS; do
  oc=$SIB/results/long_bridge/clean/m$m
  echo -e "$oc/run.log\tMUJOCO_GL=egl HF_HUB_OFFLINE=0 PYTHONPATH=$ACT:$SIB $PY -u $ACT/clean_eval_aegis.py --suite libero_10 --dataset lerobot/libero_10_image --base-ckpt $CK --arm aegis --rib-weights $RIB --fusion-mult $m --seed $SEED --episodes 10 --max-steps 520 --videos 0 --out $oc" >> "$JOBS"
  op=$SIB/results/long_bridge/plus/m$m
  echo -e "$op/run.log\tMUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=0 PYTHONPATH=$LROOT:$ACT:$SIB LIBERO_CONFIG_PATH=/home/user/Desktop/vla_projects/.libero_lplus MAGICK_HOME=$LPENV LD_LIBRARY_PATH=$LPENV/lib $PYLP -u $ACT/plus_eval_aegis.py --suite libero_10 --dataset lerobot/libero_10_image --base-ckpt $CK --arm aegis --rib-weights $RIB --fusion-mult $m --seed $SEED --n-per-cat 6 --max-steps 520 --out $op" >> "$JOBS"
done
N=$(wc -l < "$JOBS"); echo "[long-sweep] $N jobs (mults=[$MULTS], HARDCAP=$HARDCAP)"
while IFS=$'\t' read -r log cmd; do
  mkdir -p "$(dirname "$log")"; ( eval "$cmd" > "$log" 2>&1; echo "[done] $log" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$HARDCAP" ]; do sleep 10; done
done < "$JOBS"; wait; rm -f "$JOBS"
echo "[long-sweep] DONE -> results/long_bridge/"
# quick summary
$PY - <<'PY'
import json,glob,os
print("\n=== LONG FUSION SWEEP (seed42) ===")
print(f"{'mult':>5s} {'cleanSR':>8s} {'robustAvg':>10s}")
for m in ["0.0","0.25","0.5","0.75","1.0"]:
    c=glob.glob(f"results/long_bridge/clean/m{m}/aegis/seed42/result.json")
    p=glob.glob(f"results/long_bridge/plus/m{m}/aegis/seed42/result.json")
    cs=json.load(open(c[0]))['average']*100 if c and json.load(open(c[0])).get('average') is not None else None
    pj=json.load(open(p[0])) if p else None
    ps=pj.get('robustness_average') if pj else None
    print(f"{m:>5s} {('%.1f'%cs) if cs is not None else '-':>8s} {('%.1f'%(ps*100)) if ps is not None else '-':>10s}")
print("mult=0.0 == base (identity). Find smallest mult with cleanSR>=base AND robustAvg up.")
PY
