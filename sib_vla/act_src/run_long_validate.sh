#!/usr/bin/env bash
# 3-seed validation of de-strengthed Long RIB at mult={0.25,0.5}. Base = main-sweep Long base
# (act_clean_v2/Long/base, act_plus_v2/Long/base). aegis arm here = de-strengthed RIB.
set -uo pipefail
cd "$(dirname "$0")"; ACT=$PWD; SIB=$(dirname "$ACT")
LROOT=/home/user/Desktop/vla_projects/LIBERO-plus
LENV=/home/user/miniconda3/envs/lerobot; LPENV=/home/user/miniconda3/envs/lerobot_lplus
PY=$LENV/bin/python; PYLP=$LPENV/bin/python
RIB=$SIB/results/aegis_act_v2/Long/rib.pt; CK=$SIB/act_ckpts/Long/act/30000
MULTS=${MULTS:-"0.25 0.5"}; SEEDS=${SEEDS:-"42 123 456"}; HARDCAP=${HARDCAP:-6}
JOBS=$(mktemp)
for m in $MULTS; do for sd in $SEEDS; do
  oc=$SIB/results/long_bridge_v/m$m/clean
  echo -e "$oc/aegis_s$sd.log\tMUJOCO_GL=egl HF_HUB_OFFLINE=0 PYTHONPATH=$ACT:$SIB $PY -u $ACT/clean_eval_aegis.py --suite libero_10 --dataset lerobot/libero_10_image --base-ckpt $CK --arm aegis --rib-weights $RIB --fusion-mult $m --seed $sd --episodes 20 --max-steps 520 --videos 0 --out $oc" >> "$JOBS"
  op=$SIB/results/long_bridge_v/m$m/plus
  echo -e "$op/aegis_s$sd.log\tMUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=0 PYTHONPATH=$LROOT:$ACT:$SIB LIBERO_CONFIG_PATH=/home/user/Desktop/vla_projects/.libero_lplus MAGICK_HOME=$LPENV LD_LIBRARY_PATH=$LPENV/lib $PYLP -u $ACT/plus_eval_aegis.py --suite libero_10 --dataset lerobot/libero_10_image --base-ckpt $CK --arm aegis --rib-weights $RIB --fusion-mult $m --seed $sd --n-per-cat 12 --max-steps 520 --out $op" >> "$JOBS"
done; done
N=$(wc -l < "$JOBS"); echo "[long-val] $N jobs (mults=[$MULTS] seeds=[$SEEDS])"
while IFS=$'\t' read -r log cmd; do
  mkdir -p "$(dirname "$log")"; ( eval "$cmd" > "$log" 2>&1; echo "[done] $log" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$HARDCAP" ]; do sleep 10; done
done < "$JOBS"; wait; rm -f "$JOBS"
echo "[long-val] DONE -> results/long_bridge_v/"
