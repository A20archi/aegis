#!/usr/bin/env bash
# Full base-vs-AEGIS sweep on the colleague's ACT: clean (3-seed) + LIBERO-Plus (3-seed x 7 cat).
# Parallel job queue, HARDCAP concurrent. Resumable (drivers skip completed tasks).
set -uo pipefail
cd "$(dirname "$0")"; ACT=$PWD; SIB=$(dirname "$ACT")
LROOT=/home/user/Desktop/vla_projects/LIBERO-plus
LENV=/home/user/miniconda3/envs/lerobot; LPENV=/home/user/miniconda3/envs/lerobot_lplus
PY=$LENV/bin/python; PYLP=$LPENV/bin/python
SEEDS=${SEEDS:-"42 123 456"}; HARDCAP=${HARDCAP:-6}; NPC=${NPC:-12}; EP=${EP:-20}
declare -A DS=( [Spatial]=libero_spatial_image [Object]=libero_object_image [Goal]=libero_goal_image [Long]=libero_10_image )
declare -A SUITE=( [Spatial]=libero_spatial [Object]=libero_object [Goal]=libero_goal [Long]=libero_10 )
declare -A PMS=( [Spatial]=300 [Object]=300 [Goal]=300 [Long]=520 )   # lplus max-steps
CKPT(){ echo "$SIB/act_ckpts/$1/act/30000"; }
RIB(){ echo "$SIB/results/aegis_act_v2/$1/rib.pt"; }

# wait for all 4 RIB trainings to COMPLETE. rib.train.json is written only at the very end
# (after the final step-6000 save), so this never reads a step-3000 intermediate mid-train.
echo "[sweep] waiting for RIB training completion ..."
for S in Spatial Object Goal Long; do
  while [ ! -f "$SIB/results/aegis_act_v2/$S/rib.train.json" ]; do sleep 15; done
done
echo "[sweep] all RIB training complete. building job list."

JOBS=$(mktemp)
for S in Spatial Object Goal Long; do for sd in $SEEDS; do for arm in base aegis; do
  rw=""; [ "$arm" = "aegis" ] && rw="--rib-weights $(RIB $S)"
  # clean (lerobot env)
  od="$SIB/results/act_clean_v2/$S"
  echo -e "clean\t$od/${arm}_s${sd}.log\tMUJOCO_GL=egl HF_HUB_OFFLINE=0 PYTHONPATH=$ACT:$SIB $PY -u $ACT/clean_eval_aegis.py --suite ${SUITE[$S]} --dataset lerobot/${DS[$S]} --base-ckpt $(CKPT $S) --arm $arm $rw --seed $sd --episodes $EP --max-steps 520 --out $od" >> "$JOBS"
  # libero-plus (lerobot_lplus env)
  od2="$SIB/results/act_plus_v2/$S"
  echo -e "plus\t$od2/${arm}_s${sd}.log\tMUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=0 PYTHONPATH=$LROOT:$ACT:$SIB LIBERO_CONFIG_PATH=/home/user/Desktop/vla_projects/.libero_lplus MAGICK_HOME=$LPENV LD_LIBRARY_PATH=$LPENV/lib $PYLP -u $ACT/plus_eval_aegis.py --suite ${SUITE[$S]} --dataset lerobot/${DS[$S]} --base-ckpt $(CKPT $S) --arm $arm $rw --seed $sd --n-per-cat $NPC --max-steps ${PMS[$S]} --out $od2" >> "$JOBS"
done; done; done
N=$(wc -l < "$JOBS"); echo "[sweep] $N jobs queued (HARDCAP=$HARDCAP)"

run_one(){ local log="$1" cmd="$2"; mkdir -p "$(dirname "$log")"; eval "$cmd" > "$log" 2>&1; echo "[done] $(basename "$log")"; }
export -f run_one
# throttle to HARDCAP concurrent
while IFS=$'\t' read -r phase log cmd; do
  run_one "$log" "$cmd" &
  while [ "$(jobs -rp | wc -l)" -ge "$HARDCAP" ]; do sleep 10; done
done < "$JOBS"
wait
rm -f "$JOBS"
echo "[sweep] ALL DONE"
