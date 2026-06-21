#!/bin/bash
# STAGE 2 — LIBERO-Plus on AEGIS (+ baseline), per-category 50, object + goal.
# 7 perturbation categories sampled at 50 tasks each, both arms, both suites = 4
# processes (each runs its 350 tasks serially in the isolated lerobot_lplus env).
set -uo pipefail
cd "$(dirname "$0")"
source ./run_queue_core.sh
PERCAT=${PERCAT:-50}
JL=$(mktemp); od=results/liberoplus; mkdir -p "$od"
for suite in libero_object libero_goal; do
  for method in baseline aegis; do
    echo -e "$od/${suite}_${method}.log\tMETHOD=$method SUITE=$suite PERCAT=$PERCAT bash run_libero_plus_aegis.sh" >> "$JL"
  done
done
run_stage "LIBERO-Plus object+goal (per-cat $PERCAT)" "$JL"
rm -f "$JL"
echo "[$(date +%T)] STAGE 2 COMPLETE — logs in results/liberoplus/"
