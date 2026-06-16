#!/bin/bash
# Canonical SAFE sweep runner. Makes the silent-failure class impossible:
#   1. PREFLIGHT gate  -- validate every job (config/weights/device paths) in
#      seconds; abort the whole sweep if any job would crash.
#   2. run jobs K-concurrent (render-bound -> default K=4, the measured knee).
#   3. POST-CHECK      -- verify every job wrote its result JSON; if any are
#      missing (a mid-run crash), print a LOUD failure summary and exit nonzero.
#
# Job file: one  "tag|config|weights|corruption|action_noise"  per line
#           (weights / corruption / action_noise may be empty).
# Usage:  bash scripts/run_sweep.sh jobs.txt [K]
set -u
cd "$(dirname "$0")/.."
JOBS_FILE="${1:?usage: run_sweep.sh jobs.txt [K]}"
K="${2:-4}"

echo "=== [1/3] PREFLIGHT $(date '+%H:%M') ==="
# 1a. device/path regression tests (the bug class) -- ~3s, runs on EVERY sweep.
echo "--- regression tests (tests/test_eval_paths.py) ---"
if ! python -m pytest tests/test_eval_paths.py -q >/tmp/_sweep_pytest.log 2>&1; then
  echo "!!! REGRESSION TESTS FAILED -- sweep NOT launched."; tail -20 /tmp/_sweep_pytest.log; exit 1
fi
echo "    $(grep -E 'passed|failed' /tmp/_sweep_pytest.log | tail -1)"
# 1b. validate the actual jobs (config/weights/device placement).
echo "--- job validation ---"
if ! python scripts/preflight.py "$JOBS_FILE"; then
  echo "!!! PREFLIGHT FAILED -- sweep NOT launched. Fix the jobs above." ; exit 1
fi

suffix() { # corruption action_noise -> filename suffix eval.py uses
  local c="$1" an="$2"
  if [ -n "$c" ]; then echo "__${c//:/}"; elif [ -n "$an" ] && [ "$an" != "0" ]; then echo "__action_noise${an}"; else echo ""; fi
}

echo "=== [2/3] RUN (K=$K) $(date '+%H:%M') ==="
declare -a EXP=()
while IFS='|' read -r tag cfg wts corr an; do
  [ -z "${tag// }" ] && continue; case "$tag" in \#*) continue;; esac
  while [ "$(jobs -rp | wc -l)" -ge "$K" ]; do sleep 5; done
  a=(--config "$cfg" --tag "$tag")
  [ -n "${wts:-}" ] && a+=(--weights "$wts")
  [ -n "${corr:-}" ] && a+=(--corruption "$corr")
  [ -n "${an:-}" ] && [ "$an" != "0" ] && a+=(--action-noise "$an")
  EXP+=("results/eval_${tag}$(suffix "${corr:-}" "${an:-}").json")
  echo ">>> [$(date '+%H:%M')] $tag ${corr:+corr=$corr}${an:+ an=$an}"
  env MUJOCO_GL=egl python -u scripts/eval.py "${a[@]}" > "results/log_sweep_${tag}$(suffix "${corr:-}" "${an:-}").txt" 2>&1 &
done < "$JOBS_FILE"
wait

echo "=== [3/3] POST-CHECK $(date '+%H:%M') ==="
miss=0
for f in "${EXP[@]}"; do
  if [ -f "$f" ]; then echo "  OK   $f"; else echo "  MISSING  $f"; miss=$((miss+1)); fi
done
if [ "$miss" -gt 0 ]; then
  echo "!!! $miss/${#EXP[@]} jobs produced NO result -- a crash slipped past preflight."
  echo "    Inspect results/log_sweep_*.txt for the traceback. DO NOT trust this sweep."
  exit 1
fi
echo "ALL ${#EXP[@]} jobs produced results. sweep clean."
