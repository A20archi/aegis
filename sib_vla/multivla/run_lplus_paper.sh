#!/usr/bin/env bash
# Full LIBERO-Plus paper run, fully automated + guarded. Each phase via safe_modal_run.sh
# (preflight clean-slate, trap kill-on-exit, billing watchdog auto-abort at HARD_TOTAL).
# Order: smoke -> [GATE] -> clean SR -> LIBERO-Plus -> pairwise videos.
# The smoke GATE blocks the big spend if the pipeline is broken (rc!=0 or eval error).
set -uo pipefail
cd "$(dirname "$0")"                       # sib_vla/multivla
HT=${HT:-190}                              # absolute month-to-date ceiling ($200 limit − 10)
LOG(){ echo "[paper $(date '+%F %T')] $*"; }

run(){ local g="$1"; shift; HARD_TOTAL=$HT BUDGET_GUARD="$g" ./safe_modal_run.sh "$@"; }

LOG "PHASE 0 — smoke (lplus, calibrate + validate pipeline)"
run 2 modal run lplus_modal/lplus_modal.py::main --stage smoke ; rc=$?
SLOG=$(ls -t /tmp/safe_modal_*.log 2>/dev/null | head -1)
if [ "${rc:-1}" -ne 0 ] || grep -qE "Traceback \(most recent|unrecognized arguments|'err':|Error:" "${SLOG:-/dev/null}" 2>/dev/null; then
  LOG "SMOKE FAILED (rc=$rc) — ABORTING chain, NO big spend. log=$SLOG"; exit 1
fi
LOG "SMOKE OK (log=$SLOG) — proceeding to full run"

LOG "PHASE A — clean SR (4 suites × 3 seeds × 2 arms = 24 cells)"
run 24 modal run smolvla_modal/smolvla_modal.py::main --stage clean --episodes 10 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "GUARD tripped on clean (rc=9) — STOP"; exit 9; }

LOG "PHASE B — LIBERO-Plus (4 suites × 3 seeds × 2 arms, per_cat 40 = 24 cells)"
run 85 modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 20 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "GUARD tripped on LIBERO-Plus (rc=9) — STOP (resume-skip keeps done cells)"; exit 9; }

LOG "PHASE C — LIBERO-Plus-native pairwise videos (object, 7 cats × {base,aegis})"
run 8 modal run lplus_modal/lplus_modal.py::main --stage video ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "GUARD tripped on videos (rc=9) — STOP"; exit 9; }

LOG "===== LIBERO-Plus PAPER RUN COMPLETE (clean + robustness + videos) ====="
