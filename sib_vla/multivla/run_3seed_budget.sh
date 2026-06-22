#!/usr/bin/env bash
# BOTH tables 3-seeded (42,123,456), BOTH within budget. Fix verified by the proof-smoke
# (spatial/goal/long baseline off the bug floor). All 3 seeds kept; per-cell n shrunk so it
# fits the cap — still a genuine multi-seed result, NOT single-seed.
#   PHASE A: 3-seed CLEAN completion  (seeds 123,456 @ n=50; seed42 n=100 resume-skipped)
#   PHASE B: 3-seed LIBERO-Plus        (4 suites × 3 seeds × 2 arms, per_cat 6)
# HARD_TOTAL=195 watchdog => spend physically cannot exceed $200.
set -uo pipefail
cd "$(dirname "$0")"
HT=${HT:-195}
LOG(){ echo "[3seed $(date '+%F %T')] $*"; }
run(){ local g="$1"; shift; HARD_TOTAL=$HT BUDGET_GUARD="$g" ./safe_modal_run.sh "$@"; }

LOG "PHASE A — 3-seed CLEAN completion (seeds 123,456 @ n=50; seed42 n=100 protected)"
run 30 modal run smolvla_modal/smolvla_modal.py::main --stage cleanfill ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling during clean-fill (rc=9) — STOP (resume-skippable)."; exit 9; }

LOG "PHASE B — 3-seed LIBERO-Plus (4 suites × 3 seeds × 2 arms, per_cat 6 = 24 cells)"
run 35 modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 6 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling during LIBERO-Plus (rc=9) — STOP (resume-skip keeps done cells)."; exit 9; }

LOG "===== 3-SEED RUN COMPLETE — clean 3-seed + LIBERO-Plus 3-seed, within budget ====="
