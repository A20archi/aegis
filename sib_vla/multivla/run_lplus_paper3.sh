#!/usr/bin/env bash
# LIBERO-Plus paper run v3 — AFTER the instruction-suffix bug fix.
# Order: SMOKE-GATE (verify fix recovers spatial/goal/long baseline) -> headline stage1
#        (24 cells, per_cat 12, fixed code) -> pairwise videos.
# Clean SR is DROPPED here (we already have seed42 clean numbers; budget priority is the
# LIBERO-Plus headline + videos). HARD_TOTAL=195 (near-full $200, $5 safety).
set -uo pipefail
cd "$(dirname "$0")"
HT=${HT:-195}
LOG(){ echo "[paper3 $(date '+%F %T')] $*"; }
run(){ local g="$1"; shift; HARD_TOTAL=$HT BUDGET_GUARD="$g" ./safe_modal_run.sh "$@"; }

# ---- SMOKE GATE: the instruction fix MUST lift spatial/goal/long baseline off the ~0-6% floor.
LOG "SMOKE — instruction-fix check (spatial/goal/long/object baseline, per_cat=1)"
run 4 modal run lplus_modal/lplus_modal.py::main --stage smoke ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling hit during smoke (rc=9) — STOP"; exit 9; }
SLOG=$(ls -t /tmp/safe_modal_*.log 2>/dev/null | head -1)
fail=0
for s in libero_spatial libero_goal libero_10; do
  sr=$(grep -E "SMOKE +$s .*SR=" "$SLOG" | grep -oE "SR=[0-9.]+" | head -1 | cut -d= -f2)
  LOG "  smoke $s baseline SR=${sr:-MISSING}"
  if [ -z "$sr" ] || awk -v v="$sr" 'BEGIN{exit !(v+0 < 30)}'; then
    LOG "  -> $s baseline still <30% (or missing) — FIX FAILED"; fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  LOG "SMOKE GATE FAILED — instruction fix did not recover the broken suites. ABORTING, no big spend."
  exit 1
fi
LOG "SMOKE GATE PASSED — spatial/goal/long baseline recovered. Proceeding to headline."

# ---- HEADLINE: full LIBERO-Plus table, fixed code, 8-wide.
LOG "PHASE B — LIBERO-Plus HEADLINE (4 suites × 3 seeds × 2 arms, per_cat 12 = 24 cells)"
run 60 modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 12 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling hit during headline (rc=9) — STOP (resume-skip keeps done cells)."; exit 9; }

LOG "PHASE C — LIBERO-Plus-native pairwise videos (object, 7 cats × {base,aegis})"
run 12 modal run lplus_modal/lplus_modal.py::main --stage video ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling hit during videos (rc=9) — STOP."; exit 9; }

LOG "===== LIBERO-Plus PAPER RUN v3 COMPLETE (headline + videos, fixed instructions) ====="
