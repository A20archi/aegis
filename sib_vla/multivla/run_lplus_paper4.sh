#!/usr/bin/env bash
# LIBERO-Plus paper run v4 — AFTER the fix was UPLOADED TO THE VOLUME (verified) + re-smoked.
# v3 failed because Modal runs the eval from /assets/sib_vla on the volume, not the local
# edit; the fix is now pushed. Order: SMOKE-GATE (prove fix on Modal) -> headline stage1
# (per_cat 12) -> clean soak (extra seeds, resume-skips seed42 n=100). HARD_TOTAL=195.
set -uo pipefail
cd "$(dirname "$0")"
HT=${HT:-195}
LOG(){ echo "[paper4 $(date '+%F %T')] $*"; }
run(){ local g="$1"; shift; HARD_TOTAL=$HT BUDGET_GUARD="$g" ./safe_modal_run.sh "$@"; }

# ---- PROOF SMOKE: spatial/goal/long baseline MUST recover off the bug floor (>30%).
# Purge the smoke od FIRST: resume-skip keys only on n_episodes>0 (ignores per_cat/code
# version), so a stale smoke JSON would be falsely reported instead of re-running fixed code.
LOG "purge stale smoke cache (force fixed-code re-run)"
modal volume rm smolvla-assets results_modal/liberoplus_smoke -r >/dev/null 2>&1 || true
LOG "SMOKE — proof the volume fix works (spatial/goal/long/object baseline, per_cat=2)"
run 4 modal run lplus_modal/lplus_modal.py::main --stage smoke ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling during smoke (rc=9) — STOP"; exit 9; }
SLOG=$(ls -t /tmp/safe_modal_*.log 2>/dev/null | head -1)
fail=0
for s in libero_spatial libero_goal libero_10; do
  sr=$(grep -E "SMOKE +$s .*SR=" "$SLOG" | grep -oE "SR=[0-9.]+" | head -1 | cut -d= -f2)
  LOG "  smoke $s baseline SR=${sr:-MISSING}"
  if [ -z "$sr" ] || awk -v v="$sr" 'BEGIN{exit !(v+0 < 30)}'; then
    LOG "  -> $s baseline still <30% (or missing) — FIX STILL BROKEN"; fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  LOG "SMOKE GATE FAILED — NOT spending on the headline. Investigate."; exit 1
fi
LOG "SMOKE GATE PASSED — fix confirmed on Modal. Proceeding."

# ---- HEADLINE: full LIBERO-Plus table (fixed code, 8-wide).
LOG "PHASE B — LIBERO-Plus HEADLINE (4 suites × 3 seeds × 2 arms, per_cat 12 = 24 cells)"
run 60 modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 12 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling hit during headline (rc=9) — STOP (resume-skip keeps done cells)."; exit 9; }

# ---- CLEAN soak: extra seeds (123,456) at n=100; resume-skips the protected seed42 n=100.
#      Whatever fits under $195 completes; the rest is resume-skippable later (free local).
LOG "PHASE A — clean SR soak (extra seeds; seed42 n=100 protected by resume-skip)"
run 40 modal run smolvla_modal/smolvla_modal.py::main --stage clean --episodes 10 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "Budget exhausted on clean (rc=9) — fine, resumable. STOP."; exit 0; }

LOG "===== LIBERO-Plus PAPER RUN v4 COMPLETE (headline + clean, fixed/verified code) ====="
