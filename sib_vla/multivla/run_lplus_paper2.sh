#!/usr/bin/env bash
# Re-prioritized LIBERO-Plus paper run — HEADLINE FIRST, maximize results under $200.
# Decision (2026-06-22): full clean + full LIBERO-Plus + videos do NOT fit under the
# ceiling from $118.91, so spend the headroom on the paper's headline first.
#
#   Order: LIBERO-Plus (headline §5.2, 3-seed) -> pairwise videos -> clean SR (secondary
#          §5.1, soaks whatever budget remains; fully resume-skipped so partial is fine).
#
# Each phase via safe_modal_run.sh (preflight clean-slate, trap kill-on-exit, billing
# watchdog auto-abort at HARD_TOTAL). HARD_TOTAL=195 = near-full $200 with a $5 safety
# buffer. Per-phase BUDGET_GUARD is generous so only the month ceiling can cut the
# headline; clean (last) is allowed to soak to the ceiling.
set -uo pipefail
cd "$(dirname "$0")"                       # sib_vla/multivla
HT=${HT:-195}                              # absolute month-to-date ceiling ($200 − 5)
LOG(){ echo "[paper2 $(date '+%F %T')] $*"; }
run(){ local g="$1"; shift; HARD_TOTAL=$HT BUDGET_GUARD="$g" ./safe_modal_run.sh "$@"; }

LOG "PHASE B — LIBERO-Plus HEADLINE (4 suites × 3 seeds × 2 arms, per_cat 12 = 24 cells, max_containers=8)"
run 60 modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 12 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling hit during LIBERO-Plus (rc=9) — STOP (resume-skip keeps done cells). Headline got priority."; exit 9; }

LOG "PHASE C — LIBERO-Plus-native pairwise videos (object, 7 cats × {base,aegis})"
run 12 modal run lplus_modal/lplus_modal.py::main --stage video ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "HARD ceiling hit during videos (rc=9) — STOP."; exit 9; }

LOG "PHASE A — clean SR (secondary §5.1, soaks remaining budget; resume-skips cached cells)"
run 40 modal run smolvla_modal/smolvla_modal.py::main --stage clean --episodes 10 ; rc=$?
[ "${rc:-0}" -eq 9 ] && { LOG "Budget slice used on clean (rc=9) — that's fine, clean is secondary + resumable. STOP."; exit 0; }

LOG "===== RE-PRIORITIZED PAPER RUN COMPLETE (headline LIBERO-Plus + videos + clean) ====="
