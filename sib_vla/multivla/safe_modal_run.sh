#!/usr/bin/env bash
# ============================================================================
# safe_modal_run.sh — the ONLY sanctioned way to launch a Modal run.
#
# Exists because two preventable runaways once ate 93% of a $100 budget:
#   (1) an UNCAPPED 24-cell burst        -> fixed by max_containers in app code
#   (2) an ORPHANED `modal run` client   -> `modal app stop` does NOT kill the
#       local client; it silently re-drove Modal for ~11h.
#
# This wrapper makes BOTH impossible:
#   * PREFLIGHT  — kills any stray clients + stops any running apps (clean slate)
#   * TRAP       — kills the launched client + stops its app on ANY exit
#                  (normal end, Ctrl-C, kill, error) -> a client CANNOT orphan
#   * WATCHDOG   — polls real billing; auto-aborts if THIS run adds > BUDGET_GUARD
#                  or month-to-date crosses HARD_TOTAL
#
# Usage:
#   ./safe_modal_run.sh modal run smolvla_modal/smolvla_modal.py::main --stage stage1
#   BUDGET_GUARD=10 HARD_TOTAL=95 ./safe_modal_run.sh modal run ...
# ============================================================================
set -uo pipefail

BUDGET_GUARD="${BUDGET_GUARD:-15}"   # $ this single run may add before auto-abort
HARD_TOTAL="${HARD_TOTAL:-95}"       # $ absolute month-to-date ceiling -> auto-abort
POLL="${POLL:-45}"                   # billing poll interval (seconds)

log(){ echo "[safe-modal $(date +%H:%M:%S)] $*"; }
billing_total(){ modal billing report --for "this month" 2>/dev/null \
  | grep -oE '[0-9]+\.[0-9]{4,}' | awk '{s+=$1} END{printf "%.2f", s+0}'; }
running_apps(){ modal app list --json 2>/dev/null | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
rows=d if isinstance(d,list) else d.get('apps',d)
for a in rows:
    st=str(a.get('state','')).lower()
    if 'stop' not in st and 'done' not in st:
        print(a.get('app_id') or a.get('id',''))"; }
stop_all_running(){ for a in $(running_apps); do [ -n "$a" ] && modal app stop "$a" --yes >/dev/null 2>&1 && log "stopped app $a"; done; }
kill_all_clients(){ pkill -f "modal run" 2>/dev/null; sleep 1; pkill -9 -f "modal run" 2>/dev/null; true; }

if [ $# -eq 0 ]; then echo "usage: $0 modal run <app>::main --stage <stage> [...]"; exit 2; fi

# ---- PREFLIGHT: clean slate -------------------------------------------------
log "PREFLIGHT — killing stray 'modal run' clients + stopping running apps"
kill_all_clients; stop_all_running
START_TOTAL="$(billing_total)"
log "month-to-date at start: \$$START_TOTAL  | guard: +\$$BUDGET_GUARD this run, hard ceiling \$$HARD_TOTAL"
# refuse to even start if already at/over the hard ceiling
if awk -v t="$START_TOTAL" -v h="$HARD_TOTAL" 'BEGIN{exit !(t+0 >= h+0)}'; then
  log "ABORT: month-to-date \$$START_TOTAL already >= hard ceiling \$$HARD_TOTAL. Not launching."; exit 9
fi

# ---- LAUNCH (backgrounded so we can watchdog it) ---------------------------
RUNLOG="/tmp/safe_modal_$(date +%s).log"
( "$@" ) > "$RUNLOG" 2>&1 &
RUN_PID=$!
APP_ID=""
log "LAUNCHED pid=$RUN_PID  log=$RUNLOG  cmd: $*"

# ---- GUARANTEED TEARDOWN on ANY exit ---------------------------------------
cleanup(){
  trap - EXIT INT TERM
  log "TEARDOWN — stopping app + killing client (idempotent)"
  [ -n "$APP_ID" ] && modal app stop "$APP_ID" --yes >/dev/null 2>&1
  kill "$RUN_PID" 2>/dev/null; pkill -P "$RUN_PID" 2>/dev/null
  sleep 2; kill -9 "$RUN_PID" 2>/dev/null; pkill -9 -P "$RUN_PID" 2>/dev/null
  kill_all_clients; stop_all_running
  log "TEARDOWN done. month-to-date now: \$$(billing_total).  clients: $(pgrep -fc 'modal run' 2>/dev/null || echo 0) running."
}
trap cleanup EXIT INT TERM

# ---- WATCHDOG ---------------------------------------------------------------
while kill -0 "$RUN_PID" 2>/dev/null; do
  sleep "$POLL"
  [ -z "$APP_ID" ] && APP_ID="$(grep -oE 'ap-[A-Za-z0-9]+' "$RUNLOG" 2>/dev/null | head -1)"
  NOW="$(billing_total)"
  DELTA="$(awk -v a="$NOW" -v b="$START_TOTAL" 'BEGIN{printf "%.2f", a-b}')"
  log "spend: month \$$NOW  (this run +\$$DELTA)  app=${APP_ID:-pending}"
  if awk -v d="$DELTA" -v g="$BUDGET_GUARD" 'BEGIN{exit !(d+0 >= g+0)}'; then
    log "!! BUDGET_GUARD TRIPPED: this run added \$$DELTA >= \$$BUDGET_GUARD — ABORTING NOW"; exit 9
  fi
  if awk -v t="$NOW" -v h="$HARD_TOTAL" 'BEGIN{exit !(t+0 >= h+0)}'; then
    log "!! HARD_TOTAL TRIPPED: month \$$NOW >= \$$HARD_TOTAL — ABORTING NOW"; exit 9
  fi
done
wait "$RUN_PID"; RC=$?
log "run process exited rc=$RC"
exit "$RC"
