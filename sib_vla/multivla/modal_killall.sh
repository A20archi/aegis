#!/usr/bin/env bash
# modal_killall.sh — emergency teardown. Kills every local `modal run` client
# AND stops every running Modal app, then prints month-to-date spend.
# Run this any time you are unsure whether something is still spending.
set -uo pipefail
log(){ echo "[killall $(date +%H:%M:%S)] $*"; }

log "killing local 'modal run' clients..."
pkill -f "modal run" 2>/dev/null; sleep 2; pkill -9 -f "modal run" 2>/dev/null
log "remaining clients: $(pgrep -fc 'modal run' 2>/dev/null || echo 0)"

log "stopping running apps..."
for a in $(modal app list --json 2>/dev/null | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
rows=d if isinstance(d,list) else d.get('apps',d)
for a in rows:
    st=str(a.get('state','')).lower()
    if 'stop' not in st and 'done' not in st:
        print(a.get('app_id') or a.get('id',''))"); do
  [ -n "$a" ] && modal app stop "$a" --yes >/dev/null 2>&1 && log "  stopped $a"
done

TOTAL="$(modal billing report --for "this month" 2>/dev/null | grep -oE '[0-9]+\.[0-9]{4,}' | awk '{s+=$1} END{printf "%.2f", s+0}')"
log "DONE. month-to-date spend: \$$TOTAL  | clients: $(pgrep -fc 'modal run' 2>/dev/null || echo 0) | check: modal app list"
