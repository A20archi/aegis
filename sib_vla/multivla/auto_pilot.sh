#!/usr/bin/env bash
# Self-driving Modal orchestrator for the SmolVLA robustness sweep. Self-contained:
# LAUNCHES the validate build, then chains GPU smoke -> stage1 (24-cell parallel sweep).
# Each stage advances ONLY if the previous one passed. STATUS.txt is the source of truth.
set -u
ROOT=/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla/multivla
SM=$ROOT/smolvla_modal
STATUS=$ROOT/STATUS.txt
LOG=$ROOT/auto_pilot.log

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(stamp)] $*" | tee -a "$LOG"; }
setst(){ echo "[$(stamp)] $*" >> "$STATUS"; }

: > "$STATUS"
say "AUTO-PILOT START (SmolVLA robustness chain)"
setst "AUTO-PILOT START"

wait_settle(){  # $1=logfile -> block until any terminal marker appears
  until grep -qE "egl_render_ok|VALIDATE:|SMOKE:|launching .* cells|Image build for .* failed|Traceback|ResolutionImpossible|Failed building wheel|No matching distribution|Error: |App completed|Stopping app - uncaught" "$1" 2>/dev/null; do
    sleep 8
  done
}

cd "$SM" || { setst "FATAL: cannot cd $SM"; exit 1; }

# ---- Stage A: build + validate the image (deps import + EGL headless render) ----
say "Stage A: launching validate build"
modal run smolvla_modal.py::main --stage validate > "$SM/smolvla_validate.log" 2>&1
if ! grep -q "'egl_render_ok': True" "$SM/smolvla_validate.log"; then
  setst "SmolVLA validate: FAIL (image/EGL). Chain STOPPED. See smolvla_validate.log"
  say  "Stage A validate: FAIL -> stop"
  exit 1
fi
setst "SmolVLA validate: PASS (deps import + EGL headless render OK)"
say  "Stage A validate: PASS"

# ---- Stage B: 1 cheap GPU smoke cell (2 episodes) ----
say "Stage B: launching GPU smoke (1 cell)"
modal run smolvla_modal.py::main --stage smoke > "$SM/smoke.log" 2>&1
if ! grep -qE "'sr':" "$SM/smoke.log" || ! grep -qE "'rc': 0" "$SM/smoke.log"; then
  setst "SmolVLA GPU smoke: FAIL (no SR returned). Chain STOPPED. See smoke.log"
  say  "Stage B smoke: FAIL -> stop"
  exit 1
fi
setst "SmolVLA GPU smoke: PASS ($(grep -oE "'sr': [0-9.]+" "$SM/smoke.log" | head -1))"
say  "Stage B smoke: PASS"

# ---- Stage C: full 24-cell parallel robustness sweep ----
say "Stage C: launching stage1 (24 cells, parallel .map) — MAX UTILISATION"
setst "SmolVLA stage1: LAUNCHED (24 cells fanning out)"
modal run smolvla_modal.py::main --stage stage1 --episodes 20 > "$SM/stage1.log" 2>&1
RC=$?
NRES=$(grep -cE "SR=" "$SM/stage1.log" 2>/dev/null || echo 0)
if [ "$RC" -eq 0 ] && [ "$NRES" -ge 1 ]; then
  setst "SmolVLA stage1: DONE ($NRES/24 cells). Table in stage1.log; npz on volume."
  say  "Stage C stage1: DONE ($NRES/24)"
else
  setst "SmolVLA stage1: ENDED rc=$RC, $NRES/24 cells. See stage1.log"
  say  "Stage C stage1: ENDED rc=$RC ($NRES/24)"
fi
setst "AUTO-PILOT COMPLETE"
say "AUTO-PILOT COMPLETE"
