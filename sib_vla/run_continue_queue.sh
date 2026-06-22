#!/bin/bash
# Continuation watcher: the in-flight RASF clean eval (rasf_on86_tempens/_pure) is
# already running orphaned after the old queue wrapper was restarted. Do NOT re-run it.
# Wait for it to finish, then chain: IB-Adapter eval -> RASF robustness sweep (headline).
cd "$(dirname "$0")"

echo "[cont] waiting for in-flight RASF clean eval to finish..."
while pgrep -f "run_rasf_on86_eval.sh" >/dev/null \
   || pgrep -f "eval.py --config configs/rasf_on86.yaml --weights results/rasf_on86/rasf_on86.pt --n-action-steps 1 --forge-ensemble" >/dev/null \
   || pgrep -f "tag rasf_on86_pure" >/dev/null; do
  sleep 30
done
sleep 5

echo "[cont] === IB-Adapter eval ==="
bash run_ib_on86_eval.sh

echo "[cont] === RASF ROBUSTNESS SWEEP (headline: retention under action noise) ==="
bash run_rasf_robustness.sh

echo "[cont] ALL EVALS DONE"
