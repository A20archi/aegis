#!/bin/bash
# Wait for the IB finetune to finish, then run evals in priority order:
#   1) RASF eval (the redesign — user's active request)
#   2) IB-Adapter eval
cd "$(dirname "$0")"

echo "[queue] waiting for IB finetune to finish..."
while pgrep -f "finetune_ib.py --config configs/ib_on86" >/dev/null; do sleep 30; done
sleep 5

echo "[queue] === RASF eval (priority) ==="
bash run_rasf_on86_eval.sh

echo "[queue] === IB-Adapter eval ==="
bash run_ib_on86_eval.sh

echo "[queue] === RASF ROBUSTNESS SWEEP (headline: retention under action noise) ==="
bash run_rasf_robustness.sh

echo "[queue] ALL EVALS DONE"
