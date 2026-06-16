#!/bin/bash
# Gaussian-noise GRACEFUL-DEGRADATION sweep: baseline+TE vs AEGIS across
# std {0.05, 0.12, 0.20, 0.30}. Tests the hypothesis that TE handles low-sigma
# stochastic noise (tie) but AEGIS's RIB pulls ahead as sigma grows (~5pp target).
# Base/modules via env (default = current 86 base; set CFG/RIB/RASF for v2 base).
#   bash run_noise_sweep.sh [episodes] [task_ids]
set -uo pipefail
cd "$(dirname "$0")"
EP=${1:-20}
TASKS=${2:-0,1,2,3,4,5,6,7,8,9}
CONDS="gaussian_noise_0 gaussian_noise_1 gaussian_noise_2 gaussian_noise_3 gaussian_noise_4 gaussian_noise_5 gaussian_noise_6 gaussian_noise_7"

CFG=${CFG:-configs/ib_on86.yaml} RIB=${RIB:-} RASF=${RASF:-} \
  bash run_libero_v_headline.sh "$EP" "$TASKS" "$CONDS"

echo "######## NOISE DEGRADATION CURVE (std 0.05 -> 0.30) ########"
python3 - "$CFG" <<'PY'
import json, sys
cfg=sys.argv[1]; out='results/ib_on86'
try:
    import yaml; out=yaml.safe_load(open(cfg)).get('output_dir', out)
except Exception: pass
std={f'gaussian_noise_{i}':v for i,v in enumerate([0.05,0.12,0.20,0.30,0.50,0.70,0.75,1.00])}
print(f'{"std":>6} {"baseline+TE":>13} {"AEGIS":>11} {"gap":>9}')
for c in [f'gaussian_noise_{i}' for i in range(8)]:
    try:
        b=json.load(open(f'{out}/libero_v/baseline/eval_{c}.json'))['success_rate']*100
        a=json.load(open(f'{out}/libero_v/aegis/eval_{c}.json'))['success_rate']*100
        print(f'{std[c]:>6.2f} {b:>12.1f}% {a:>10.1f}% {a-b:>+8.1f}pp')
    except Exception as e:
        print(f'{std[c]:>6}: (missing) {e}')
print("\nstory holds if gap is ~0 at low std and grows toward +5pp as std rises")
PY
