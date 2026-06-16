#!/bin/bash
# ACTION-NOISE ROBUSTNESS — 4-arm comparison (the honest attribution).
#   bp = base  pure n=1           (naked floor)
#   rp = RASF  pure n=1           (within-chunk spectral denoise only)
#   bt = base  + temporal ensemble(cross-chunk averaging only)
#   rt = RASF  + temporal ensemble(BOTH = deployed action-locus defense)
# Key cell: rt vs bt  -> does RASF add robustness BEYOND ensembling?
# sigma=0 is already known (base-pure 79.0 / RASF-pure 78.5 / base+TE 86.0 / RASF+TE 84.5).
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
CFG=configs/rasf_on86.yaml
W=results/rasf_on86/rasf_on86.pt
OUT=results/rasf_on86
TE="--forge-ensemble --ensemble-coeff 0.01 --lever3-clamp 1.0"
LEVELS="0.15 0.30"

run () { # tag, extra-args, noise
  echo "==== $1  (noise=$3) ===="
  python scripts/eval.py --config $CFG --n-action-steps 1 --action-noise $3 $2 \
    --tag $1 2>&1 | tee $OUT/log_$1.txt
}

echo "######## ACTION-NOISE 4-ARM SWEEP ########"
for nz in $LEVELS; do
  t=$(echo "$nz" | tr -d '.')
  run rob2_bp_n${t} ""             $nz      # base pure
  run rob2_rp_n${t} "--weights $W" $nz      # RASF pure
  run rob2_bt_n${t} "$TE"          $nz      # base + TE
  run rob2_rt_n${t} "--weights $W $TE" $nz  # RASF + TE
done

echo "==== ROBUSTNESS SUMMARY (with sigma=0 from clean evals) ===="
python3 -c "
import json, glob, re
known={0.0:{'bp':79.0,'rp':78.5,'bt':86.0,'rt':84.5}}
rows={(a,0.0):v for a,v in known[0.0].items()}
for f in glob.glob('$OUT/eval_rob2_*.json'):
    r=json.load(open(f)); m=re.match(r'rob2_(bp|rp|bt|rt)_n(\d+)', r['name'])
    if m: rows[(m.group(1), int(m.group(2))/100.0)]=r['success_rate']*100
levels=sorted({k[1] for k in rows})
print(f'{\"sigma\":>6} | {\"base-pure\":>9} {\"RASF-pure\":>9} | {\"base+TE\":>8} {\"RASF+TE\":>8} | {\"RASF gain/TE\":>12}')
for nz in levels:
    g=lambda a: rows.get((a,nz))
    bp,rp,bt,rt=g('bp'),g('rp'),g('bt'),g('rt')
    d = f'{rt-bt:+.1f}pp' if (rt is not None and bt is not None) else '?'
    f4=lambda x: f'{x:8.1f}' if x is not None else '     ?  '
    print(f'{nz:6.2f} | {f4(bp)} {f4(rp)} | {f4(bt)} {f4(rt)} | {d:>12}')
print()
print('Headline cell = RASF+TE minus base+TE  (RASF marginal robustness over ensembling).')
"
echo "######## DONE ########"
