#!/bin/bash
# AEGIS live pipeline monitor. Watch with:   watch -n 10 bash sib_vla/monitor.sh
# (auto-detects stage: RIB training -> smoke-test -> LIBERO-V eval -> done)
cd "$(dirname "$0")" 2>/dev/null || cd .
RIBLOG=results/ib_on86/log_finetune_rib.txt
LIBVB=results/ib_on86/log_libv_baseline.txt
LIBVA=results/ib_on86/log_libv_aegis.txt

line() { printf '─%.0s' {1..62}; echo; }
echo "================================================================"
echo "  AEGIS PIPELINE MONITOR        $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ---- detect stage ----
STAGE="idle / done"
pgrep -f finetune_rib.py >/dev/null && STAGE="RIB TRAINING"
pgrep -f "eval_libero_v.py" >/dev/null && STAGE="LIBERO-V EVAL"
echo "  STAGE: $STAGE"
echo

# ---- RIB training panel ----
if [ -f "$RIBLOG" ]; then
  line; echo "  RIB TRAINING (perception leg)"
  python3 - "$RIBLOG" <<'PY' 2>/dev/null
import re,sys
L=[l for l in open(sys.argv[1]) if l.startswith('[rib] step=')]
if L:
    g=lambda l,p:float(re.search(p,l).group(1))
    s=lambda l:int(re.search(r'step=\s*(\d+)',l).group(1))
    cur=s(L[-1]); tot=12000
    task=lambda l:g(l,r'task=([\d.]+)'); coeff=g(L[-1],r'coeff=([+\-\d.]+)'); kl=g(L[-1],r'kl=([\d.]+)')
    st=[task(l) for l in L if s(l)<500]; rc=[task(l) for l in L if s(l)>=cur-1000]
    st=sum(st)/len(st) if st else 0; rc=sum(rc)/len(rc) if rc else 0
    d=rc-st; verdict='LEARNING' if d<-0.003 else 'FLAT' if abs(d)<=0.003 else 'RISING!'
    eta=(tot-cur)/70.0
    bar=int(28*cur/tot);
    print(f"    [{'#'*bar}{'.'*(28-bar)}] {cur}/{tot} ({100*cur//tot}%)  ETA ~{eta:.0f} min")
    print(f"    task-loss: start {st:.4f} -> recent {rc:.4f}   delta {d:+.4f}  [{verdict}]")
    print(f"    coeff {coeff:+.3f}   latent-L2 {kl:.3f}")
    print(f"    last: {L[-1].strip()[:70]}")
else:
    print("    (warming up: loading model + dataset...)")
PY
  echo
fi

# ---- LIBERO-V eval panel ----
if [ -f "$LIBVB" ] || [ -f "$LIBVA" ]; then
  line; echo "  LIBERO-V ROBUSTNESS EVAL"
  for arm in baseline aegis; do
    d=results/ib_on86/libero_v/$arm
    n=$(ls $d/eval_*.json 2>/dev/null | wc -l)
    cur=$(grep -hoE "== [a-z_0-9]+: SR" results/ib_on86/log_libv_$arm.txt 2>/dev/null | tail -1)
    printf "    %-9s conditions done: %s/6   %s\n" "$arm" "$n" "$cur"
  done
  # live AEGIS-vs-baseline if both have any results
  python3 - <<'PY' 2>/dev/null
import json,glob
def load(m):
    d={}
    for f in glob.glob(f'results/ib_on86/libero_v/{m}/eval_*.json'):
        r=json.load(open(f)); d[r['condition']]=r['success_rate']*100
    return d
b,a=load('baseline'),load('aegis'); cc=sorted(set(b)&set(a))
if cc:
    print("    --- live gap (AEGIS - baseline) ---")
    for c in cc: print(f"      {c:18s} {b[c]:5.1f}% -> {a[c]:5.1f}%  {a[c]-b[c]:+5.1f}pp")
PY
  echo
fi

# ---- GPU + known results ----
line; echo "  GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
echo
echo "  clean SR (done): baseline+TE 86.0% [80.5-90.1] | RASF+TE 84.5% [78.8-88.9]"
echo "================================================================"
