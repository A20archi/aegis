#!/bin/bash
# LIBERO-V robustness headline, CONDITION-BY-CONDITION (baseline then AEGIS per
# condition) so the AEGIS-vs-baseline gap appears live on monitor.sh and we can
# abort early if a signal dies. Strongest-mechanism axes first (noise/blur), the
# geometric viewpoint axis (expected parity) last.
#   baseline = SmolVLA + TE        |  aegis = RIB + RASF + TE
# Args:  $1 = episodes/task (default 20 -> n=20/task, capped at eval_n_envs)
#        $2 = comma task ids (default all 10)
#        $3 = space/comma condition list (default 6 conditions / 4 axes)
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
# Base+modules overridable via env (default = current 86 base; set for v2 base):
#   CFG=configs/_rib_v2.yaml RIB=results/ib_on_v2/rib_v2.pt RASF=results/rasf_on_v2/rasf_v2.pt bash run_libero_v_headline.sh ...
CFG=${CFG:-configs/ib_on86.yaml}
RIB=${RIB:-results/ib_on86/rib_on86.pt}
RASF=${RASF:-results/rasf_on86/rasf_on86.pt}
EP=${1:-20}
TASKS=${2:-0,1,2,3,4,5,6,7,8,9}
# Order: systematic shifts first (where RIB complements TE), stochastic noise last
# (TE already handles it -> expected tie; kept as the informative control axis).
CONDS=${3:-"viewpoint_medium lighting_1 texture_1 motion_blur_1 viewpoint_large gaussian_noise_1"}
CONDS=${CONDS//,/ }
TE="--forge-ensemble --ensemble-coeff 0.01"
REC="--record --videos-per-task 2"

[ -f "$RIB" ]  || { echo "FATAL: RIB ckpt missing ($RIB)";  exit 1; }
[ -f "$RASF" ] || { echo "FATAL: RASF ckpt missing ($RASF)"; exit 1; }

BL=results/ib_on86/log_libv_baseline.txt
AG=results/ib_on86/log_libv_aegis.txt
: > "$BL"; : > "$AG"
echo "headline run: EP=$EP TASKS=$TASKS" | tee "$BL" "$AG" >/dev/null
echo "START $(date '+%F %T')  conditions: $CONDS"

for c in $CONDS; do
  echo "######## $c  —  BASELINE  $(date +%T) ########" | tee -a "$BL"
  python scripts/eval_libero_v.py --config $CFG --method baseline $TE $REC \
    --n-action-steps 1 --episodes $EP --tasks "$TASKS" --only "$c" 2>&1 | tee -a "$BL"
  echo "######## $c  —  AEGIS  $(date +%T) ########" | tee -a "$AG"
  python scripts/eval_libero_v.py --config $CFG --method aegis \
    --rib-weights $RIB --rasf-weights $RASF $TE $REC \
    --n-action-steps 1 --episodes $EP --tasks "$TASKS" --only "$c" 2>&1 | tee -a "$AG"
  python3 -c "
import json
def g(m):
    try:
        r=json.load(open(f'results/ib_on86/libero_v/{m}/eval_${c}.json'))
        return r['success_rate']*100, r['n_episodes']
    except Exception: return None,0
b,nb=g('baseline'); a,na=g('aegis')
if b is not None and a is not None:
    print(f'>>> LIVE GAP ${c}: baseline {b:.1f}% -> AEGIS {a:.1f}%  =  {a-b:+.1f}pp   (n={nb}/{na})')
"
done

echo "######## FINAL RETENTION TABLE  $(date +%T) ########"
python3 -c "
import json,glob
def load(m):
    d={}
    for f in glob.glob(f'results/ib_on86/libero_v/{m}/eval_*.json'):
        r=json.load(open(f)); d[r['condition']]=(r['success_rate']*100, r['success_wilson95'], r['n_episodes'])
    return d
b=load('baseline'); a=load('aegis')
order=['gaussian_noise_1','motion_blur_1','lighting_1','texture_1','viewpoint_medium','viewpoint_large']
conds=[c for c in order if c in b and c in a]
print(f'{\"condition\":18s}{\"baseline+TE\":>20}{\"AEGIS\":>20}{\"gap\":>9}')
for c in conds:
    bs,bci,n=b[c]; as_,aci,_=a[c]
    print(f'{c:18s}  {bs:5.1f}% [{bci[0]*100:3.0f},{bci[1]*100:3.0f}]   {as_:5.1f}% [{aci[0]*100:3.0f},{aci[1]*100:3.0f}]  {as_-bs:+6.1f}pp  (n={n})')
app=[c for c in conds if not c.startswith('viewpoint')]
if app:
    bm=sum(b[c][0] for c in app)/len(app); am=sum(a[c][0] for c in app)/len(app)
    print(f'{\"APPEARANCE-AVG\":18s}  {bm:5.1f}%            {am:5.1f}%           {am-bm:+6.1f}pp   <-- headline')
allc=conds
bm=sum(b[c][0] for c in allc)/len(allc); am=sum(a[c][0] for c in allc)/len(allc)
print(f'{\"ALL-AXES-AVG\":18s}  {bm:5.1f}%            {am:5.1f}%           {am-bm:+6.1f}pp')
"
echo "######## DONE  $(date '+%F %T') ########"
