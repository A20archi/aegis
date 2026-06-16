#!/bin/bash
# ============================================================================
# track_predict.sh — combined PROCESS TRACKER + grounded PREDICTOR for the
# 4-suite eval. Every 30 min (or on a new suite completion, or all-8-done) it
# writes results/allsuites/PREDICT.md and EXITS so the agent is re-invoked to
# relay the update. Re-arm: DONE0=<current_done_count> nohup bash track_predict.sh &
#
# Predictions are NON-HALLUCINATED: suite SR is the running mean over COMPLETED
# tasks only (parsed from "[libero_v] <arm> tN: k/20" final per-task lines, or
# the authoritative eval_clean.json once written), reported with a Wilson 95% CI
# that shrinks as tasks land. Suites with 0 completed tasks => "no data yet".
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
DONE0=${DONE0:-0}
HEARTBEAT=${HEARTBEAT:-1800}      # 30 min
OUT=results/allsuites/PREDICT.md
t0=$(date +%s)

write_predict(){
python3 - > "$OUT" 2>/dev/null <<'PY'
import json,os,re,glob,time
order=['long','spatial','object','goal']
PAPER={'long':71.0,'spatial':90.0,'object':96.0,'goal':92.0}; PAVG=87.3
eplen={'long':520,'spatial':220,'object':280,'goal':300}
NTASK=10
start=int(open('results/allsuites/.start_epoch').read()) if os.path.exists('results/allsuites/.start_epoch') else int(time.time())
elapsed=max(1,int(time.time())-start)

def wilson(s,n,z=1.96):
    if n==0: return (0.0,0.0)
    p=s/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; m=(z*((p*(1-p)/n + z*z/(4*n*n))**0.5))/d
    return (max(0.0,c-m)*100, min(1.0,c+m)*100)

# per-task difficulty priors (denoised shape from prior FULL runs); object/goal absent on purpose
PRI=json.load(open('results/allsuites/difficulty_priors.json')) if os.path.exists('results/allsuites/difficulty_priors.json') else {}
SUITEKEY={'long':'libero_10','spatial':'libero_spatial','object':'libero_object','goal':'libero_goal'}

def parse(name,arm):
    """return dict: per-task {tid:(s,t)}, plus final flag. Prefer json (authoritative, has per_task)."""
    jp=f'results/allsuites/{name}/libero_v/{arm}/eval_clean.json'
    if os.path.exists(jp):
        d=json.load(open(jp)); pt=d.get('per_task',[])
        tasks={t['task_id']:(round(t['success_rate']*t.get('n',20)),t.get('n',20)) for t in pt}
        if not tasks:  # no per_task -> fall back to aggregate as one bucket
            n=d.get('n_episodes',0); tasks={'all':(d.get('n_success',round(d.get('success_rate',0)*n)),n)}
        return tasks,True
    lp=f'results/allsuites/{name}/{arm}.log'
    if not os.path.exists(lp) or os.path.getmtime(lp) < start: return {},False  # missing/STALE -> ignore
    last={}
    for ln in open(lp,errors='ignore'):
        m=re.search(r'\[libero_v\]\s+\S+\s+t(\d+):\s+(\d+)/(\d+)',ln)
        if m: last[int(m.group(1))]=(int(m.group(2)),int(m.group(3)))
    return last,False

def adjust(name,tasks):
    """calibration-anchored difficulty-adjusted FINAL-suite projection.
    proj = (observed successes on done tasks + cal*prior on REMAINING tasks)/10.
    cal = (our mean on done tasks)/(prior mean on those same tasks). No prior -> None."""
    key=SUITEKEY[name]; pri=PRI.get(key,{}).get('shape')
    done={t:v for t,v in tasks.items() if isinstance(t,int)}
    if not pri or not done: return None,None
    C=sorted(done); obs={t:100*done[t][0]/done[t][1] for t in C}
    pm=sum(pri[t] for t in C)/len(C)
    cal=(sum(obs.values())/len(C))/pm if pm>0 else 1.0
    cal=min(2.0,max(0.3,cal))                      # clip wild early-cal swings
    R=[t for t in range(NTASK) if t not in C]
    rem=[min(100.0,max(0.0,cal*pri[t])) for t in R]
    proj=(sum(obs[t] for t in C)+sum(rem))/NTASK
    rem_avg=sum(rem)/len(rem) if R else 0
    flat=sum(obs.values())/len(C)
    return proj,(cal,sorted(R),round(rem_avg,1),round(flat,1))

print('# 4-SUITE PREDICTOR — grounded + difficulty-adjusted')
print(f'\n_updated {time.strftime("%Y-%m-%d %H:%M:%S")} · elapsed {elapsed//60}m · n=200/suite (20 trials/task)_\n')
print('**Observed** = Wilson-95 over done tasks. **Proj** = observed + (calibration×historical difficulty) on *remaining* tasks. Paper avg **87.3**.\n')

rows=[]; total_done=0; total_rem=0
for name in order:
    r={'name':name}
    for arm in ['baseline','aegis']:
        tasks,fin=parse(name,arm)
        k=len([t for t in tasks if isinstance(t,int)]) or (NTASK if fin else 0)
        r[arm]=(tasks,fin,k); total_done+=k; total_rem+=(NTASK-k)
    rows.append(r)

print('| suite | paper | base+TE obs → proj | AEGIS obs → proj | Δproj | done |')
print('|---|---|---|---|---|---|')
projavg={'baseline':[], 'aegis':[]}
notes=[]
for r in rows:
    name=r['name']; cells={}
    for arm in ['baseline','aegis']:
        tasks,fin,k=r[arm]
        s=sum(v[0] for v in tasks.values()); n=sum(v[1] for v in tasks.values())
        if n==0: cells[arm]='— no data'; continue
        sr=100*s/n; lo,hi=wilson(s,n)
        if fin:
            cells[arm]=f'**{sr:.1f}** [{lo:.0f}–{hi:.0f}] FINAL'; projavg[arm].append((name,sr,True))
        else:
            pj,info=adjust(name,tasks)
            if pj is None:
                cells[arm]=f'{sr:.1f} [{lo:.0f}–{hi:.0f}] → *flat (no prior)*'; projavg[arm].append((name,sr,False))
            else:
                cal,R,remavg,flat=info
                cells[arm]=f'{sr:.1f} → **{pj:.1f}**'; projavg[arm].append((name,pj,False))
                if arm=='aegis':
                    notes.append(f"- **{name}** ({k}/10): cal×{cal:.2f}; remaining {R} avg-difficulty≈{remavg} → adj **{flat:.1f}→{pj:.1f}**")
    # delta on PROJECTED values
    def pv(arm):
        tasks,fin,k=r[arm]; s=sum(v[0] for v in tasks.values()); n=sum(v[1] for v in tasks.values())
        if n==0: return None
        if fin: return 100*s/n
        pj,_=adjust(name,tasks); return pj if pj is not None else 100*s/n
    pb,pa=pv('baseline'),pv('aegis')
    dl=f'{pa-pb:+.1f}' if (pb is not None and pa is not None) else '—'
    print(f"| {name}{' 🔑' if name=='long' else ''} | {PAPER[name]:.0f} | {cells['baseline']} | {cells['aegis']} | {dl} | b{r['baseline'][2]}/a{r['aegis'][2]} |")

def avg(lst): return sum(x[1] for x in lst)/len(lst) if lst else float('nan')
for arm,lbl in [('baseline','base+TE'),('aegis','AEGIS')]:
    lst=projavg[arm]; ncov=len(lst); nfin=sum(1 for x in lst if x[2])
    if lst:
        a=avg(lst)
        print(f'\n**{lbl} projected avg = {a:.1f}** over {ncov}/4 suites ({nfin} final, {ncov-nfin} proj) · vs 87.3 → **{a-PAVG:+.1f}**')
    else:
        print(f'\n**{lbl}**: no suite data yet.')
if notes:
    print('\n_difficulty-adjustment trace (AEGIS arm):_'); print('\n'.join(notes))

# grounded ETA from empirical task throughput across all parallel arms
rate=total_done/elapsed if total_done>0 else 0  # tasks/sec
if rate>0:
    eta_min=int((total_rem/rate)/60)
    print(f'\n**ETA (empirical, refines each cycle):** {total_done} tasks done / {total_rem} left · '
          f'~{eta_min} min (~{eta_min/60:.1f} h) to full 8-arm completion.')
else:
    print('\n**ETA:** warming up (no completed tasks yet).')
print(f'\n_done-arms: {sum(1 for f in glob.glob("results/allsuites/*/libero_v/*/eval_clean.json"))}/8_')
PY
}

while true; do
  write_predict
  done=$(ls results/allsuites/*/libero_v/*/eval_clean.json 2>/dev/null | wc -l)
  # ground-truth: are any eval procs still alive?
  alive=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
            tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q eval_libero_v && echo 1; done | wc -l)
  if [ "$done" -ge 8 ]; then echo "PREDICT: ALL 8 DONE at $(date +%T)"; cat "$OUT"; break; fi
  if [ "$done" -gt "$DONE0" ]; then echo "PREDICT: new suite-arm completion -> $done/8 at $(date +%T)"; cat "$OUT"; break; fi
  if [ "$alive" -eq 0 ] && [ "$done" -lt 8 ]; then echo "PREDICT: WARNING no eval procs alive but only $done/8 done (stall?) at $(date +%T)"; cat "$OUT"; break; fi
  if [ $(( $(date +%s)-t0 )) -ge $HEARTBEAT ]; then echo "PREDICT 30-min heartbeat ($done/8) at $(date +%T)"; cat "$OUT"; break; fi
  sleep 60
done
