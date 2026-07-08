#!/usr/bin/env python3
"""Rigorous 3-way LIBERO-Plus table: VLA-Adapter Pro (base, deterministic ref) vs AEGIS (3 seeds)
vs StableVLA-native (3 seeds). Per-suite RAW SR + 3-seed mean + PAIRED task-level bootstrap 95% CI
on the deltas (AEGIS-base, StableVLA-base). Equal-suite-weight overall. No oracle, no cherry-pick.
Handles partial StableVLA (reports what is complete). Bootstrap = task resampling, 5000 iters."""
import json, glob, os, numpy as np
VP="/home/user/Desktop/vla_projects"
AX=json.load(open(f"{VP}/data/axis_ranges.json"))
SEEDS=[42,123,456]; NB=5000
SUITES=[("libero_object","Object",True),("libero_goal","Goal",False),
        ("libero_10","Long",False),("libero_spatial","Spatial",False)]
rng=np.random.default_rng(0)

def jl(paths):
    m={}
    for f in paths:
        for l in open(f):
            try: r=json.loads(l)
            except: continue
            m[r["task_id"]]=(r["successes"],r["episodes"])
    return m
def base_map(suite): return jl(glob.glob(f"{VP}/results/vla_adapter_base_pro/{suite}.*.jsonl"))
def arm_obj(arm,seed): return jl(glob.glob(f"{VP}/results/obj_full/{arm}_s{seed}/*.jsonl"))
def arm_suite(arm,suite,seed): return jl(glob.glob(f"{VP}/results/suite_3seed/{arm}_{suite}_s{seed}/*.jsonl"))

def sr(m,ids):
    S=sum(m[t][0] for t in ids if t in m); E=sum(m[t][1] for t in ids if t in m)
    return 100*S/E if E else None
def arm_mean_sr(maps,ids):           # mean over available seeds
    vals=[sr(m,ids) for m in maps if m]; vals=[v for v in vals if v is not None]
    return np.mean(vals) if vals else None

def suite_data(suite):
    if suite=="libero_object":
        ids=list(range(2518))
        base=base_map(suite)
        aeg=[arm_obj("aegis",s) for s in SEEDS]; nat=[arm_obj("native",s) for s in SEEDS]
    else:
        EVEN=json.load(open(f"{VP}/data/even80_ids_{suite}.json"))
        ids=sorted({i for v in EVEN.values() for i in v})
        base=base_map(suite)
        aeg=[arm_suite("aegis",suite,s) for s in SEEDS]; nat=[arm_suite("native",suite,s) for s in SEEDS]
    ids=[t for t in ids if t in base]                          # tasks with a base result
    return ids,base,aeg,nat

def paired_boot(ids,base,arms):
    """returns dict: point deltas + 95% CI for each arm vs base, over resampled tasks."""
    ids=np.array([t for t in ids if any(t in m for m in arms if m)])
    if len(ids)==0: return None
    def arm_sr_ids(arms_,idset):
        vals=[sr(m,idset) for m in arms_ if m]; vals=[v for v in vals if v is not None]
        return np.mean(vals) if vals else None
    pt_b=sr(base,ids); pt_a=arm_sr_ids(arms,ids)
    if pt_a is None: return None
    boots=[]
    for _ in range(NB):
        samp=rng.choice(ids,size=len(ids),replace=True)
        # count-based resample (ids may repeat) -> use multiset via successes/episodes sums
        from collections import Counter; c=Counter(samp.tolist())
        Sb=Eb=0.0
        for t,k in c.items():
            if t in base: Sb+=k*base[t][0]; Eb+=k*base[t][1]
        bb=100*Sb/Eb if Eb else None
        avals=[]
        for m in arms:
            if not m: continue
            Sa=Ea=0.0
            for t,k in c.items():
                if t in m: Sa+=k*m[t][0]; Ea+=k*m[t][1]
            if Ea: avals.append(100*Sa/Ea)
        aa=np.mean(avals) if avals else None
        if bb is not None and aa is not None: boots.append(aa-bb)
    boots=np.array(boots)
    lo,hi=np.percentile(boots,[2.5,97.5])
    return dict(base=pt_b,arm=pt_a,delta=pt_a-pt_b,lo=lo,hi=hi,sig=(lo>0 or hi<0))

print("="*86)
print("VLA-Adapter Pro (base)  vs  AEGIS (3-seed)  vs  StableVLA-native (3-seed) — LIBERO-Plus")
print("RAW SR, no oracle. CI = 95% paired task-bootstrap on the delta.  * = CI excludes 0 (significant)")
print("="*86)
print(f"{'suite':9}{'base':>7}{'AEGIS':>8}{'Δ vs base [95% CI]':>26}{'StVLA':>8}{'Δ vs base [95% CI]':>26}")
agg={"AEGIS":[], "StVLA":[]}; nat_seeds_present=set()
for suite,disp,_ in SUITES:
    ids,base,aeg,nat=suite_data(suite)
    for si,m in zip(SEEDS,nat):
        if m: nat_seeds_present.add(si)
    A=paired_boot(ids,base,aeg); N=paired_boot(ids,base,nat)
    bs=A['base'] if A else sr(base,ids)
    if A:
        agg["AEGIS"].append(A['delta'])
        a_str=f"{A['arm']:6.1f}"; ad=f"{A['delta']:+4.1f} [{A['lo']:+4.1f},{A['hi']:+4.1f}]{'*' if A['sig'] else ' '}"
    else: a_str="   –  "; ad=" (pending)"
    if N and any(nat):
        agg["StVLA"].append(N['delta']); n_str=f"{N['arm']:6.1f}"
        nd=f"{N['delta']:+4.1f} [{N['lo']:+4.1f},{N['hi']:+4.1f}]{'*' if N['sig'] else ' '}"
    else: n_str="   –  "; nd=" (training)"
    print(f"{disp:9}{bs:7.1f}{a_str:>8}{ad:>26}{n_str:>8}{nd:>26}")
# overall equal-suite-weight
if agg["AEGIS"]:
    oa=np.mean(agg["AEGIS"])
    ns=f"{np.mean(agg['StVLA']):+.1f}" if len(agg['StVLA'])==4 else "(StableVLA incomplete)"
    print("-"*86)
    print(f"{'OVERALL':9}{'':7}{'':8}{'AEGIS Δ '+f'{oa:+.1f}':>26}{'':8}{'StVLA Δ '+ns:>26}  (equal-suite-weight)")
    nseeds=sorted(nat_seeds_present)
    print(f"\nAEGIS: 3 training seeds {SEEDS}.  StableVLA-native seeds present: {nseeds} "
          f"({'3-SEED COMPLETE' if len(nseeds)==3 else 'PARTIAL — s123/s456 still training'}).")
