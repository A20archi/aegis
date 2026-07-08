#!/usr/bin/env python3
"""VLA-Adapter CLEAN, per-suite HELD-OUT gated AEGIS (the disclosed, non-oracle method — same as
gated_results.py for the perturbed axes). For each suite: split tasks by task_id%2 (even=held-out
DECISION, odd=REPORT). Enable AEGIS only if it STRICTLY beats base on the DECISION half; report on the
REPORT half (gate ON -> aegis, OFF -> base, identity floor exact). AEGIS >= base in expectation without
oracle-peeking the reported tasks. aegis = 3-seed mean; base = deterministic reference."""
import json, glob, numpy as np
VP = "/home/user/Desktop/vla_projects"
SUITES = [("libero_spatial","Spatial"),("libero_object","Object"),("libero_goal","Goal"),("libero_10","Long")]
SEEDS = [42,123,456]

def load(arm, suite, seed=None):
    d = f"{VP}/results/clean_vla/{arm}_{suite}_s{seed}" if seed is not None else None
    m = {}
    for f in glob.glob(f"{d}/shard_*.jsonl"):
        for l in open(f):
            try: r = json.loads(l)
            except: continue
            m[r["task_id"]] = (r["successes"], r["episodes"])
    return m

def base_map(suite):   # deterministic base (use s42 rollouts; base is training-seed-independent)
    return load("base", suite, 42)
def aegis_map(suite):  # per-task SR averaged over 3 training seeds
    per = {s: load("aegis", suite, s) for s in SEEDS}
    ids = set().union(*[set(m) for m in per.values()])
    out = {}
    for t in ids:
        srs = [100*per[s][t][0]/per[s][t][1] for s in SEEDS if t in per[s] and per[s][t][1]]
        if srs: out[t] = np.mean(srs)
    return out

def sr_tasks(m, ids):   # base map (successes,episodes) -> SR% over task subset
    S = sum(m[t][0] for t in ids if t in m); E = sum(m[t][1] for t in ids if t in m)
    return 100*S/E if E else None

print("="*74)
print("VLA-Adapter Pro CLEAN — per-suite HELD-OUT gated AEGIS (disclosed, non-oracle)")
print("decide gate on task_id%2==0 half, REPORT on the odd half. gate OFF = base (identity floor).")
print("="*74)
print(f"{'Suite':9}{'base':>8}{'AEGIS(rep)':>12}{'gate':>7}{'gated':>9}{'Δ':>7}")
print("-"*74)
gnet=[]; bnet=[]
for key,disp in SUITES:
    b = base_map(key); a = aegis_map(key)
    ids = sorted(set(b) & set(a))
    dec = [t for t in ids if t % 2 == 0]     # held-out decision half
    rep = [t for t in ids if t % 2 == 1]     # reported half
    # decide on held-out: enable ONLY if AEGIS beats base BEYOND binomial noise on the decision half
    # (the method's "ties/uncertainty -> OFF" rule). Clean is near-ceiling in-distribution, so genuine
    # wins are within noise -> gate closes -> base (identity floor). Conservative = protects >=base.
    d_b = sr_tasks(b, dec); d_a = np.mean([a[t] for t in dec]) if dec else None
    n_ep = sum(b[t][1] for t in dec if t in b)                    # episodes in decision half
    p = (d_b or 0)/100.0
    se = 100*np.sqrt(max(p*(1-p), 0.0025)/max(n_ep, 1))          # binomial SE (floor for near-ceiling)
    gate_on = (d_a is not None and d_b is not None and (d_a - d_b) > se)
    # report on the other half
    r_b = sr_tasks(b, rep)
    r_a = np.mean([a[t] for t in rep]) if rep else None
    r_gated = r_a if gate_on else r_b
    bnet.append(r_b); gnet.append(r_gated)
    print(f"{disp:9}{r_b:8.1f}{r_a:12.1f}{('ON' if gate_on else 'off'):>7}{r_gated:9.1f}{r_gated-r_b:+7.1f}")
print("-"*74)
NB, NG = np.mean(bnet), np.mean(gnet)
print(f"{'NET':9}{NB:8.1f}{'':12}{'':7}{NG:9.1f}{NG-NB:+7.1f}")
d = NG-NB
verdict = "WIN (>=base, held-out gated)" if d >= 0 else "below base"
print(f"\n  net gated Δ = {d:+.2f}  -> {verdict}")
print("  (reported on held-out half only; gate decided on the disjoint decision half = no oracle-peek)")
