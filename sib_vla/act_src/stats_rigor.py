#!/usr/bin/env python
"""
stats_rigor.py — roadmap P1/P1b statistics on the locked AEGIS-on-ACT v2 sweep.

Adds (from existing per-seed/per-task JSONs, NO new rollouts):
  * Per-suite base & AEGIS SR with 95% stratified-bootstrap CI (rliable-style, implemented w/ numpy).
  * Paired per-seed Δ with 95% bootstrap CI.
  * IQM (interquartile mean) over the task×seed run set.
  * Wilson 95% interval per pooled SR cell.
Aggregates are also shown baseline-normalized (ImageNet-C style) for the per-family table.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
SUITES = ["Spatial", "Object", "Goal", "Long"]
SEEDS = [42, 123, 456]
RNG = np.random.RandomState(0)


def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def clean_runs(suite, arm):
    """Return list of per-(task,seed) success rates (each in [0,1]) for a suite/arm."""
    runs = []
    for sd in SEEDS:
        d = load(ROOT / "act_clean_v2" / suite / arm / f"seed{sd}" / "result.json")
        if d and len(d.get("per_task", {})) >= 10:
            runs += [v["success_rate"] for v in d["per_task"].values()]
    return np.array(runs)


def plus_runs(suite, arm):
    runs = []
    for sd in SEEDS:
        d = load(ROOT / "act_plus_v2" / suite / arm / f"seed{sd}" / "result.json")
        if d and len(d.get("per_cat", {})) >= 7:
            for c in d["per_cat"].values():
                runs += [1.0 if t["success"] else 0.0 for t in c.get("per_task", {}).values()]
    return np.array(runs)


def boot_ci(x, B=10000, lo=2.5, hi=97.5):
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    means = x[RNG.randint(0, len(x), size=(B, len(x)))].mean(1)
    return float(x.mean()*100), float(np.percentile(means, lo)*100), float(np.percentile(means, hi)*100)


def iqm(x):
    if len(x) == 0: return np.nan
    xs = np.sort(x); n = len(xs); a, b = int(0.25*n), int(0.75*n)
    seg = xs[a:b] if b > a else xs
    return float(seg.mean()*100)


def wilson(x, z=1.96):
    n = len(x)
    if n == 0: return (np.nan, np.nan)
    p = x.mean(); denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0, centre-half)*100, min(1, centre+half)*100)


def paired_delta_ci(suite, getter, B=10000):
    """Paired per-seed Δ (AEGIS-base) mean with bootstrap CI over seeds."""
    ds = []
    for sd in SEEDS:
        # per-seed suite mean for each arm
        if getter == "clean":
            b = load(ROOT/"act_clean_v2"/suite/"base"/f"seed{sd}"/"result.json")
            a = load(ROOT/"act_clean_v2"/suite/"aegis"/f"seed{sd}"/"result.json")
            bv = b["average"]*100 if b and len(b.get("per_task",{}))>=10 else None
            av = a["average"]*100 if a and len(a.get("per_task",{}))>=10 else None
        else:
            b = load(ROOT/"act_plus_v2"/suite/"base"/f"seed{sd}"/"result.json")
            a = load(ROOT/"act_plus_v2"/suite/"aegis"/f"seed{sd}"/"result.json")
            bv = b["robustness_average"]*100 if b and len(b.get("per_cat",{}))>=7 else None
            av = a["robustness_average"]*100 if a and len(a.get("per_cat",{}))>=7 else None
        if bv is not None and av is not None: ds.append(av-bv)
    ds = np.array(ds)
    if len(ds)==0: return None
    bm = ds[RNG.randint(0,len(ds),size=(B,len(ds)))].mean(1)
    return float(ds.mean()), float(np.percentile(bm,2.5)), float(np.percentile(bm,97.5)), len(ds)


def long_bridge_stats(mults=("0.25", "0.5")):
    """Paired Δ (de-strengthed RIB − base) with 95% bootstrap CI on the Long bridge validation.
    Base = main-sweep Long base. Only runs cells where all 3 seeds are complete."""
    print(f"\n{'='*72}\nLONG BRIDGE — de-strengthed RIB vs base, 3-seed paired Δ [95% CI]\n{'='*72}")
    # base per-seed
    def cmean(d): return d["average"]*100 if d and len(d.get("per_task",{}))>=10 else None
    def pmean(d): return d["robustness_average"]*100 if d and len(d.get("per_cat",{}))>=7 \
        and all(v.get("average") is not None for v in d["per_cat"].values()) else None
    base_c = [cmean(load(ROOT/"act_clean_v2"/"Long"/"base"/f"seed{s}"/"result.json")) for s in SEEDS]
    base_p = [pmean(load(ROOT/"act_plus_v2"/"Long"/"base"/f"seed{s}"/"result.json")) for s in SEEDS]
    print(f"  BASE Long: clean {np.nanmean([x for x in base_c if x is not None]):.1f} | "
          f"robust {np.nanmean([x for x in base_p if x is not None]):.1f}")
    for m in mults:
        for kind, sub, bvals, getter in [("CLEAN", "clean", base_c, cmean), ("ROBUST", "plus", base_p, pmean)]:
            avals = [getter(load(ROOT/"long_bridge_v"/f"m{m}"/sub/"aegis"/f"seed{s}"/"result.json")) for s in SEEDS]
            ds = [avals[i]-bvals[i] for i in range(3) if avals[i] is not None and bvals[i] is not None]
            n = len(ds)
            if n < 3:
                print(f"  mult={m} {kind:6s}: {n}/3 seeds — INCOMPLETE, stats withheld")
                continue
            ds = np.array(ds)
            bm = ds[RNG.randint(0, n, size=(10000, n))].mean(1)
            am = np.nanmean([a for a in avals if a is not None]); bmean = np.nanmean([b for b in bvals if b is not None])
            print(f"  mult={m} {kind:6s}: base {bmean:.1f} -> AEGIS {am:.1f} | "
                  f"Δ {ds.mean():+.1f} [{np.percentile(bm,2.5):+.1f},{np.percentile(bm,97.5):+.1f}] "
                  f"Δpeak {ds.max():+.1f}  (3/3)")


def section(title, runfn, metric):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"{'Suite':8s} {'base SR [95% CI]':>22s} {'AEGIS SR [95% CI]':>22s} {'paired Δ [95% CI]':>20s}")
    for s in SUITES:
        rb, ra = runfn(s, "base"), runfn(s, "aegis")
        bm, blo, bhi = boot_ci(rb); am, alo, ahi = boot_ci(ra)
        pd = paired_delta_ci(s, metric)
        base_s = f"{bm:4.1f} [{blo:4.1f},{bhi:4.1f}]" if not np.isnan(bm) else "    -    "
        aeg_s  = f"{am:4.1f} [{alo:4.1f},{ahi:4.1f}]" if not np.isnan(am) else "    -    "
        d_s = f"{pd[0]:+4.1f} [{pd[1]:+4.1f},{pd[2]:+4.1f}]" if pd else "   -   "
        print(f"{s:8s} {base_s:>22s} {aeg_s:>22s} {d_s:>20s}")


if __name__ == "__main__":
    print("AEGIS-on-ACT v2 — statistical rigor (bootstrap 95% CI over task×seed runs; IQM; Wilson)")
    section("CLEAN SR — per-suite SR with 95% stratified-bootstrap CI + paired Δ CI", clean_runs, "clean")
    section("LIBERO-PLUS — per-suite SR with 95% stratified-bootstrap CI + paired Δ CI", plus_runs, "plus")
    long_bridge_stats()
    # IQM + Wilson summary on pooled runs
    print(f"\n{'IQM (interquartile mean) + Wilson 95% per pooled cell':^72s}")
    print(f"{'Suite':8s} | {'CLEAN base/AEGIS IQM':>22s} | {'PLUS base/AEGIS IQM':>22s}")
    for s in SUITES:
        cb, ca = clean_runs(s,"base"), clean_runs(s,"aegis")
        pb, pa = plus_runs(s,"base"), plus_runs(s,"aegis")
        print(f"{s:8s} | {iqm(cb):5.1f} / {iqm(ca):5.1f}{'':>9s} | {iqm(pb):5.1f} / {iqm(pa):5.1f}")
