#!/usr/bin/env python3
"""GR00T N1.5 3-seed LIBERO-Plus CI table. Paired task-bootstrap 95% CI on Δ (AEGIS - base).
Pairs are matched (seed, task_id) units: base and aegis share the same eval --seed, so identical
task sampling => proper paired comparison. Net pools all suites. IQM over per-suite seed means.
ponytail: fixed rng seed for reproducibility; 10k resamples is plenty for 95% CI."""
import json, glob, collections
import numpy as np

OUT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla/groot_src/results/groot_canon/eval"
SUITES = ["Object", "Spatial", "Goal", "Long"]
SEEDS = ["42", "123", "456"]
NBOOT = 10000
rng = np.random.default_rng(0)

def load_tasks(seed, suite, arm):
    """flatten per_cat->per_task into {task_id: success(0/1)}"""
    hits = glob.glob(f"{OUT}/{arm}/{suite}/{suite}/*/seed{seed}/result.json")
    if not hits:
        return None
    d = json.load(open(hits[0]))
    out = {}
    for cat in d.get("per_cat", {}).values():
        for tid, t in cat.get("per_task", {}).items():
            if "success" in t:
                out[tid] = 1 if t["success"] else 0
    return out

# collect paired units per suite: list of (base, aegis) over all seeds/tasks
suite_pairs = collections.defaultdict(list)
suite_seedmean = collections.defaultdict(list)   # per suite: list of per-seed Δ (for IQM)
per_seed_suite = {}                               # (seed,suite)->(base%,aegis%,Δ)
for seed in SEEDS:
    for suite in SUITES:
        b = load_tasks(seed, suite, "base")
        a = load_tasks(seed, suite, "aegis")
        if not b or not a:
            print(f"  WARN missing {seed} {suite}: base={bool(b)} aegis={bool(a)}")
            continue
        common = sorted(set(b) & set(a))
        pairs = [(b[t], a[t]) for t in common]
        suite_pairs[suite] += pairs
        bm = np.mean([p[0] for p in pairs]) * 100
        am = np.mean([p[1] for p in pairs]) * 100
        per_seed_suite[(seed, suite)] = (bm, am, am - bm)
        suite_seedmean[suite].append(am - bm)

def boot_ci(pairs):
    pairs = np.array(pairs, float)  # (N,2) base,aegis
    d = pairs[:, 1] - pairs[:, 0]
    obs = d.mean() * 100
    N = len(d)
    idx = rng.integers(0, N, size=(NBOOT, N))
    boot = d[idx].mean(axis=1) * 100
    return obs, np.percentile(boot, 2.5), np.percentile(boot, 97.5)

print("\n=== GR00T N1.5 — LIBERO-Plus, 3 seeds {42,123,456} — paired task-bootstrap 95% CI ===\n")
print(f"{'Suite':8}{'base':>7}{'AEGIS':>7}{'Δ mean':>9}{'95% CI':>18}  {'seeds Δ':>22}")
all_pairs = []
suite_ci = {}
for suite in SUITES:
    pairs = suite_pairs[suite]
    all_pairs += pairs
    obs, lo, hi = boot_ci(pairs)
    suite_ci[suite] = (obs, lo, hi)
    # mean base/aegis across all pooled tasks
    bm = np.mean([p[0] for p in pairs]) * 100
    am = np.mean([p[1] for p in pairs]) * 100
    seedsd = "/".join(f"{per_seed_suite[(s,suite)][2]:+.1f}" for s in SEEDS if (s,suite) in per_seed_suite)
    star = " *" if (lo > 0 or hi < 0) else "  "
    print(f"{suite:8}{bm:7.1f}{am:7.1f}{obs:+9.2f}  [{lo:+6.2f},{hi:+6.2f}]{star}  {seedsd:>22}")

obs, lo, hi = boot_ci(all_pairs)
bm = np.mean([p[0] for p in all_pairs]) * 100
am = np.mean([p[1] for p in all_pairs]) * 100
star = " *" if (lo > 0 or hi < 0) else "  "
print("-"*72)
print(f"{'NET':8}{bm:7.1f}{am:7.1f}{obs:+9.2f}  [{lo:+6.2f},{hi:+6.2f}]{star}")

# per-seed net
print("\nPer-seed net Δ (mean over 4 suites):")
for seed in SEEDS:
    ds = [per_seed_suite[(seed,s)][2] for s in SUITES if (seed,s) in per_seed_suite]
    print(f"  s{seed}: {np.mean(ds):+.2f}   ({'  '.join(f'{s} {per_seed_suite[(seed,s)][2]:+.1f}' for s in SUITES if (seed,s) in per_seed_suite)})")
print("\n* = 95% CI excludes 0")
