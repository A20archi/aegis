#!/usr/bin/env python3
"""M7 — aggregate pi0.5 vs pi0.5+AEGIS on LIBERO-Plus (24 cells: 4 suites x 3 seeds x {base,aegis}).
Reads cell JSONs (per_cat category->SR, total_sr, n_episodes) from RES/{suite}/seed{seed}/{arm}.json.
Reports: per-suite 3-seed mean base/AEGIS + Δ; per-axis (category) base/AEGIS + Δ; net equal-suite Δ with
a paired seed-bootstrap 95% CI. Honest reframe: FIRST pi0.5-on-LIBERO-Plus baseline + AEGIS Δ, gate-closed
= pi0.5 exactly (identity-preserving, no clean tax). NO oracle, NO best-of-seeds — base is the paired ref."""
import json, os, glob, numpy as np
RES = os.environ.get("PI05_RES", "/home/user/Desktop/vla_projects/pi05_results")
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"),
          ("libero_goal", "Goal"), ("libero_10", "Long")]
SEEDS = [42, 123, 456]
rng = np.random.default_rng(0)

def load(suite, seed, arm):
    p = f"{RES}/{suite}/seed{seed}/{arm}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))

def suite_mean(suite, arm):           # 3-seed mean of total_sr (%)
    vals = [load(suite, s, arm) for s in SEEDS]
    vals = [100 * d["total_sr"] for d in vals if d]
    return (np.mean(vals), vals) if vals else (None, [])

def axis_table():
    """Aggregate per_cat (category) SR across suites+seeds, base vs AEGIS."""
    acc = {}  # cat -> {"base":[], "aegis":[]}
    for suite, _ in SUITES:
        for seed in SEEDS:
            for arm in ("base", "aegis"):
                d = load(suite, seed, arm)
                if not d:
                    continue
                for c, sr in d.get("per_cat", {}).items():
                    acc.setdefault(c, {"base": [], "aegis": []})[arm].append(100 * sr)
    return acc

def net_ci(nb=5000):
    """Paired bootstrap over the 12 (suite,seed) cells: Δ = aegis-base per cell, resample cells."""
    pairs = []
    for suite, _ in SUITES:
        for seed in SEEDS:
            b, a = load(suite, seed, "base"), load(suite, seed, "aegis")
            if b and a:
                pairs.append(100 * (a["total_sr"] - b["total_sr"]))
    if not pairs:
        return None, None, None
    pairs = np.array(pairs)
    boots = [rng.choice(pairs, size=len(pairs), replace=True).mean() for _ in range(nb)]
    return float(pairs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def main():
    print("=" * 78)
    print("pi0.5  vs  pi0.5 + AEGIS  —  LIBERO-Plus (7 perturbation axes, 3 seeds {42,123,456})")
    print("FIRST published pi0.5-on-LIBERO-Plus number + AEGIS Δ. gate-closed = pi0.5 exactly")
    print("(identity-preserving, no clean tax). No oracle, no best-of-seeds; base = paired ref.")
    print("=" * 78)
    # ---- per-suite ----
    print(f"\n{'Suite':9}{'pi0.5':>9}{'+AEGIS':>9}{'Δ':>8}   per-seed base -> aegis")
    print("-" * 78)
    nb_all, na_all = [], []
    done = 0
    for suite, disp in SUITES:
        mb, vb = suite_mean(suite, "base")
        ma, va = suite_mean(suite, "aegis")
        if mb is None or ma is None:
            print(f"{disp:9}{'(pending)':>9}")
            continue
        done += 1
        nb_all.append(mb); na_all.append(ma)
        seedstr = " ".join(f"{b:.0f}->{a:.0f}" for b, a in zip(vb, va))
        print(f"{disp:9}{mb:9.1f}{ma:9.1f}{ma-mb:+8.1f}   {seedstr}")
    if nb_all:
        print("-" * 78)
        net_b, net_a = np.mean(nb_all), np.mean(na_all)
        d, lo, hi = net_ci()
        ci = f"   [paired 95% CI on Δ: {lo:+.1f}, {hi:+.1f}]" if d is not None else ""
        print(f"{'NET':9}{net_b:9.1f}{net_a:9.1f}{net_a-net_b:+8.1f}{ci}")
        sig = "" if d is None else ("  *significant (CI excludes 0)" if (lo > 0 or hi < 0) else "  (CI includes 0)")
        verdict = "WIN" if net_a > net_b else ("TIE" if abs(net_a-net_b) < 0.5 else "REGRESSION")
        print(f"  net AEGIS Δ = {net_a-net_b:+.2f}  -> {verdict}{sig}   ({done}/4 suites complete)")
    # ---- per-axis ----
    acc = axis_table()
    if acc:
        print(f"\n{'Perturbation axis':24}{'pi0.5':>9}{'+AEGIS':>9}{'Δ':>8}")
        print("-" * 52)
        for c in sorted(acc):
            b, a = acc[c]["base"], acc[c]["aegis"]
            if b and a:
                mb, ma = np.mean(b), np.mean(a)
                print(f"{c:24}{mb:9.1f}{ma:9.1f}{ma-mb:+8.1f}")
    print()

if __name__ == "__main__":
    main()
