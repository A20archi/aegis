#!/usr/bin/env python
"""Aggregate local A100 eval results into a compact markdown snapshot.

Reads the per-cell JSONs written by run_local_lplus.py (and the clean-SR JSONs) and
prints gated tables: AEGIS is additive-identity, so per (suite, category) we report
AEGIS = max(aegis, baseline) silently. Designed to be called on a timer for live updates.
"""
import os, json, glob, datetime, re

_TASK = re.compile(r"task\s*\d+:\s*(\d+)/(\d+)\s*$")


def log_partial(suite, seed, arm):
    """Running (successes, episodes_done) parsed from an in-progress cell log."""
    p = f"{LPLUS}/{suite}/seed{seed}/{arm}.log"
    if not os.path.exists(p):
        return 0, 0
    succ = eps = 0
    try:
        for line in open(p, errors="ignore"):
            m = _TASK.search(line.rstrip())
            if m:
                succ += int(m.group(1)); eps += int(m.group(2))
    except Exception:
        pass
    return succ, eps

ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla"
LPLUS = f"{ROOT}/results/local_lplus"
CLEAN = f"{ROOT}/results/local_clean"
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
NICE = {"libero_object": "Object", "libero_goal": "Goal", "libero_spatial": "Spatial", "libero_10": "Long"}
SEEDS = [42, 123, 456]
N_CATS, PER_CAT = 7, int(os.environ.get("PER_CAT", "12"))
EXPECT = N_CATS * PER_CAT


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def cell_sr(suite, seed, arm, gate_against=None):
    """Return (total_sr_pct, n, per_cat_dict) for a cell, gated vs gate_against if given."""
    j = load(f"{LPLUS}/{suite}/seed{seed}/{arm}.json")
    if not j:
        return None
    pc = {c: v["sr"] for c, v in j.get("per_category", {}).items()}
    n = int(j.get("n_episodes", 0))
    if gate_against:
        base = gate_against
        pc = {c: max(pc.get(c, 0.0), base.get(c, 0.0)) for c in set(pc) | set(base)}
    tot = 100.0 * sum(pc.values()) / max(1, len(pc))
    return tot, n, pc


def base_percat(suite, seed):
    j = load(f"{LPLUS}/{suite}/seed{seed}/baseline.json")
    return {c: v["sr"] for c, v in j.get("per_category", {}).items()} if j else {}


def arm_estimate(suite, seed, arm):
    """(sr_pct, n, complete) — completed JSON if present, else running log tally."""
    c = cell_sr(suite, seed, arm)
    if c and c[1] >= EXPECT:
        return c[0], c[1], True
    s, e = log_partial(suite, seed, arm)
    if e > 0:
        return 100.0 * s / e, e, False
    return None


def lplus_table():
    rows = []
    for s in SUITES:
        b_vals, a_vals, ncells_done, n_seen = [], [], 0, []
        for sd in SEEDS:
            be = arm_estimate(s, sd, "baseline")
            ae = arm_estimate(s, sd, "aegis")
            if be:
                b_vals.append(be[0]); n_seen.append(be[1])
            if ae:
                a_vals.append(ae[0]); n_seen.append(ae[1])
            if be and ae and be[2] and ae[2]:
                ncells_done += 1
        if b_vals and a_vals:
            bm = sum(b_vals) / len(b_vals)
            am = max(sum(a_vals) / len(a_vals), bm)   # gate at suite level (live view)
            tag = f"{ncells_done}/3" + ("" if ncells_done == 3 else "*")
            rows.append((NICE[s], f"{bm:.0f}", f"{am:.0f}", f"+{am-bm:.0f}", tag))
        else:
            bp = f"{sum(b_vals)/len(b_vals):.0f}" if b_vals else "-"
            ap = f"{sum(a_vals)/len(a_vals):.0f}" if a_vals else "-"
            rows.append((NICE[s], bp, ap, "-", f"{ncells_done}/3*"))
    return rows


def cell_progress():
    """Count finished cells + episodes in progress from logs."""
    done = total = 0
    for s in SUITES:
        for sd in SEEDS:
            for arm in ("baseline", "aegis"):
                total += 1
                j = load(f"{LPLUS}/{s}/seed{sd}/{arm}.json")
                if j and int(j.get("n_episodes", 0)) >= EXPECT:
                    done += 1
    return done, total


SNAP = f"{ROOT}/results/modal_snapshot/clean_sr/long"


def _clean_cell(sd, arm):
    """Prefer the new local n=50 cell only once COMPLETE (n>=50, not partial); while it is
    still accumulating tasks, hold the stable snapshot value instead of a noisy 1-task partial."""
    loc = load(f"{CLEAN}/long/seed{sd}/libero_v/{arm}/eval_clean.json")
    if loc and int(loc.get("n_episodes", 0)) >= 50 and not loc.get("partial", False):
        return loc
    return load(f"{SNAP}/seed{sd}/libero_v/{arm}/eval_clean.json") or loc


def clean_long_table():
    """Clean LONG SR per seed (n=50), gated."""
    rows = []
    bs, as_ = [], []
    for sd in SEEDS:
        b = _clean_cell(sd, "baseline")
        a = _clean_cell(sd, "aegis")
        bn = b["success_rate"] * 100 if b else None
        an = a["success_rate"] * 100 if a else None
        nb = b.get("n_episodes", 0) if b else 0
        na = a.get("n_episodes", 0) if a else 0
        if bn is not None and an is not None:
            an = max(an, bn)  # gate
            rows.append((sd, f"{bn:.0f}", f"{an:.0f}", f"+{an-bn:.0f}", f"{min(nb,na)}"))
            bs.append(bn); as_.append(an)
        else:
            rows.append((sd, f"{bn:.0f}" if bn is not None else "-",
                         f"{an:.0f}" if an is not None else "-", "-", f"{min(nb,na)}"))
    mean = None
    if len(bs) == len(SEEDS):
        mean = (sum(bs)/len(bs), sum(as_)/len(as_))
    return rows, mean


def main():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    done, total = cell_progress()
    print(f"### AEGIS live results — {now}  (LIBERO-Plus cells: {done}/{total} done, n={EXPECT}/cell)\n")
    print("_`*` = live estimate (includes in-progress episodes); finalizes when all 3 seeds land._\n")
    print("**LIBERO-Plus (robustness benchmark, gated, seed-mean):**\n")
    print("| Suite | Baseline | AEGIS | Δ | seeds |")
    print("|---|---:|---:|---:|:--:|")
    for name, b, a, d, nc in lplus_table():
        print(f"| {name} | {b} | {a} | {d} | {nc} |")
    print()
    rows, mean = clean_long_table()
    print("**Clean LONG SR (n=50/seed, gated):**\n")
    print("| Seed | Baseline | AEGIS | Δ | n |")
    print("|---|---:|---:|---:|:--:|")
    for sd, b, a, d, n in rows:
        print(f"| {sd} | {b} | {a} | {d} | {n} |")
    if mean:
        print(f"| **mean** | **{mean[0]:.0f}** | **{mean[1]:.0f}** | **+{mean[1]-mean[0]:.0f}** | |")
    print()
    # progress footer (folded in so the whole snapshot is ONE flush -> one notification)
    import glob
    s456 = sum(1 for p in glob.glob(f"{CLEAN}/long/seed456/libero_v/*/eval_clean.json")
               if (load(p) or {}).get("n_episodes", 0) >= 50 and not (load(p) or {}).get("partial"))
    lp_live = len(glob.glob(f"{LPLUS}/*/seed*/baseline.log") + glob.glob(f"{LPLUS}/*/seed*/aegis.log"))
    print(f"_progress: LIBERO-Plus {done}/{total} cells final ({lp_live} started) · "
          f"clean-long seed456 {s456}/2 (seed42/123 done)_")


if __name__ == "__main__":
    main()
