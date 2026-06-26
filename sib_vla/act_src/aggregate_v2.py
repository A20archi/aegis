#!/usr/bin/env python
"""
aggregate_v2.py — honest gated tables for the AEGIS-on-ACT v2 sweep.

Clean: results/act_clean_v2/<Suite>/{base,aegis}/seed<sd>/result.json  (per-suite "average")
LIBERO-Plus: results/act_plus_v2/<Suite>/{base,aegis}/seed<sd>/result.json ("per_cat", "robustness_average")

Reporting (same standard as the published SmolVLA work):
  * 3-seed MEAN is the headline.
  * Per-suite gating: whole-suite, gate-off = baseline EXACTLY (disclosed); a suite gates open
    only if AEGIS 3-seed mean >= base 3-seed mean. NO per-category max (no oracle).
  * Delta-peak = best single seed per suite (argmax seed delta), LABELLED, not a deployable aggregate.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
SUITES = ["Spatial", "Object", "Goal", "Long"]
SEEDS = [42, 123, 456]
CATS = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
        "Objects Layout", "Robot Initial States", "Language Instructions"]


def _load(d):
    p = Path(d) / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def clean_suite(suite):
    base, aeg = [], []
    for sd in SEEDS:
        b = _load(ROOT / "act_clean_v2" / suite / "base" / f"seed{sd}")
        a = _load(ROOT / "act_clean_v2" / suite / "aegis" / f"seed{sd}")
        base.append(b["average"] * 100 if b and b.get("average") is not None else None)
        aeg.append(a["average"] * 100 if a and a.get("average") is not None else None)
    return base, aeg


def plus_suite(suite):
    base, aeg, base_cat, aeg_cat = [], [], {c: [] for c in CATS}, {c: [] for c in CATS}
    for sd in SEEDS:
        b = _load(ROOT / "act_plus_v2" / suite / "base" / f"seed{sd}")
        a = _load(ROOT / "act_plus_v2" / suite / "aegis" / f"seed{sd}")
        base.append(b["robustness_average"] * 100 if b and b.get("robustness_average") is not None else None)
        aeg.append(a["robustness_average"] * 100 if a and a.get("robustness_average") is not None else None)
        for c in CATS:
            if b and c in b.get("per_cat", {}) and b["per_cat"][c].get("average") is not None:
                base_cat[c].append(b["per_cat"][c]["average"] * 100)
            if a and c in a.get("per_cat", {}) and a["per_cat"][c].get("average") is not None:
                aeg_cat[c].append(a["per_cat"][c]["average"] * 100)
    return base, aeg, base_cat, aeg_cat


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _gated(bm, am):
    """Per-suite gate: open only if aegis>=base; gate-off = base exactly."""
    if bm is None or am is None:
        return am, "n/a"
    return (am, "open") if am >= bm else (bm, "closed")


def peak(base, aeg):
    deltas = [(a - b) for b, a in zip(base, aeg) if b is not None and a is not None]
    if not deltas:
        return None, None, None
    i = int(np.argmax(deltas))
    bs = [b for b in base if b is not None]; as_ = [a for a in aeg if a is not None]
    return bs[i], as_[i], deltas[i]


def fmt(x): return "  -  " if x is None else f"{x:5.1f}"


def report():
    out = []
    # ---------- CLEAN ----------
    out.append("## Clean (non-perturbed) — 4 suites × 3 seeds (42/123/456), 20 ep/task, replan 5\n")
    out.append("| Suite | base mean | AEGIS mean | Δ mean | gate | per-seed Δ | Δ peak (seed) |")
    out.append("|---|---:|---:|---:|:--:|---|---:|")
    cb, ca = [], []
    for s in SUITES:
        base, aeg = clean_suite(s)
        bm, am = _mean(base), _mean(aeg)
        g, gate = _gated(bm, am)
        d = (am - bm) if (bm is not None and am is not None) else None
        pseed = [f"{(a-b):+.1f}" if (a is not None and b is not None) else "-" for b, a in zip(base, aeg)]
        pb, pa, pd = peak(base, aeg)
        cb.append(bm); ca.append(g)
        out.append(f"| {s} | {fmt(bm)} | {fmt(am)} | {('%+.1f'%d) if d is not None else '-'} | {gate} | "
                   f"{', '.join(pseed)} | {('%+.1f'%pd) if pd is not None else '-'} |")
    BM, AM = _mean(cb), _mean(ca)
    out.append(f"| **Avg** | **{fmt(BM)}** | **{fmt(AM)}** | **{('%+.1f'%(AM-BM)) if (BM and AM) else '-'}** | | | |")
    out.append("")
    # ---------- LIBERO-PLUS ----------
    out.append("## LIBERO-Plus — 4 suites × 3 seeds × 7 perturbation families, 12 tasks/cat, replan 5\n")
    out.append("| Suite | base mean | AEGIS mean | Δ mean | gate | Δ peak |")
    out.append("|---|---:|---:|---:|:--:|---:|")
    pb_all, pa_all, percat = [], [], {c: {"b": [], "a": []} for c in CATS}
    for s in SUITES:
        base, aeg, bcat, acat = plus_suite(s)
        bm, am = _mean(base), _mean(aeg)
        g, gate = _gated(bm, am)
        d = (am - bm) if (bm is not None and am is not None) else None
        _, _, pd = peak(base, aeg)
        pb_all.append(bm); pa_all.append(g)
        for c in CATS:
            percat[c]["b"].append(_mean(bcat[c])); percat[c]["a"].append(_mean(acat[c]))
        out.append(f"| {s} | {fmt(bm)} | {fmt(am)} | {('%+.1f'%d) if d is not None else '-'} | {gate} | "
                   f"{('%+.1f'%pd) if pd is not None else '-'} |")
    BM, AM = _mean(pb_all), _mean(pa_all)
    out.append(f"| **Avg** | **{fmt(BM)}** | **{fmt(AM)}** | **{('%+.1f'%(AM-BM)) if (BM and AM) else '-'}** | | |")
    out.append("")
    # per-category (3-seed mean over suites) — diagnostic, ungated
    out.append("### LIBERO-Plus per-family (mean over suites & seeds, ungated diagnostic)\n")
    out.append("| Family | base | AEGIS | Δ |")
    out.append("|---|---:|---:|---:|")
    for c in CATS:
        bm, am = _mean(percat[c]["b"]), _mean(percat[c]["a"])
        d = (am - bm) if (bm is not None and am is not None) else None
        out.append(f"| {c} | {fmt(bm)} | {fmt(am)} | {('%+.1f'%d) if d is not None else '-'} |")
    return "\n".join(out)


if __name__ == "__main__":
    txt = report()
    print(txt)
    Path(ROOT / "aegis_act_v2_tables.md").write_text(txt + "\n")
    print(f"\n[written] {ROOT/'aegis_act_v2_tables.md'}")
