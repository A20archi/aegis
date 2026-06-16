"""Aggregate eval JSONs into result tables and evaluate the Stage 3 gate.

    python scripts/aggregate.py --results results --markdown

Produces ``results/summary.csv`` and (with ``--markdown``) ``results/summary.md``.
The Stage 3 gate asks whether ``sib`` beats ``raw_vib`` on at least one of
{corruption robustness, jerk at matched success}; the verdict (and a
two-proportion z-test on the cleanest matched corruption condition) is written
into the summary so the README can state it honestly.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from sib.metrics import two_proportion_ztest


def load_rows(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(Path(results_dir) / "eval_*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"no eval_*.json found in {results_dir}")
    df = pd.DataFrame(rows)
    df["corruption"] = df["corruption"].fillna("clean")
    return df


def stage3_gate(df: pd.DataFrame) -> dict:
    """Compare sib vs raw_vib on clean jerk and on corruption robustness."""
    clean = df[df["corruption"] == "clean"]
    out = {"verdict": "indeterminate", "details": {}}

    def pick(name):
        sub = clean[clean["name"].str.startswith(name)]
        return sub.sort_values("success_rate").iloc[-1] if len(sub) else None

    sib, raw = pick("sib"), pick("raw_vib")
    if sib is None or raw is None:
        out["details"]["note"] = "need both sib and raw_vib eval rows"
        return out

    jerk_better = sib["rms_jerk_mean"] < raw["rms_jerk_mean"]
    out["details"]["clean_jerk_sib"] = float(sib["rms_jerk_mean"])
    out["details"]["clean_jerk_raw_vib"] = float(raw["rms_jerk_mean"])

    # Corruption robustness: average success under all corrupted conditions.
    def corr_mean(name):
        sub = df[(df["name"].str.startswith(name)) & (df["corruption"] != "clean")]
        return sub["success_rate"].mean() if len(sub) else float("nan")

    sib_c, raw_c = corr_mean("sib"), corr_mean("raw_vib")
    robust_better = (sib_c > raw_c) if pd.notna(sib_c) and pd.notna(raw_c) else False
    out["details"]["corruption_success_sib"] = sib_c
    out["details"]["corruption_success_raw_vib"] = raw_c

    # z-test on aggregate corrupted episodes (pooled), if counts available.
    def pooled(name):
        sub = df[(df["name"].str.startswith(name)) & (df["corruption"] != "clean")]
        return int(sub["n_success"].sum()), int(sub["n_episodes"].sum())
    s1, n1 = pooled("sib"); s2, n2 = pooled("raw_vib")
    z, pval = two_proportion_ztest(s1, n1, s2, n2)
    out["details"]["corruption_ztest"] = {"z": z, "p_value": pval}

    if jerk_better or robust_better:
        out["verdict"] = "spectral basis helps (sib beats raw_vib on jerk or robustness)"
    else:
        out["verdict"] = ("HONEST NEGATIVE: spectral basis is not the active "
                          "ingredient (raw_vib matches sib)")
    out["details"]["jerk_better"] = bool(jerk_better)
    out["details"]["robust_better"] = bool(robust_better)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    df = load_rows(args.results)
    cols = ["name", "corruption", "success_rate", "success_wilson95",
            "rms_jerk_mean", "hf_energy_fraction_mean", "total_bits", "n_episodes"]
    table = df[[c for c in cols if c in df.columns]].sort_values(["name", "corruption"])
    table.to_csv(Path(args.results) / "summary.csv", index=False)

    gate = stage3_gate(df)
    with open(Path(args.results) / "stage3_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    if args.markdown:
        md = ["# Results summary\n", table.to_markdown(index=False),
              "\n## Stage 3 gate (sib vs raw_vib)\n",
              f"**Verdict:** {gate['verdict']}\n",
              "```json", json.dumps(gate["details"], indent=2), "```"]
        (Path(args.results) / "summary.md").write_text("\n".join(md))
        print("\n".join(md))
    print(f"\n[aggregate] verdict: {gate['verdict']}")


if __name__ == "__main__":
    main()
