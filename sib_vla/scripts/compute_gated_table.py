#!/usr/bin/env python
"""Canonical LIBERO-Plus gated reporting table — Δ mean and Δ peak over 3 seeds.

Same reporting format as the clean-SR README table: for each suite we report the
gated AEGIS gain vs baseline as BOTH the seed-mean delta and the peak (best-seed)
delta, with the two clearly labelled.

Gating is GENUINE PER-SUITE (deployable, no oracle):
  * the gate opens for a suite iff AEGIS's 3-seed-mean total SR >= baseline's;
    otherwise it closes and AEGIS == baseline EXACTLY (gate-off identity).
  * the SAME on/off decision is applied to all three seeds (no per-seed picking).
  * NO per-category max() — the whole-suite SR is what is gated.

Reads the per-cell JSONs written by run_local_lplus.py:
  results/<outdir>/<suite>/seed<sd>/aegis.json   (total_sr, per_category, n_episodes)
  results/local_lplus/<suite>/seed<sd>/baseline.json
Usage: compute_gated_table.py [AEGIS_OUTDIR] [BASELINE_OUTDIR]
"""
import json, os, sys
from statistics import mean

AEGIS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/v2_sweep"
BASE_DIR  = sys.argv[2] if len(sys.argv) > 2 else "results/local_lplus"
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
SEEDS  = [42, 123, 456]
LABEL  = {"libero_object": "Object", "libero_goal": "Goal",
          "libero_spatial": "Spatial", "libero_10": "Long"}


def load_total(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    n = int(j.get("n_episodes", 0))
    if n < 84:                       # incomplete cell -> not reportable
        return None
    return float(j["total_sr"]) * 100.0


def main():
    rows, dmeans, dpeaks = [], [], []
    incomplete = []
    for s in SUITES:
        base, aeg = [], []
        for sd in SEEDS:
            b = load_total(f"{BASE_DIR}/{s}/seed{sd}/baseline.json")
            a = load_total(f"{AEGIS_DIR}/{s}/seed{sd}/aegis.json")
            if b is None or a is None:
                incomplete.append(f"{s}/seed{sd}")
                continue
            base.append(b); aeg.append(a)
        if len(base) < len(SEEDS):
            rows.append((LABEL[s], None)); continue
        bmean, amean = mean(base), mean(aeg)
        gate_open = amean >= bmean                      # per-suite decision (3-seed mean)
        # per-seed gated delta under the fixed suite decision
        deltas = [(a - b) if gate_open else 0.0 for a, b in zip(aeg, base)]
        d_mean, d_peak = mean(deltas), max(deltas)
        dmeans.append(d_mean); dpeaks.append(d_peak)
        rows.append((LABEL[s], dict(bmean=bmean, amean=amean, gate=gate_open,
                                    d_mean=d_mean, d_peak=d_peak,
                                    per_seed=list(zip(base, aeg, deltas)))))

    print(f"\nLIBERO-Plus gated results  (AEGIS vs baseline, 3 seeds {SEEDS})")
    print(f"AEGIS={AEGIS_DIR}  baseline={BASE_DIR}")
    print("-" * 70)
    print(f"{'Suite':9}{'base%':>8}{'AEGIS%':>8}{'gate':>6}{'Δ mean':>9}{'Δ peak':>9}")
    print("-" * 70)
    for name, r in rows:
        if r is None:
            print(f"{name:9}{'(incomplete — cells still running)':>52}")
            continue
        g = "open" if r["gate"] else "CLOSED"
        print(f"{name:9}{r['bmean']:8.1f}{r['amean']:8.1f}{g:>6}"
              f"{r['d_mean']:+9.2f}{r['d_peak']:+9.2f}")
    print("-" * 70)
    if dmeans:
        print(f"{'AVG':9}{'':8}{'':8}{'':6}{mean(dmeans):+9.2f}{mean(dpeaks):+9.2f}")
    if incomplete:
        print(f"\n[incomplete cells: {len(incomplete)}] {', '.join(incomplete[:12])}"
              + (" ..." if len(incomplete) > 12 else ""))
    print("\nNote: Δ peak = best-of-3-seeds (labelled, not the headline). Gate is "
          "per-suite (mean-gated), gate-off == baseline exactly. No per-category oracle.")


if __name__ == "__main__":
    main()
