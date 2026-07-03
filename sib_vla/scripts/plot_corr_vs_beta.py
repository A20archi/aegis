"""Summarize allocation validation across the beta sweep.

Reads every results/allocation_*.json and plots
  (a) Pearson corr(learned R_k, water-filling R_k) vs beta  (semilog-x)
  (b) water level theta vs beta/2                            (log-log)

    python scripts/plot_corr_vs_beta.py

Saves results/allocation_corr_vs_beta.png.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"

rows = []
for p in sorted(glob.glob(str(RES / "allocation_*.json"))):
    d = json.load(open(p))
    beta = d["beta"]
    if beta <= 0:
        continue  # non-rate / degenerate; nothing to place on a log axis
    rows.append((d["name"], beta, d["pearson_corr_learned_vs_waterfill"],
                 d["waterlevel_theta"], d["beta_over_2_prediction"]))

assert rows, "no allocation_*.json with beta>0 found"
rows.sort(key=lambda r: r[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

for name, beta, corr, theta, b2 in rows:
    ax1.plot(beta, corr, "o")
    ax1.annotate(name, (beta, corr), fontsize=7,
                 textcoords="offset points", xytext=(4, 4))
ax1.set_xscale("log")
ax1.set_xlabel(r"$\beta$")
ax1.set_ylabel(r"Pearson corr (learned $R_k$ vs water-filling $R_k$)")
ax1.set_title("Allocation shape match vs $\\beta$")
ax1.grid(True, which="both", ls=":", alpha=0.5)

for name, beta, corr, theta, b2 in rows:
    ax2.plot(b2, theta, "s")
    ax2.annotate(name, (b2, theta), fontsize=7,
                 textcoords="offset points", xytext=(4, 4))
lo = min(min(r[3] for r in rows), min(r[4] for r in rows))
hi = max(max(r[3] for r in rows), max(r[4] for r in rows))
ax2.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label=r"$\theta=\beta/2$")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel(r"$\beta/2$ (compression-objective prediction)")
ax2.set_ylabel(r"water level $\theta$ (matched-rate)")
ax2.set_title("Water level vs $\\beta/2$")
ax2.legend(); ax2.grid(True, which="both", ls=":", alpha=0.5)

fig.tight_layout()
out = RES / "allocation_corr_vs_beta.png"
fig.savefig(out, dpi=150)
print(f"[plot_corr_vs_beta] wrote {out} ({len(rows)} points)")
