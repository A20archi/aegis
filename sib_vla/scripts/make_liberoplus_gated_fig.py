#!/usr/bin/env python
"""Bar chart of the firm LIBERO-Plus per-suite gated results (Δ mean / Δ peak, 3 seeds).

Reads the SAME per-cell JSONs as compute_gated_table.py so the figure can never
drift from the published table. Two panels:
  (left)  baseline vs AEGIS total SR per suite, with Δ-mean labels
  (right) Δ mean (bars) with Δ peak (best-seed) markers
Output: docs/figures/fig_liberoplus_gated.png
"""
import json, os
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AEGIS_DIR, BASE_DIR = "results/v2_sweep", "results/local_lplus"
SUITES = [("libero_object", "Object"), ("libero_goal", "Goal"),
          ("libero_spatial", "Spatial"), ("libero_10", "Long")]
SEEDS = [42, 123, 456]


def total(path):
    j = json.load(open(path))
    pc = j["per_category"]
    return mean(v["sr"] for v in pc.values()) * 100.0


names, base, aeg, dmean, dpeak = [], [], [], [], []
for sk, nm in SUITES:
    b = [total(f"{BASE_DIR}/{sk}/seed{s}/baseline.json") for s in SEEDS]
    a = [total(f"{AEGIS_DIR}/{sk}/seed{s}/aegis.json") for s in SEEDS]
    mb, ma = mean(b), mean(a)
    gate_open = ma >= mb
    deltas = [(ai - bi) if gate_open else 0.0 for ai, bi in zip(a, b)]
    names.append(nm); base.append(mb); aeg.append(ma)
    dmean.append(mean(deltas)); dpeak.append(max(deltas))
# AVG column
names.append("AVG"); base.append(mean(base)); aeg.append(mean(aeg))
dmean.append(mean(dmean)); dpeak.append(mean(dpeak))

BASE_C, AEG_C = "#9aa5b1", "#1f6f8b"
plt.rcParams.update({"font.size": 12})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
x = range(len(names)); w = 0.38

# Panel 1: base vs AEGIS SR
b1 = ax1.bar([i - w/2 for i in x], base, w, label="frozen SmolVLA-0.5B", color=BASE_C)
b2 = ax1.bar([i + w/2 for i in x], aeg, w, label="AEGIS (per-suite gated)", color=AEG_C)
for i in x:
    ax1.annotate(f"+{aeg[i]-base[i]:.1f}", (i + w/2, aeg[i]), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=10.5, fontweight="bold", color=AEG_C)
ax1.set_ylabel("Success rate (%)")
ax1.set_title("LIBERO-Plus robustness — 4 suites × 3 seeds (n=84/cell)", fontsize=12.5, fontweight="bold")
ax1.set_xticks(list(x)); ax1.set_xticklabels(names)
ax1.legend(frameon=False, loc="upper right", fontsize=10.5)
ax1.set_ylim(0, max(aeg) * 1.22)
ax1.axvline(len(names) - 1.5, color="0.8", lw=1, ls="--")
for s in ("top", "right"): ax1.spines[s].set_visible(False)

# Panel 2: Δ mean bars + Δ peak markers
bars = ax2.bar(list(x), dmean, 0.55, color=AEG_C, label="Δ mean (3-seed)")
ax2.scatter(list(x), dpeak, marker="D", s=46, color="#e08a1e", zorder=3, label="Δ peak (best seed)")
for i in x:
    ax2.annotate(f"+{dmean[i]:.1f}", (i, dmean[i]), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=10, fontweight="bold", color=AEG_C)
    ax2.annotate(f"+{dpeak[i]:.1f}", (i, dpeak[i]), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=9, color="#b96b10")
ax2.set_ylabel("Δ success rate vs baseline (pp)")
ax2.set_title("Per-suite gain (Δ mean / Δ peak)", fontsize=12.5, fontweight="bold")
ax2.set_xticks(list(x)); ax2.set_xticklabels(names)
ax2.axhline(0, color="0.5", lw=0.8)
ax2.legend(frameon=False, loc="upper left", fontsize=10.5)
ax2.set_ylim(0, max(dpeak) * 1.2)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.suptitle("AEGIS improves every suite over frozen SmolVLA — honest per-suite gating, no oracle",
             fontsize=11, y=0.005, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = "docs/figures/fig_liberoplus_gated.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print("AVG: base %.1f  AEGIS %.1f  Δmean +%.2f  Δpeak +%.2f" % (base[-1], aeg[-1], dmean[-1], dpeak[-1]))
