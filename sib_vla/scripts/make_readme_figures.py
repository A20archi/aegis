#!/usr/bin/env python3
"""Generate the paper-grade README figures from the verified AEGIS numbers.

Pure-CPU matplotlib; no GPU, no Modal. All values are hard-coded from the
final, verified result tables (final_module_architecture.md / SAVED_STATE.md) so
the figures are reproducible and auditable. Outputs PNGs to docs/figures/.

    python sib_vla/scripts/make_readme_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- house style -----------------------------------------------------------
BASE   = "#94a3b8"   # slate-400  (frozen baseline)
AEGIS  = "#0d9488"   # teal-600   (our method)
FROZEN = "#e2e8f0"   # slate-200  (frozen backbone blocks)
INK    = "#0f172a"   # slate-900  (text)
GOOD   = "#0d9488"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _delta_grouped_barh(rows, title, subtitle, fname, mean_delta):
    """rows = [(label, base, aegis), ...] -> horizontal grouped bars sorted by delta."""
    rows = sorted(rows, key=lambda r: (r[2] - r[1]))   # ascending delta, biggest on top
    labels = [r[0] for r in rows]
    base   = np.array([r[1] for r in rows], float)
    aeg    = np.array([r[2] for r in rows], float)
    y = np.arange(len(rows)); h = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(rows) + 1.7))
    ax.barh(y + h/2, base, h, color=BASE,  label="Base (SmolVLA + TE)", zorder=3)
    ax.barh(y - h/2, aeg,  h, color=AEGIS, label="AEGIS (RIB + RASF + TE)", zorder=3)
    for yi, (b, a) in enumerate(zip(base, aeg)):
        d = a - b
        ax.annotate(f"+{d:.0f}" if d == int(d) else f"+{d:.1f}",
                    (max(a, b) + 1.5, yi), va="center", ha="left",
                    fontsize=9.5, fontweight="bold",
                    color=GOOD if d > 0 else "#64748b")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 109); ax.set_xlabel("Success rate (%)")
    ax.xaxis.grid(True, color="#eef2f7", zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.suptitle(title, x=0.012, ha="left", fontsize=15, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=10.5, color="#475569", pad=8)
    ax.annotate(f"mean Δ = +{mean_delta}  ·  0 regressions",
                (0.012, 0.0), xycoords="figure fraction", ha="left", va="bottom",
                fontsize=10, color=GOOD, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    p = os.path.join(OUT, fname); fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p))


# ---- Figure 1: cross-suite (Object + Goal), 10 conditions ------------------
_delta_grouped_barh(
    [("object · motion blur", 0, 86), ("object · gaussian noise", 36, 90),
     ("object · lighting", 58, 92),   ("object · texture", 83, 97),
     ("object · viewpoint (mod)", 0, 0), ("goal · motion blur", 19, 78),
     ("goal · viewpoint (mod)", 17, 43), ("goal · viewpoint (ext)", 8, 29),
     ("goal · texture", 90, 93),       ("goal · lighting", 80, 82)],
    "Cross-suite robustness  —  LIBERO-V (Object + Goal)",
    "Held-out suites (modules trained only on Spatial). AEGIS never regresses; rescues the dead cells.",
    "fig1_crosssuite_robustness.png", "29.9")

# ---- Figure 2: in-distribution Spatial, 6 axes -----------------------------
_delta_grouped_barh(
    [("motion blur", 4.0, 50.0), ("gaussian noise (σ=0.12)", 47.0, 61.0),
     ("lighting", 75.0, 84.5),   ("viewpoint (moderate)", 11.0, 20.5),
     ("texture", 82.0, 86.5),    ("viewpoint (extreme)", 0.0, 1.0)],
    "In-distribution robustness  —  LIBERO-V (Spatial), n=200/axis",
    "AEGIS wins all six perturbation axes.",
    "fig2_spatial_robustness.png", "14.1")


# ---- Figure 3: architecture schematic --------------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(9.4, 7.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc=INK, fs=10.5, bold=False, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
                                    fc=fc, ec=ec, lw=1.6, zorder=3))
        ax.text(x + w/2, y + h/2 + (0.16 if sub else 0), text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=4)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.26, sub, ha="center", va="center",
                    fontsize=8.2, color="#475569", style="italic", zorder=4)

    def arrow(x, y0, y1, label=None):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                                     lw=1.7, color="#475569", zorder=2))
        if label:
            ax.text(x + 0.15, (y0 + y1)/2, label, ha="left", va="center",
                    fontsize=8.2, color="#475569")

    cx, w = 2.7, 4.6
    # frozen backbone band
    ax.add_patch(FancyBboxPatch((0.5, 0.4), 9.0, 11.2, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#f8fafc", ec="#cbd5e1", lw=1.2, ls=(0, (5, 4)), zorder=1))
    ax.text(0.72, 11.28, "SmolVLA  —  backbone FROZEN", fontsize=10.5,
            color="#64748b", fontweight="bold", style="italic")

    box(cx, 10.0, w, 1.0, "RGB observation", "white", "#cbd5e1", fs=10)
    arrow(cx + w/2, 10.0, 9.55)
    box(cx, 8.5, w, 1.0, "Vision Encoder", FROZEN, "#94a3b8", sub="frozen · patch tokens")
    arrow(cx + w/2, 8.5, 8.05)
    box(cx, 6.9, w, 1.05, "RIB  ·  perception interface", "#ccfbf1", AEGIS, tc="#134e4a", bold=True,
        sub="VIB @ vision→LLM connector · ~2.27M · pass-through @ init")
    arrow(cx + w/2, 6.9, 6.45)
    box(cx, 5.3, w, 1.05, "Action Expert", FROZEN, "#94a3b8",
        sub="flow-matching ODE · frozen · chunk A:(H=50, d=7)")
    arrow(cx + w/2, 5.3, 4.85)
    box(cx, 3.7, w, 1.05, "RASF  ·  action interface", "#ccfbf1", AEGIS, tc="#134e4a", bold=True,
        sub="DCT-II per-band gain · bounded residual · pass-through @ init")
    arrow(cx + w/2, 3.7, 3.25)
    box(cx, 2.1, w, 1.05, "TE  ·  receding-horizon consensus", "#eef2ff", "#6366f1", tc="#3730a3",
        sub="overlapping-chunk average · in BOTH arms")
    arrow(cx + w/2, 2.1, 1.55)
    box(cx + 1.0, 0.7, w - 2.0, 0.85, "env step", "white", "#cbd5e1", fs=10)

    # side legend
    ax.add_patch(FancyBboxPatch((0.7, 0.95), 1.55, 1.7, boxstyle="round,pad=0.04,rounding_size=0.1",
                                fc="white", ec="#e2e8f0", lw=1.0, zorder=5))
    for i, (c, ec2, t) in enumerate([(FROZEN, "#94a3b8", "frozen"),
                                     ("#ccfbf1", AEGIS, "AEGIS\n(trained)"),
                                     ("#eef2ff", "#6366f1", "inference\nonly")]):
        ax.add_patch(FancyBboxPatch((0.85, 2.18 - i*0.52), 0.32, 0.26,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=c, ec=ec2, lw=1.3, zorder=6))
        ax.text(1.28, 2.31 - i*0.52, t, fontsize=7.6, va="center", color=INK, zorder=6)

    ax.text(5.0, 11.75, "AEGIS: one bottleneck, two interfaces, one consensus",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)
    p = os.path.join(OUT, "fig3_architecture.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p))


architecture()
print("done ->", os.path.relpath(OUT))
