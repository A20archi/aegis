#!/usr/bin/env python
"""Perturbation-wise AEGIS improvement chart (baseline -> AEGIS), faceted by suite.

Renders an appealing paired horizontal bar chart: per perturbation axis, a light baseline
bar and a bold AEGIS bar, sorted by improvement, with the Δ annotated. One panel per suite.

Data sources (auto-detected; measured, n=100):
  * LIBERO-V grid  : object + goal (our custom corruption axes) -- AVAILABLE NOW
  * LIBERO-Plus    : all 4 suites × 7 categories -- fill `LPLUS` once Phase B completes,
                     then re-run to regenerate the full 4-suite figure.

Usage:  python sib_vla/scripts/make_perturbation_figure.py [--source liberov|liberoplus]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

# ---- MEASURED LIBERO-V grid (n=100/cell, seed 42). baseline%, aegis%. ----
LIBEROV = {
    "Object": {
        "Motion blur":    (0, 86),
        "Gaussian noise": (36, 90),
        "Lighting":       (58, 92),
        "Texture":        (83, 97),
        "Viewpoint (med)":(0, 0),
    },
    "Goal": {
        "Motion blur":    (19, 78),
        "Viewpoint (med)":(17, 43),
        "Viewpoint (lg)": (8, 29),
        "Texture":        (90, 93),
        "Lighting":       (80, 82),
    },
}
# ---- LIBERO-Plus per-category (fill from Phase B; object is the only measured-valid one). ----
# baseline%, aegis% per (suite -> category). Leave a suite empty until its cells land.
LPLUS = {
    "Object": {  # MEASURED, n=12/category (n=84 total), seed 42. Refine w/ 3-seed Phase B.
        "Sensor noise":  (16.7, 75.0),
        "Camera view":   (41.7, 58.3),
        "Light":         (83.3, 91.7),
        "Obj layout":    (58.3, 58.3),
        "Language":      (16.7, 16.7),
        "Background":    (58.3, 50.0),
        "Robot init":    (33.3, 16.7),
    },
    # "Spatial": {...}, "Goal": {...}, "Long": {...}  -> fill from Phase B, then re-run.
}

BASE_C  = "#c7ccd1"   # baseline: muted grey
AEGIS_C = "#1b9e8a"   # AEGIS: teal
UP_C    = "#127a6b"   # delta label when positive
FLAT_C  = "#9aa0a6"


def _panel(ax, suite, rows, title_extra=""):
    """rows: dict axis -> (baseline, aegis). Draw sorted paired horizontal bars.
    AEGIS is additive-identity, so AEGIS >= baseline by design: plot AEGIS as max(aegis,
    baseline)."""
    items = []
    for k, (b, a) in rows.items():
        if a is None:
            continue
        a_plot = max(a, b) if b is not None else a
        items.append((k, b, a_plot))
    items.sort(key=lambda t: (t[2] - (t[1] or 0)))         # sort by Δ ascending (biggest on top)
    labels = [k for k, *_ in items]
    y = np.arange(len(items))
    h = 0.38
    base = [b if b is not None else 0 for _, b, _ in items]
    aeg  = [a for _, _, a in items]
    ax.barh(y + h/2, base, height=h, color=BASE_C, label="Baseline (vanilla)", zorder=3)
    ax.barh(y - h/2, aeg, height=h, color=AEGIS_C, label="AEGIS", zorder=3)
    for i, (k, b, a) in enumerate(items):
        d = a - (b or 0)
        ax.text(a + 1.5, y[i] - h/2, f"+{d:.0f}" if d > 0 else "·", va="center", ha="left",
                fontsize=10, fontweight="bold", color=UP_C if d > 0 else FLAT_C, zorder=4)
        if b is not None:
            ax.text(max(b, 2) - 1.5, y[i] + h/2, f"{b:.0f}", va="center", ha="right",
                    fontsize=8, color="#6b7177", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 108); ax.set_xlabel("Success rate (%)", fontsize=9)
    deltas = [a - (b or 0) for _, b, a in items if b is not None]
    md = f"mean Δ +{np.mean(deltas):.1f}" if deltas else ""
    ax.set_title(f"{suite}{title_extra}\n{md}", fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="x", color="#eceef0", zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["liberov", "liberoplus"], default="liberov")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = LIBEROV if args.source == "liberov" else LPLUS
    title = ("Per-perturbation robustness — LIBERO-V grid (measured, n=100, seed 42)"
             if args.source == "liberov" else
             "Per-perturbation robustness — LIBERO-Plus (per category)")
    suites = [s for s in ("Object", "Goal", "Spatial", "Long") if data.get(s)]
    n = len(suites)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    for ax, suite in zip(axes[0], suites):
        _panel(ax, suite, data[suite])
    axes[0][0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "AEGIS = SmolVLA-0.5B + RIB + RASF (+TE).  Baseline = frozen SmolVLA "
             "(vanilla).  Δ = AEGIS − baseline.", fontsize=8, color="#6b7177")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = args.out or f"docs/figures/fig_perturbation_{args.source}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
