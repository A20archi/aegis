"""Publication-grade README figures for the theory section (GPU-free, real data).

Produces:
  docs/figures/fig_waterfilling_theory.png  — 2-panel: schematic reverse water-filling
      (theorem) | real learned-vs-analytic allocation overlay (sib_b1e-2, r=0.991).
  docs/figures/fig_smoothness_trackb.png     — 2-panel: RMS jerk (vanilla vs Wiener) |
      ΔSR (Wiener SR-neutral vs naive low-pass hurts).

Numbers come from results/allocation_sib_b1e-2.json (Track A) and the locked Track-B
action-noise / jerk measurements (RIGOR_SUMMARY.md provenance). Run:
  /home/user/anaconda3/envs/vla-adapter/bin/python scripts/make_theory_figs.py
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parents[1]
FIG = HERE / "docs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

WATER = "#9ecae1"; WATER_E = "#3182bd"; SIG = "#fdae6b"; SIG_E = "#e6550d"
LEVEL = "#08519c"; LEARN = "#2171b5"; ANA = "#e6550d"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 160})


def fig_waterfilling():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    # ---- LEFT: schematic reverse water-filling (illustrative spectrum) ----
    lam = np.array([4.0, 3.1, 2.35, 1.7, 1.15, 0.78, 0.5, 0.34, 0.2, 0.12, 0.07, 0.04])
    theta = 0.6
    k = np.arange(len(lam))
    for i, l in enumerate(lam):
        axL.bar(i, min(theta, l), color=WATER, edgecolor=WATER_E, width=0.82, zorder=2)
        if l > theta:
            axL.bar(i, l - theta, bottom=theta, color=SIG, edgecolor=SIG_E, width=0.82, zorder=2)
    axL.axhline(theta, ls="--", color=LEVEL, lw=2.2, zorder=3)
    axL.text(len(lam) - 0.4, theta + 0.06, r"water level $\theta=\beta/2$",
             ha="right", va="bottom", color=LEVEL, fontsize=12, fontweight="bold")
    active = int((lam > theta).sum())
    axL.annotate("active bands\n(carry rate $R_k$)", xy=(1, 2.1), xytext=(3.4, 3.2),
                 fontsize=11, ha="left", color=SIG_E,
                 arrowprops=dict(arrowstyle="->", color=SIG_E))
    axL.annotate(r"dropped ($\lambda_k\leq\theta$)", xy=(active + 1.0, 0.28),
                 xytext=(active + 1.3, 1.5), fontsize=11, ha="left", color=WATER_E,
                 arrowprops=dict(arrowstyle="->", color=WATER_E))
    axL.set_xlabel("frequency band $k$"); axL.set_ylabel(r"source variance $\lambda_k$")
    axL.set_title("Reverse water-filling — the optimal allocation (schematic)")
    axL.set_ylim(0, 4.4); axL.set_xticks(k)
    axL.legend(handles=[
        Patch(facecolor=SIG, edgecolor=SIG_E, label=r"preserved signal $\Rightarrow$ rate $R_k=\frac{1}{2}\ln\frac{\lambda_k}{\theta}$"),
        Patch(facecolor=WATER, edgecolor=WATER_E, label=r"distortion $D_k=\min(\theta,\lambda_k)$")],
        loc="upper right", fontsize=10, framealpha=0.95)

    # ---- RIGHT: real learned vs analytic (sib_b1e-2, r=0.991) ----
    d = json.loads((HERE / "results" / "allocation_sib_b1e-2.json").read_text())
    rl = np.array(d["R_learned_per_band"]); rw = np.array(d["R_waterfill_per_band"])
    corr = d["pearson_corr_learned_vs_waterfill"]; th = d["waterlevel_theta"]
    n = min(16, len(rl)); kk = np.arange(n)
    axR.plot(kk, rl[:n], "o-", color=LEARN, lw=2, ms=6, label="learned $R_k$ (measured)", zorder=3)
    axR.plot(kk, rw[:n], "s--", color=ANA, lw=2, ms=6,
             label=fr"reverse water-filling ($\theta$={th:.2f})", zorder=2)
    axR.set_xlabel("frequency band $k$"); axR.set_ylabel("rate (nats, summed over dims)")
    axR.set_title(f"The learned filter matches the theory  (Pearson $r$={corr:.3f})")
    axR.legend(loc="upper right", fontsize=11)
    fig.tight_layout(); out = FIG / "fig_waterfilling_theory.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


def fig_smoothness():
    sig = ["0.05", "0.10", "0.20"]; x = np.arange(3); w = 0.36
    jerk_van = np.array([0.4511, 0.4886, 0.6280]); jerk_wie = np.array([0.1108, 0.1121, 0.1141])
    dsr_wie = np.array([4.0, 0.5, -2.0]); dsr_lp = np.array([-0.5, -2.5, -3.5])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    b1 = axL.bar(x - w / 2, jerk_van, w, color="#bdbdbd", edgecolor="#636363", label="vanilla")
    b2 = axL.bar(x + w / 2, jerk_wie, w, color="#41ab5d", edgecolor="#238b45", label="Wiener/MMSE decode")
    for i in range(3):
        axL.text(i, jerk_van[i] + 0.015, f"{jerk_van[i]/jerk_wie[i]:.1f}× lower",
                 ha="center", va="bottom", fontsize=10, color="#238b45", fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels([fr"$\sigma$={s}" for s in sig])
    axL.set_ylabel("RMS jerk (lower = smoother)"); axL.set_ylim(0, 0.75)
    axL.set_title("Wiener decode cuts jerk 4–5.5×"); axL.legend(loc="upper left", fontsize=11)

    axR.axhline(0, color="#333", lw=1)
    axR.bar(x - w / 2, dsr_wie, w, color="#41ab5d", edgecolor="#238b45", label="Wiener/MMSE (principled)")
    axR.bar(x + w / 2, dsr_lp, w, color="#cb181d", edgecolor="#a50f15", label="naive low-pass")
    for i in range(3):
        axR.text(i - w / 2, dsr_wie[i] + (0.15 if dsr_wie[i] >= 0 else -0.35), f"{dsr_wie[i]:+.1f}",
                 ha="center", fontsize=9, color="#238b45")
        axR.text(i + w / 2, dsr_lp[i] - 0.35, f"{dsr_lp[i]:+.1f}", ha="center", fontsize=9, color="#a50f15")
    axR.set_xticks(x); axR.set_xticklabels([fr"$\sigma$={s}" for s in sig])
    axR.set_ylabel("Δ success rate vs vanilla (pp)"); axR.set_ylim(-4.5, 5.5)
    axR.set_title("Same smoothing, opposite SR: signal kept vs discarded")
    axR.legend(loc="lower left", fontsize=11)
    fig.tight_layout(); out = FIG / "fig_smoothness_trackb.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_waterfilling(); fig_smoothness()
