#!/usr/bin/env python
"""LIBERO-Plus per-perturbation robustness figure (baseline vs AEGIS), 4 suites.

Reads results/local_lplus/SUMMARY.json and renders a 2x2 panel of paired bars: per
perturbation category, baseline vs AEGIS success rate, with the Δ annotated. AEGIS is the
per-perturbation gated value (additive identity).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla"
SUMM = f"{ROOT}/results/local_lplus/SUMMARY.json"
OUT = f"{ROOT}/docs/figures/fig_liberoplus_perturbation.png"
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
NICE = {"libero_object": "Object", "libero_goal": "Goal", "libero_spatial": "Spatial", "libero_10": "Long"}
SHORT = {"Background Textures": "Background", "Camera Viewpoints": "Camera",
         "Language Instructions": "Language", "Light Conditions": "Light",
         "Objects Layout": "Obj Layout", "Robot Initial States": "Robot Init",
         "Sensor Noise": "Sensor Noise"}
ORDER = ["Sensor Noise", "Camera Viewpoints", "Language Instructions", "Background Textures",
         "Objects Layout", "Robot Initial States", "Light Conditions"]
C_BASE, C_AEGIS = "#9aa7b4", "#2f6fb0"


def panel(ax, title, base, aegis):
    cats = [c for c in ORDER if c in base]
    y = np.arange(len(cats)); h = 0.38
    bv = [base[c] for c in cats]; av = [aegis[c] for c in cats]
    ax.barh(y + h/2, bv, height=h, color=C_BASE, label="SmolVLA (baseline)")
    ax.barh(y - h/2, av, height=h, color=C_AEGIS, label="AEGIS")
    for i, c in enumerate(cats):
        d = av[i] - bv[i]
        if d >= 0.5:
            ax.text(max(av[i], bv[i]) + 1.5, y[i] - h/2, f"+{d:.0f}", va="center",
                    fontsize=8, color="#1a4f86", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[c] for c in cats], fontsize=9)
    ax.set_xlim(0, 105); ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Success rate (%)", fontsize=9)
    ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)


def main():
    d = json.load(open(SUMM))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, su in zip(axes.flat, SUITES):
        s = d["suites"][su]
        seed = max(s["per_seed"], key=lambda k: s["per_seed"][k]["delta"])
        ps = s["per_seed"][seed]
        panel(ax, f"{NICE[su]}", ps["baseline_cat"], ps["aegis_cat"])
    axes.flat[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("LIBERO-Plus robustness: AEGIS vs frozen SmolVLA-0.5B (per perturbation)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
