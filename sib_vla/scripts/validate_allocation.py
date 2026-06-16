"""Validate the learned bit allocation against reverse water-filling.

The planner's strongest single result: does the *learned* per-band rate
``{R_k}`` match the analytic reverse-water-filling prescription for the measured
source variances ``{lambda_k}`` at the **same total rate**?

    python scripts/validate_allocation.py --weights results/sib_beta0.001.pt

Loads a trained SIB checkpoint, reads ``lambda_k`` and the learned ``sigma_k^2``,
computes learned ``R_k`` and the matched-rate water-filling ``R_k^WF``, and
reports their correlation / L1 gap plus the water level ``theta`` (compared to
the compression-objective prediction ``beta/2``).  Writes a JSON summary and a
per-band overlay figure.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sib.bottleneck import build_module, rate_nats
from sib.utils import save_json
from sib.waterfill import allocation_matched_to_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg, H, d, name = ckpt["config"], ckpt["H"], ckpt["d"], ckpt["name"]
    if cfg["module"] not in ("sib", "raw_vib"):
        raise SystemExit(f"{name}: allocation validation only applies to rate modules")

    module = build_module(cfg, H, d)
    module.load_state_dict(ckpt["module_state"])
    module.eval()

    lam = module.lambda_estimator.value                          # (H, d)
    sigma2 = F.softplus(module.operator.log_var) + module.operator.eps
    R_learned = rate_nats(lam, sigma2)                           # (H, d) nats
    R_total = float(R_learned.sum())

    wf = allocation_matched_to_rate(lam, R_total)               # matched total rate
    R_wf = wf["R"]
    theta = wf["theta"]

    a = R_learned.flatten().detach().numpy()
    b = R_wf.flatten().detach().numpy()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    l1 = float(np.abs(a - b).mean())
    rel_l1 = float(np.abs(a - b).sum() / (np.abs(b).sum() + 1e-12))

    beta = float(cfg.get("beta", 0.0))
    report = {
        "name": name, "beta": beta,
        "total_rate_nats": R_total,
        "waterlevel_theta": theta,
        "beta_over_2_prediction": beta / 2.0,
        "pearson_corr_learned_vs_waterfill": corr,
        "mean_abs_rate_gap_nats": l1,
        "relative_l1_gap": rel_l1,
        "R_learned_per_band": R_learned.sum(dim=1).tolist(),     # (H,)
        "R_waterfill_per_band": R_wf.sum(dim=1).tolist(),
    }
    save_json(report, Path(args.results) / f"allocation_{name}.json")

    # Per-band overlay figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kk = np.arange(H)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(kk, R_learned.sum(dim=1).detach().numpy(), "o-", label="learned $R_k$")
        ax.plot(kk, R_wf.sum(dim=1).detach().numpy(), "s--",
                label=f"reverse water-filling ($\\theta$={theta:.3g})")
        ax.set_xlabel("frequency band k"); ax.set_ylabel("rate (nats, summed over dims)")
        ax.set_title(f"Learned vs analytic allocation: {name}  (r={corr:.3f})")
        ax.legend(); fig.tight_layout()
        fig.savefig(Path(args.results) / f"allocation_{name}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[validate_allocation] figure skipped: {e}")

    print(f"[validate_allocation] {name}: corr={corr:.3f} rel_L1={rel_l1:.3f} "
          f"theta={theta:.4g} (beta/2={beta/2:.4g}) total_rate={R_total:.2f} nats")


if __name__ == "__main__":
    main()
