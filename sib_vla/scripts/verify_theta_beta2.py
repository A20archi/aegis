"""Numerical confirmation of the theta = beta/2 theorem on pure compression.

For the compression objective ``D + beta*R`` over independent Gaussian bands,
the optimal channel noise ``sigma_k^2 = beta*lambda_k/(2*lambda_k - beta)`` gives
a *constant* operational distortion ``D_k = beta/2`` on every active band
(``2*lambda_k > beta``), i.e. water level ``theta = beta/2``.

We verify D_k = beta/2 exactly (to 1e-6) via the Wiener channel's operational
MMSE distortion D_op = lambda*(1 - g), g = lambda/(lambda+sigma2), and confirm
total_rate_at / waterlevel_for_rate invert each other.

NOTE: this holds for the COMPRESSION target (reconstruct the source A). The
deployed RASF uses a learned sigmoid gain trained with a 4-term denoising MSE
(target A_star != A); this theorem is the analytic anchor, not that objective.
"""

import torch

from sib.bottleneck import wiener_gain
from sib.waterfill import (optimal_sigma2_for_beta, total_rate_at,
                           waterlevel_for_rate)

torch.set_printoptions(precision=10)

# Deterministic, seed-free per-band variances (no RNG): 0.1, 0.2, ..., 4.0.
LAM = torch.arange(1, 41, dtype=torch.float64) * 0.1     # 40 bands, 0.1 .. 4.0
BETAS = [0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

print(f"lambda bands (n={LAM.numel()}): {LAM.min().item():.2f} .. {LAM.max().item():.2f}")
print(f"betas: {BETAS}\n")

import json, pathlib
rows = []
max_resid = 0.0
for beta in BETAS:
    sigma2 = optimal_sigma2_for_beta(LAM, beta)
    active = LAM > beta / 2.0
    g = wiener_gain(LAM, sigma2)                          # 0 on dropped bands (sigma2=inf)
    D_op = LAM * (1.0 - g)                                 # operational MMSE distortion
    target = torch.full_like(D_op[active], beta / 2.0)
    resid = (D_op[active] - target).abs().max().item()
    max_resid = max(max_resid, resid)
    n_active = int(active.sum())
    rows.append({"beta": beta, "active_bands": n_active, "total_bands": int(LAM.numel()),
                 "D_min_active": float(D_op[active].min()), "D_max_active": float(D_op[active].max()),
                 "max_abs_resid_vs_beta_over_2": resid})
    print(f"beta={beta:<5} active={n_active:2d}/{LAM.numel()}  "
          f"D_k range on active=[{D_op[active].min():.10f}, {D_op[active].max():.10f}]  "
          f"max|D_k - beta/2|={resid:.3e}")
    assert resid < 1e-6, f"D_k != beta/2 for beta={beta}: residual {resid}"
    # Dropped bands keep all their variance as distortion.
    assert torch.allclose(D_op[~active], LAM[~active], atol=1e-9)

print(f"\nMAX |D_k - beta/2| across all betas/bands: {max_resid:.3e}  (< 1e-6)")

# total_rate_at <-> waterlevel_for_rate inversion.
print("\nInversion total_rate_at / waterlevel_for_rate:")
max_inv = 0.0
inv_rows = []
for target_rate in [0.5, 1.7, 3.0, 6.0, 10.0]:
    theta = waterlevel_for_rate(LAM, target_rate)
    got = total_rate_at(LAM, theta)
    err = abs(got - target_rate)
    max_inv = max(max_inv, err)
    inv_rows.append({"target_rate": target_rate, "theta": float(theta),
                     "total_rate_at": float(got), "abs_err": float(err)})
    print(f"  target R={target_rate:<5} -> theta={theta:.8f} -> total_rate_at={got:.8f}  |err|={err:.3e}")
    assert err < 1e-4, f"inversion failed at rate {target_rate}: {err}"

print(f"\nMAX inversion |err|: {max_inv:.3e}  (< 1e-4)")
print("\nCONFIRMED: theta = beta/2 exactly on the compression objective; "
      "inversions hold. (Compression target, not the deployed denoising MSE.)")

# ---- browsable data artifact backing the machine-check claim ----
out = pathlib.Path(__file__).resolve().parents[1] / "results" / "theory_machine_check.json"
json.dump({"theorem": "theta = beta/2 on the compression objective D + beta*R",
           "lambda_bands": {"n": int(LAM.numel()), "min": float(LAM.min()), "max": float(LAM.max())},
           "betas": BETAS, "per_beta": rows,
           "max_abs_resid_vs_beta_over_2": max_resid, "resid_tolerance": 1e-6,
           "inversion": inv_rows, "max_inversion_err": max_inv, "inversion_tolerance": 1e-4,
           "conclusion": ("theta=beta/2 holds to %.3e (< 1e-6); rate<->level inversion holds to %.3e "
                          "(< 1e-4). Compression target only — NOT the deployed denoising MSE."
                          % (max_resid, max_inv))},
          open(out, "w"), indent=2)
print(f"[data] wrote {out}")
