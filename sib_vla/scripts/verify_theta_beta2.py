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
for target_rate in [0.5, 1.7, 3.0, 6.0, 10.0]:
    theta = waterlevel_for_rate(LAM, target_rate)
    got = total_rate_at(LAM, theta)
    err = abs(got - target_rate)
    max_inv = max(max_inv, err)
    print(f"  target R={target_rate:<5} -> theta={theta:.8f} -> total_rate_at={got:.8f}  |err|={err:.3e}")
    assert err < 1e-4, f"inversion failed at rate {target_rate}: {err}"

print(f"\nMAX inversion |err|: {max_inv:.3e}  (< 1e-4)")
print("\nCONFIRMED: theta = beta/2 exactly on the compression objective; "
      "inversions hold. (Compression target, not the deployed denoising MSE.)")
