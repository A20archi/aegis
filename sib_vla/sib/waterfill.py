"""Reverse water-filling: the analytic anchor for the learned bit allocation.

For independent Gaussian sources with per-band variances ``lambda_k``, the
rate-distortion function under sum-squared distortion is given by **reverse
water-filling**: pick a water level ``theta``, set per-band distortion
``D_k = min(theta, lambda_k)`` and per-band rate
``R_k = 0.5 * log(lambda_k / D_k) = max(0, 0.5 * log(lambda_k / theta))``.
Bands with ``lambda_k <= theta`` are dropped (``R_k = 0``).

Why this is the right thing to compare against (and a genuine result, not a
relabelling).  Our channel adds Gaussian noise ``sigma_k^2`` and decodes with the
Wiener gain; its operational MMSE distortion is the posterior variance
``D_k = lambda_k * sigma_k^2 / (lambda_k + sigma_k^2)`` and its rate is
``R_k = 0.5 * log(1 + lambda_k / sigma_k^2)``.  Eliminating ``sigma_k^2`` gives
``1 + lambda_k/sigma_k^2 = lambda_k / D_k``, hence ``R_k = 0.5 log(lambda_k/D_k)``
-- the water-filling rate with ``D_k`` the operational distortion.

Now minimise the **compression** objective ``sum_k D_k + beta * sum_k R_k``
(distortion target = source, rate in nats) over ``sigma_k^2``.  Setting the
derivative to zero (see ``optimal_sigma2_for_beta``) yields

    sigma_k^2 = beta * lambda_k / (2*lambda_k - beta)     (active iff 2*lambda_k > beta)
    D_k       = beta / 2                                   (constant across active bands)

i.e. the beta-penalised Gaussian channel **is** reverse water-filling with water
level ``theta = beta / 2``.  Bands with ``lambda_k <= beta/2`` are dropped.  Under
the denoising objective (target ``A_star != A``) this becomes the task-weighted
generalisation; ``scripts/validate_allocation.py`` measures how closely the
*learned* ``{R_k}`` tracks this analytic prescription at matched total rate.

All logs are natural (nats), to match ``sib.bottleneck.rate_nats``.  ``theta``,
``lambda``, ``D`` are all in the same (variance) units.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def reverse_waterfilling(lam: Tensor, theta: float) -> dict:
    """Per-band rate/distortion at water level ``theta``.

    Returns ``{'R': R_k, 'D': D_k, 'active': mask}`` with the input shape.
    """
    if theta <= 0:
        raise ValueError("theta must be > 0")
    D = torch.clamp(lam, max=theta)                       # min(theta, lambda_k)
    R = 0.5 * torch.clamp(torch.log(lam / theta), min=0.0)
    return {"R": R, "D": D, "active": lam > theta}


def total_rate_at(lam: Tensor, theta: float) -> float:
    return float(reverse_waterfilling(lam, theta)["R"].sum())


def total_distortion_at(lam: Tensor, theta: float) -> float:
    return float(reverse_waterfilling(lam, theta)["D"].sum())


def waterlevel_for_rate(lam: Tensor, rate_target: float, iters: int = 100) -> float:
    """Solve ``sum_k R_k(theta) = rate_target`` for ``theta`` (geometric bisection).

    ``R(theta)`` is continuous and strictly decreasing on ``(0, max lambda]``,
    with ``R -> inf`` as ``theta -> 0`` and ``R = 0`` at ``theta = max lambda``.
    """
    lam = lam.flatten().double()
    lam_max = float(lam.max())
    if rate_target <= 0:
        return lam_max
    lo, hi = lam_max * 1e-12, lam_max                     # R(lo) huge, R(hi)=0
    for _ in range(iters):
        mid = math.sqrt(lo * hi)                          # geometric midpoint (theta > 0)
        if total_rate_at(lam, mid) > rate_target:
            lo = mid                                      # rate too high -> raise theta
        else:
            hi = mid
    return math.sqrt(lo * hi)


def waterlevel_for_distortion(lam: Tensor, distortion_target: float,
                              iters: int = 100) -> float:
    """Solve ``sum_k min(theta, lambda_k) = distortion_target`` for ``theta``."""
    lam = lam.flatten().double()
    total = float(lam.sum())
    if distortion_target >= total:
        return float(lam.max())
    lo, hi = 0.0, float(lam.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if total_distortion_at(lam, mid) < distortion_target:
            lo = mid                                      # distortion too low -> raise theta
        else:
            hi = mid
    return 0.5 * (lo + hi)


def optimal_sigma2_for_beta(lam: Tensor, beta: float) -> Tensor:
    """Closed-form channel noise minimising ``D + beta*R`` (compression objective).

    ``sigma_k^2 = beta*lambda_k / (2*lambda_k - beta)`` where ``2*lambda_k > beta``;
    ``+inf`` (dropped band) otherwise.  Corresponds to water level ``theta = beta/2``.
    """
    denom = 2.0 * lam - beta
    sigma2 = torch.where(denom > 0, beta * lam / denom.clamp(min=1e-30),
                         torch.full_like(lam, float("inf")))
    return sigma2


def allocation_matched_to_rate(lam: Tensor, rate_target: float) -> dict:
    """Reverse-water-filling allocation whose total rate equals ``rate_target``."""
    theta = waterlevel_for_rate(lam, rate_target)
    out = reverse_waterfilling(lam, theta)
    out["theta"] = theta
    return out
