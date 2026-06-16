"""Reverse water-filling correctness, and the beta <-> water-level theorem.

The headline check: the closed-form channel noise that minimises the compression
objective ``D + beta*R`` yields constant per-band distortion ``D_k = beta/2`` on
active bands and rate ``R_k = 0.5 ln(lambda_k/(beta/2))`` -- i.e. the
beta-penalised Gaussian channel *is* reverse water-filling at ``theta = beta/2``.
"""

import math

import torch

from sib.bottleneck import rate_nats, wiener_gain
from sib.waterfill import (allocation_matched_to_rate, optimal_sigma2_for_beta,
                           reverse_waterfilling, total_rate_at,
                           waterlevel_for_distortion, waterlevel_for_rate)

LAM = torch.tensor([4.0, 2.0, 1.0, 0.5, 0.25, 0.1])


def test_reverse_waterfilling_values_and_drops():
    theta = 0.5
    wf = reverse_waterfilling(LAM, theta)
    # active bands: lambda > theta
    assert wf["active"].tolist() == [True, True, True, False, False, False]
    # D_k = min(theta, lambda_k)
    assert torch.allclose(wf["D"], torch.tensor([0.5, 0.5, 0.5, 0.5, 0.25, 0.1]))
    # R_k = 0.5 ln(lambda/theta) for active, 0 otherwise
    assert math.isclose(wf["R"][0].item(), 0.5 * math.log(8.0), rel_tol=1e-6)
    assert wf["R"][3:].sum().item() == 0.0


def test_total_rate_monotone_decreasing_in_theta():
    thetas = torch.logspace(-3, 0.6, 40).tolist()
    rates = [total_rate_at(LAM, t) for t in thetas]
    assert all(rates[i + 1] <= rates[i] + 1e-9 for i in range(len(rates) - 1))


def test_waterlevel_for_rate_inverts():
    target = 1.7
    theta = waterlevel_for_rate(LAM, target)
    assert math.isclose(total_rate_at(LAM, theta), target, rel_tol=1e-4)


def test_waterlevel_for_distortion_inverts():
    target = 1.0
    theta = waterlevel_for_distortion(LAM, target)
    D = reverse_waterfilling(LAM, theta)["D"].sum().item()
    assert math.isclose(D, target, rel_tol=1e-4)


def test_beta_equals_two_theta_theorem():
    """Optimal compression channel => D_k = beta/2 on active bands; theta=beta/2."""
    beta = 1.0
    sigma2 = optimal_sigma2_for_beta(LAM, beta)
    active = LAM > beta / 2

    # Operational MMSE distortion of the Wiener channel (inf-safe via the gain).
    g = wiener_gain(LAM, sigma2)                          # 0 for dropped bands (sigma2=inf)
    D_op = LAM * (1.0 - g)                                # = lambda*sigma2/(lambda+sigma2)
    assert torch.allclose(D_op[active], torch.full_like(D_op[active], beta / 2), atol=1e-5)
    # Dropped bands keep all their variance as distortion.
    assert torch.allclose(D_op[~active], LAM[~active], atol=1e-5)

    # Operational rate equals reverse water-filling at theta = beta/2.
    R_op = rate_nats(LAM, sigma2)                         # 0.5 ln(1 + lambda/sigma2)
    R_wf = reverse_waterfilling(LAM, beta / 2)["R"]
    assert torch.allclose(R_op, R_wf, atol=1e-5)

    # And the optimal Wiener gain on active bands is 1 - theta/lambda.
    g = wiener_gain(LAM, sigma2)
    assert torch.allclose(g[active], 1.0 - (beta / 2) / LAM[active], atol=1e-5)


def test_matched_allocation_hits_target_rate():
    target = 2.0
    alloc = allocation_matched_to_rate(LAM, target)
    assert math.isclose(alloc["R"].sum().item(), target, rel_tol=1e-4)
    assert alloc["theta"] > 0
