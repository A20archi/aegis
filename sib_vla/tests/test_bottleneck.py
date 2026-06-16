"""Bottleneck correctness: Wiener-gain and rate limits, monotonicity, gradients."""

import math

import pytest
import torch
import torch.nn as nn

from sib.bottleneck import (
    GaussianChannel,
    SpectralActionModule,
    rate_nats,
    wiener_gain,
)
from sib.losses import total_loss

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Closed-form limits of the Wiener gain and the rate.
# --------------------------------------------------------------------------- #
def test_gain_limits():
    lam = torch.tensor([1.0, 2.0, 0.5])
    assert torch.allclose(wiener_gain(lam, torch.full_like(lam, 1e-12)),
                          torch.ones_like(lam), atol=1e-6)          # sigma2 -> 0 => gain -> 1
    assert torch.allclose(wiener_gain(lam, torch.full_like(lam, 1e12)),
                          torch.zeros_like(lam), atol=1e-6)         # sigma2 -> inf => gain -> 0


def test_gain_in_unit_interval():
    lam = torch.rand(200).abs() + 1e-6
    sigma2 = torch.rand(200).abs() + 1e-6
    g = wiener_gain(lam, sigma2)
    assert torch.all(g >= 0.0) and torch.all(g <= 1.0)


def test_rate_limits():
    lam = torch.tensor([1.0, 2.0, 0.5])
    big = rate_nats(lam, torch.full_like(lam, 1e-12))
    assert torch.all(big > 10.0)                                    # sigma2 -> 0 => R large
    small = rate_nats(lam, torch.full_like(lam, 1e12))
    assert torch.allclose(small, torch.zeros_like(small), atol=1e-6)  # sigma2 -> inf => R -> 0


def test_rate_monotone_decreasing_in_sigma2():
    lam = torch.tensor(1.0)
    sigma2 = torch.logspace(-3, 3, 50)
    r = rate_nats(lam, sigma2)
    assert torch.all(r[1:] < r[:-1])                                # strictly decreasing


def test_rate_equals_mutual_information_reference():
    # R = 0.5 ln(1 + lam/sigma2); cross-check against an explicit value.
    lam, sigma2 = torch.tensor(4.0), torch.tensor(1.0)
    assert math.isclose(rate_nats(lam, sigma2).item(),
                        0.5 * math.log(5.0), rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# Forward behaviour of the channel.
# --------------------------------------------------------------------------- #
def test_channel_small_sigma_recovers_input():
    H, d = 8, 3
    ch = GaussianChannel(H, d, init_log_var=-30.0).eval()           # sigma2 ~ eps
    X = torch.randn(4, H, d)
    lam = torch.ones(H, d)
    X_hat, info = ch(X, lam)
    assert torch.all(info["gain"] > 0.999)                          # gain -> 1
    assert torch.allclose(X_hat, X, atol=1e-3)                      # MMSE decode -> X


def test_channel_large_sigma_kills_signal():
    H, d = 8, 3
    ch = GaussianChannel(H, d, init_log_var=1e6).eval()             # sigma2 ~ 1e6 (softplus(x)=x for large x)
    X = torch.randn(4, H, d)
    lam = torch.ones(H, d)
    X_hat, info = ch(X, lam)
    assert torch.all(info["gain"] < 1e-3)
    assert torch.allclose(X_hat, torch.zeros_like(X_hat), atol=1e-3)
    assert info["rate"].item() < 1e-3


def test_mmse_is_deterministic_stochastic_is_not():
    H, d = 8, 3
    X = torch.randn(4, H, d)
    lam = torch.ones(H, d)
    mmse = GaussianChannel(H, d, decode="mmse").eval()
    torch.manual_seed(1); a = mmse(X, lam)[0]
    torch.manual_seed(2); b = mmse(X, lam)[0]
    assert torch.allclose(a, b)                                     # MMSE: seed-independent
    stoch = GaussianChannel(H, d, decode="stochastic").eval()
    torch.manual_seed(1); c = stoch(X, lam)[0]
    torch.manual_seed(2); e = stoch(X, lam)[0]
    assert not torch.allclose(c, e)                                 # stochastic: seed-dependent


# --------------------------------------------------------------------------- #
# Gradients: reach the module, never the frozen "base".
# --------------------------------------------------------------------------- #
def test_gradient_reaches_log_var_not_frozen_base():
    H, d = 16, 4
    # A stand-in frozen base: produces the chunk, must receive no gradient.
    base = nn.Linear(H * d, H * d)
    base.requires_grad_(False)
    module = SpectralActionModule("sib", H, d).train()

    obs = torch.randn(8, H * d)
    with torch.no_grad():                                           # sampler is frozen
        A = base(obs).view(8, H, d)
    out = module(A)
    A_star = torch.randn(8, H, d)
    loss, _ = total_loss({"module": "sib", "beta": 1e-2}, out, A_star)
    loss.backward()

    assert module.operator.log_var.grad is not None
    assert module.operator.log_var.grad.abs().sum() > 0
    for p in base.parameters():
        assert p.grad is None                                       # frozen base: no grad


def test_lambda_is_detached_buffer():
    H, d = 8, 2
    module = SpectralActionModule("sib", H, d).train()
    A = torch.randn(4, H, d, requires_grad=True)
    out = module(A)
    assert not module.lambda_estimator.value.requires_grad
    out["rate"].backward()
    # rate depends on sigma2 (=> log_var), not on lambda or A.
    assert module.operator.log_var.grad is not None
