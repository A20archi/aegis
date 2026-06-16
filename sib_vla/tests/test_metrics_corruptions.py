"""Metrics, corruptions, and loss-regulariser guards (they feed reported numbers)."""

import numpy as np
import pytest
import torch

from sib import corruptions as C
from sib import metrics as M
from sib.losses import jerk_penalty, third_difference


# --------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,sev", C.eval_corruption_list())
def test_corruption_preserves_shape_range_and_alters(name, sev):
    img = torch.rand(3, 32, 32)
    out = C.apply(img, name, sev)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, img)             # not a no-op


def test_corruption_uint8_roundtrip_and_batched():
    img = torch.rand(3, 16, 16)
    u8 = (img * 255).to(torch.uint8)
    assert C.apply(u8, "gaussian_noise", 0).dtype == torch.uint8
    assert C.apply(img.unsqueeze(0), "gaussian_blur", 1).shape == (1, 3, 16, 16)


def test_corruption_list_is_six():
    assert len(C.eval_corruption_list()) == 6      # 3 families x 2 severities


def test_perturb_actions():
    a = torch.zeros(8, 7)
    assert torch.allclose(C.perturb_actions(a, 0.0), a)        # std=0 -> identity
    g = torch.Generator().manual_seed(0)
    out = C.perturb_actions(a, 0.1, generator=g)
    assert out.shape == a.shape and not torch.allclose(out, a)
    assert abs(out.mean().item()) < 0.05                       # zero-mean noise


# --------------------------------------------------------------------------- #
# Trajectory metrics
# --------------------------------------------------------------------------- #
def test_rms_jerk_zero_for_linear_and_nan_for_short():
    ramp = np.linspace(0, 1, 200)[:, None] * np.ones((1, 7))
    assert M.rms_jerk(ramp) < 1e-9
    assert np.isnan(M.rms_jerk(np.zeros((2, 7))))   # T < 4


def test_hf_energy_fraction_separates_high_and_low():
    hi = np.cos(0.85 * np.pi * np.arange(64))[:, None]   # band ~54 >> cutoff 32
    lo = np.cos(0.05 * np.pi * np.arange(64))[:, None]   # band ~3
    assert M.hf_energy_fraction(hi, 0.5) > 0.9
    assert M.hf_energy_fraction(lo, 0.5) < 0.05


# --------------------------------------------------------------------------- #
# Inference statistics
# --------------------------------------------------------------------------- #
def test_wilson_ci_bracket_and_empty():
    p, lo, hi = M.wilson_ci(8, 10)
    assert 0 <= lo < p < hi <= 1
    assert np.isnan(M.wilson_ci(0, 0)[0])


def test_two_proportion_ztest_sign_and_null():
    z, pv = M.two_proportion_ztest(80, 100, 60, 100)
    assert z > 0 and 0 < pv < 0.05
    z0, pv0 = M.two_proportion_ztest(50, 100, 50, 100)
    assert abs(z0) < 1e-9 and abs(pv0 - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Loss regularisers
# --------------------------------------------------------------------------- #
def test_third_difference_and_jerk():
    A = torch.randn(4, 50, 7)
    assert third_difference(A).shape == (4, 47, 7)
    assert torch.allclose(jerk_penalty(A), torch.diff(A, n=3, dim=1).pow(2).mean())
    assert jerk_penalty(torch.randn(2, 3, 7)) == 0  # H < 4 -> empty -> 0
