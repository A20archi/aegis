"""Regression tests for the eval-time perturbation paths.

These are the paths that silently crashed a whole action-noise sweep: the shared
eval generator lives on CUDA, but the action tensor reaching ``perturb_actions``
is on CPU, and image-corruption tensors may be on either device. torch.randn
rejects a cross-device generator, so every eval died at step 0 while the
orchestration ran to "DONE" with zero results. These tests pin the device/dtype
contract so it can't regress.
"""
import pytest
import torch

from sib import corruptions as corr
from sib.bottleneck import SpectralActionModule

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


# --- perturb_actions: the exact bug (CPU action + CUDA generator) -------------
@pytest.mark.parametrize("act_dev", DEVICES)
@pytest.mark.parametrize("gen_dev", DEVICES)
def test_perturb_actions_cross_device_generator(act_dev, gen_dev):
    """Must not crash for ANY (action_device, generator_device) combination."""
    actions = torch.zeros(8, 7, device=act_dev)
    g = torch.Generator(device=gen_dev).manual_seed(0)
    out = corr.perturb_actions(actions, std=0.1, generator=g)
    assert out.shape == actions.shape and out.device.type == act_dev
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0  # noise was actually added


def test_perturb_actions_zero_is_identity():
    a = torch.randn(4, 7)
    assert torch.equal(corr.perturb_actions(a, 0.0), a)


def test_perturb_actions_scale():
    a = torch.zeros(20000, 7)
    g = torch.Generator(device="cpu").manual_seed(1)
    out = corr.perturb_actions(a, std=0.3, generator=g)
    assert abs(out.std().item() - 0.3) < 0.02  # right noise scale


# --- corruptions.apply: every family, both devices, float & uint8 -------------
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("name", list(corr.SEVERITY.keys()))
@pytest.mark.parametrize("sev", [0, 1])
def test_corruption_apply_device_dtype(dev, name, sev):
    g = torch.Generator(device=dev).manual_seed(0)
    img = torch.rand(2, 3, 32, 32, device=dev)  # float [0,1] BCHW
    out = corr.apply(img, name, sev, generator=g)
    assert out.shape == img.shape and out.device.type == dev
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dev", DEVICES)
def test_corruption_apply_cross_device_generator(dev):
    """Mirror eval: generator on one device, image possibly on another."""
    other = "cuda" if (dev == "cpu" and CUDA) else "cpu"
    g = torch.Generator(device=other).manual_seed(0)
    img = torch.rand(2, 3, 16, 16, device=dev)
    out = corr.apply(img, "gaussian_noise", 1, generator=g)
    assert out.shape == img.shape and torch.isfinite(out).all()


# --- module forward across every variant/device (shape/device contract) -------
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kwargs", [
    {},
    {"preserve_energy": True},
    {"adaptive_sigma": True},
    {"preserve_energy": True, "adaptive_sigma": True},
])
def test_module_forward_variants(dev, kwargs):
    m = SpectralActionModule("sib", 50, 7, **kwargs).to(dev).eval()
    A = torch.randn(4, 50, 7, device=dev)
    out = m(A, update_lambda=True)
    assert out["A_hat"].shape == A.shape and out["A_hat"].device.type == dev
    assert torch.isfinite(out["A_hat"]).all()
