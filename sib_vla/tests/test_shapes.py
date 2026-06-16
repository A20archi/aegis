"""Shape invariants: every module preserves the chunk shape (B, H, d)."""

import pytest
import torch

from sib.bottleneck import build_module, SpectralActionModule

torch.manual_seed(0)

B, H, d = 4, 50, 7

CONFIGS = [
    {"module": "sib", "rotation": "none"},
    {"module": "sib", "rotation": "pca"},
    {"module": "raw_vib"},
    {"module": "lowpass", "cutoff": 8},
    {"module": "gain_no_rate"},
    {"module": "jerk"},
    {"module": "sib", "sigma_mode": "context", "context_dim": 32},
    {"module": "sib", "decode": "stochastic"},
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_chunk_shape_preserved(cfg):
    module = build_module(cfg, H, d)
    if cfg.get("rotation") == "pca":
        R = torch.linalg.qr(torch.randn(H, H))[0]
        module.transform.set_pca_rotation(R)
    A = torch.randn(B, H, d)
    context = torch.randn(B, cfg["context_dim"]) if cfg.get("context_dim") else None
    out = module(A, context=context)
    assert out["A_hat"].shape == (B, H, d)
    assert out["X"].shape == (B, H, d)
    assert out["X_hat"].shape == (B, H, d)


def test_bits_per_band_shape():
    module = build_module({"module": "sib"}, H, d).eval()
    module(torch.randn(B, H, d))                 # warm lambda
    b = module.bits_per_band()
    assert b.shape == (H, d)
    assert torch.all(b >= 0)


def test_non_rate_modules_have_no_bits():
    for mt in ("gain_no_rate", "jerk"):
        module = build_module({"module": mt}, H, d)
        assert module.bits_per_band() is None


def test_eval_mmse_chunk_is_deterministic():
    module = build_module({"module": "sib", "decode": "mmse"}, H, d).eval()
    A = torch.randn(B, H, d)
    module(A, update_lambda=False)
    out1 = module(A, update_lambda=False)["A_hat"]
    out2 = module(A, update_lambda=False)["A_hat"]
    assert torch.allclose(out1, out2)
