"""Transform correctness: invertibility, Parseval, scipy agreement, rotations."""

import math

import numpy as np
import pytest
import torch
from scipy.fft import dct as scipy_dct

from sib.transforms import SpectralTransform, dct_ii_matrix, cayley_orthogonal

torch.manual_seed(0)
TOL = 1e-5
SHAPES = [(1, 50, 7), (4, 50, 7), (3, 16, 2), (2, 1, 5)]


@pytest.mark.parametrize("B,H,d", SHAPES)
def test_idct_dct_roundtrip(B, H, d):
    t = SpectralTransform(H)
    x = torch.randn(B, H, d, dtype=torch.float32)
    x_rec = t.synthesize(t.analyze(x))
    assert torch.allclose(x, x_rec, atol=TOL), (x - x_rec).abs().max().item()


@pytest.mark.parametrize("B,H,d", SHAPES)
def test_parseval(B, H, d):
    t = SpectralTransform(H)
    x = torch.randn(B, H, d, dtype=torch.float32)
    X = t.analyze(x)
    # Orthonormal change of basis preserves the L2 norm along the time axis.
    assert torch.allclose(x.norm(dim=1), X.norm(dim=1), atol=TOL)


def test_matrix_orthonormal():
    for H in (1, 2, 16, 50):
        C = dct_ii_matrix(H, dtype=torch.float64)
        I = torch.eye(H, dtype=torch.float64)
        assert torch.allclose(C @ C.T, I, atol=1e-10)
        assert torch.allclose(C.T @ C, I, atol=1e-10)


def test_agrees_with_scipy():
    H, d = 50, 7
    x = torch.randn(H, d, dtype=torch.float64)
    C = dct_ii_matrix(H, dtype=torch.float64)
    ours = (C @ x).numpy()
    theirs = scipy_dct(x.numpy(), type=2, norm="ortho", axis=0)
    assert np.allclose(ours, theirs, atol=1e-10)


def test_identity_transform_is_passthrough():
    H, d = 50, 7
    t = SpectralTransform(H, identity=True)
    x = torch.randn(2, H, d)
    assert torch.allclose(t.analyze(x), x, atol=TOL)
    assert torch.allclose(t.synthesize(x), x, atol=TOL)


def test_pca_rotation_orthonormal_and_invertible():
    H, d = 16, 3
    t = SpectralTransform(H, rotation="pca")
    R = torch.linalg.qr(torch.randn(H, H, dtype=torch.float32))[0]  # random orthogonal
    t.set_pca_rotation(R)
    T = t.analysis_matrix()
    assert torch.allclose(T @ T.T, torch.eye(H), atol=1e-4)
    x = torch.randn(4, H, d)
    assert torch.allclose(t.synthesize(t.analyze(x)), x, atol=1e-4)


def test_learned_rotation_cayley_orthogonal_and_invertible():
    H, d = 16, 3
    W = torch.randn(H, H) * 0.3
    R = cayley_orthogonal(W)
    assert torch.allclose(R @ R.T, torch.eye(H), atol=1e-5)
    t = SpectralTransform(H, rotation="learned")
    with torch.no_grad():
        t.W.copy_(W)
    x = torch.randn(2, H, d)
    assert torch.allclose(t.synthesize(t.analyze(x)), x, atol=1e-4)


def test_learned_rotation_zero_init_is_identity_rotation():
    H = 8
    t = SpectralTransform(H, rotation="learned")  # W = 0 -> R = I
    assert torch.allclose(t.rotation_matrix(), torch.eye(H), atol=1e-6)
