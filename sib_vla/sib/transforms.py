"""Orthonormal spectral transforms for the action chunk.

The action chunk ``A`` has shape ``(B, H, d)`` where ``H`` is the temporal
horizon (number of action steps in the chunk) and ``d`` the action dimension.
We change basis along the *time* axis ``H`` only; the ``d`` action coordinates
are transformed independently and identically.

The forward (analysis) operator is

    T = R @ C

where ``C`` is an orthonormal DCT-II matrix and ``R`` is an optional orthogonal
rotation that further decorrelates the bands.  Because both ``C`` and ``R`` are
orthogonal, ``T`` is orthogonal, the synthesis operator is exactly ``T.T``, and
Parseval's identity ``||T A|| = ||A||`` holds.  This is what makes the rate term
in :mod:`sib.bottleneck` a change-of-basis-invariant statement about the signal.

``identity=True`` replaces the DCT with ``C = I`` (the ``raw_vib`` baseline):
the channel then acts on raw time-step coordinates and the whole spectral story
is removed while everything else is held fixed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def dct_ii_matrix(H: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Return the ``(H, H)`` orthonormal DCT-II matrix ``C``.

    ``C[k, h] = alpha(k) * sqrt(2/H) * cos(pi * (2h + 1) * k / (2H))`` with
    ``alpha(0) = 1/sqrt(2)`` and ``alpha(k>0) = 1``.  Rows are orthonormal, so
    ``C @ C.T = I`` and the inverse DCT is simply ``C.T``.  Built in float64 for
    numerical accuracy; cast by the caller.
    """
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")
    k = torch.arange(H, dtype=dtype).view(H, 1)      # band index, rows
    h = torch.arange(H, dtype=dtype).view(1, H)      # time index, cols
    C = torch.cos(math.pi * (2.0 * h + 1.0) * k / (2.0 * H))
    C *= math.sqrt(2.0 / H)
    C[0, :] *= 1.0 / math.sqrt(2.0)                  # alpha(0)
    return C


def _skew(W: Tensor) -> Tensor:
    """Skew-symmetric part ``W - W.T`` (so ``S = -S.T``)."""
    return W - W.transpose(-1, -2)


def cayley_orthogonal(W: Tensor) -> Tensor:
    """Orthogonal matrix from a free matrix ``W`` via the Cayley transform.

    ``S = W - W.T`` is skew-symmetric, hence ``I + S`` is always invertible
    (its eigenvalues are ``1 + i*theta != 0``), and ``R = (I - S)(I + S)^-1`` is
    orthogonal with ``det R = +1``.  Differentiable in ``W``.
    """
    H = W.shape[-1]
    S = _skew(W)
    eye = torch.eye(H, dtype=W.dtype, device=W.device)
    # R = (I - S) (I + S)^{-1}  ==  solve((I + S)^T, (I - S)^T)^T, but direct solve is clearer:
    return torch.linalg.solve(eye + S, eye - S)


class SpectralTransform(nn.Module):
    """Orthogonal change of basis along the time axis of an action chunk.

    Parameters
    ----------
    H : int
        Temporal horizon (chunk length).
    rotation : {"none", "pca", "learned"}
        Extra orthogonal rotation applied *after* the DCT.  ``none`` -> ``R = I``.
        ``pca`` uses a fixed rotation supplied via :meth:`set_pca_rotation`
        (eigenvectors of the band covariance).  ``learned`` parameterises ``R``
        through the Cayley transform and trains it.
    identity : bool
        If True, ``C = I`` (no DCT); used by the ``raw_vib`` baseline.
    dtype : torch.dtype
        Working dtype of the buffers (default float32).
    """

    def __init__(
        self,
        H: int,
        rotation: str = "none",
        identity: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if rotation not in ("none", "pca", "learned"):
            raise ValueError(f"unknown rotation {rotation!r}")
        self.H = H
        self.rotation = rotation
        self.identity = identity

        if identity:
            C = torch.eye(H, dtype=torch.float64)
        else:
            C = dct_ii_matrix(H, dtype=torch.float64)
        self.register_buffer("C", C.to(dtype))

        # Fixed rotation buffer (I for none/learned; set externally for pca).
        self.register_buffer("R_fixed", torch.eye(H, dtype=dtype))
        if rotation == "learned":
            # Free parameter; R = cayley(W). Initialise at 0 -> R = I.
            self.W = nn.Parameter(torch.zeros(H, H, dtype=dtype))
        else:
            self.W = None

    # ------------------------------------------------------------------ utils
    def set_pca_rotation(self, R: Tensor) -> None:
        """Install a fixed ``(H, H)`` orthogonal rotation (used by ``pca``)."""
        if self.rotation != "pca":
            raise RuntimeError("set_pca_rotation only valid when rotation='pca'")
        R = R.to(self.R_fixed)
        if R.shape != (self.H, self.H):
            raise ValueError(f"R must be ({self.H},{self.H}), got {tuple(R.shape)}")
        self.R_fixed.copy_(R)

    def rotation_matrix(self) -> Tensor:
        if self.rotation == "learned":
            return cayley_orthogonal(self.W)
        return self.R_fixed

    def analysis_matrix(self) -> Tensor:
        """The full orthogonal analysis operator ``T = R @ C`` (shape ``(H, H)``)."""
        return self.rotation_matrix() @ self.C

    # --------------------------------------------------------------- transforms
    def analyze(self, A: Tensor) -> Tensor:
        """Forward transform ``A (B,H,d) -> X (B,H,d)`` along the time axis."""
        T = self.analysis_matrix().to(A.dtype)
        return torch.einsum("kh, bhd -> bkd", T, A)

    def synthesize(self, X: Tensor) -> Tensor:
        """Inverse transform ``X (B,H,d) -> A (B,H,d)``; exact left inverse of analyze."""
        T = self.analysis_matrix().to(X.dtype)
        return torch.einsum("hk, bkd -> bhd", T.transpose(0, 1), X)

    # alias matching the spec's prose
    forward = analyze
