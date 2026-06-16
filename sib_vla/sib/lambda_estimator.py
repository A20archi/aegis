"""Distributional source-variance estimator ``lambda_k``.

``lambda_k`` (shape ``(H, d)``) is an EMA estimate of the per-(band, dim)
variance of the *model's predicted* DCT coefficients ``X``.  It is the "source
variance" of the Gaussian rate-distortion problem: the Wiener gain and the rate
term in :mod:`sib.bottleneck` both read it.

Two invariants, both required by the spec and enforced here:

* **Distributional, not per-sample.**  ``lambda`` is an EMA over the dataset,
  computed from ``X.detach()``.  It is a buffer, never a function of the current
  sample's gradient.  (A per-sample variant is a separate, clearly labelled
  thing -- it is *not* this class.)
* **Detached everywhere it enters the loss.**  Being a buffer, it carries no
  grad; the rate term treats it as a constant.

Floored at ``eps`` so the Wiener gain ``lambda / (lambda + sigma2)`` stays well
defined even for dead bands (``lambda -> 0`` would otherwise pin the gain to 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LambdaEstimator(nn.Module):
    def __init__(
        self,
        H: int,
        d: int,
        momentum: float = 0.99,
        eps: float = 1e-8,
        init: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0,1), got {momentum}")
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("lam", torch.full((H, d), float(init)))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, X: Tensor) -> Tensor:
        """EMA update from a batch of coefficients ``X`` of shape ``(B, H, d)``.

        Skips degenerate batches (``B < 2``), whose unbiased=False variance is
        exactly 0 and would corrupt the running estimate.  Returns the current
        ``lambda``.
        """
        if X.dim() != 3:
            raise ValueError(f"expected (B,H,d), got {tuple(X.shape)}")
        if X.shape[0] < 2:
            return self.lam
        batch_var = X.detach().var(dim=0, unbiased=False)        # (H, d)
        batch_var = batch_var.to(self.lam)
        if not bool(self.initialized):
            self.lam.copy_(batch_var)
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.lam.mul_(m).add_(batch_var, alpha=1.0 - m)
        self.lam.clamp_(min=self.eps)
        return self.lam

    @torch.no_grad()
    def load_estimate(self, lam: Tensor) -> None:
        """Warm-start from a precomputed estimate (e.g. ``estimate_lambda.py``)."""
        lam = lam.to(self.lam).clamp(min=self.eps)
        if lam.shape != self.lam.shape:
            raise ValueError(f"shape {tuple(lam.shape)} != {tuple(self.lam.shape)}")
        self.lam.copy_(lam)
        self.initialized.fill_(True)

    @property
    def value(self) -> Tensor:
        """Current ``lambda`` (detached buffer, shape ``(H, d)``)."""
        return self.lam
