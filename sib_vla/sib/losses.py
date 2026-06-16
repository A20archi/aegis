"""Training objectives.

The distortion target is always the **ground-truth action chunk** ``A_star`` (in
the policy's normalised action space), never the model's own output -- this lets
the module *denoise* the frozen policy, not merely compress it.

    L = mse(A_hat, A_star) + reg

where ``reg`` is the only thing that varies across configs:

    sib / raw_vib : beta  * R          (R = channel rate, nats; from the module)
    jerk          : gamma * J          (J = mean squared third difference of A_hat)
    gain_no_rate  : 0                  (beta = 0)
    lowpass       : 0                  (no trainable parameters)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def distortion_mse(A_hat: Tensor, A_star: Tensor) -> Tensor:
    """Mean squared error over the chunk (mean over B, H, d)."""
    if A_hat.shape != A_star.shape:
        raise ValueError(f"shape mismatch {tuple(A_hat.shape)} vs {tuple(A_star.shape)}")
    return F.mse_loss(A_hat, A_star, reduction="mean")


def third_difference(A: Tensor) -> Tensor:
    """Third finite difference along the time axis: ``(B, H, d) -> (B, H-3, d)``.

    Equivalent to applying ``diff`` three times; the discrete analogue of jerk
    (third derivative of position).  Empty if ``H < 4``.
    """
    return torch.diff(A, n=3, dim=1)


def jerk_penalty(A_hat: Tensor) -> Tensor:
    """Mean squared third difference of ``A_hat`` (smoothness penalty)."""
    td = third_difference(A_hat)
    if td.numel() == 0:
        return A_hat.new_zeros(())
    return td.pow(2).mean()


def total_loss(
    cfg: dict,
    out: dict,
    A_star: Tensor,
) -> tuple[Tensor, dict]:
    """Assemble the scalar loss and a dict of its (detached) parts for logging.

    ``out`` is the dict returned by :class:`sib.bottleneck.SpectralActionModule`.
    """
    A_hat = out["A_hat"]
    mse = distortion_mse(A_hat, A_star)
    loss = mse
    parts = {"mse": mse.detach()}

    module_type = cfg["module"]
    if module_type in ("sib", "raw_vib"):
        beta = float(cfg.get("beta", 0.0))
        R = out["rate"]
        if R is None:
            raise RuntimeError("gaussian channel produced no rate")
        loss = loss + beta * R
        parts["rate"] = R.detach()
        parts["beta"] = torch.as_tensor(beta)
    elif module_type == "jerk":
        gamma = float(cfg.get("gamma", 0.0))
        J = jerk_penalty(A_hat)
        loss = loss + gamma * J
        parts["jerk"] = J.detach()
        parts["gamma"] = torch.as_tensor(gamma)
    # gain_no_rate, lowpass: pure MSE (no extra term).

    parts["loss"] = loss.detach()
    return loss, parts
