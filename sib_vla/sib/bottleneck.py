"""Per-band coefficient operators and the composed spectral action module.

The spectral information bottleneck factorises into three independent pieces:

    A  --analyze-->  X  --operator-->  X_hat  --synthesize-->  A_hat

The *operator* is the only part that differs across the experiment matrix, and
all six configs are covered by three operators + the choice of transform:

    config         transform   operator          regulariser (in losses.py)
    -----------    ---------   --------------     --------------------------
    sib            DCT(+rot)   GaussianChannel    rate  (beta * R)
    raw_vib        identity    GaussianChannel    rate  (beta * R)
    lowpass        DCT         LowpassMask        none
    gain_no_rate   DCT         LearnedGain        none  (beta = 0)
    jerk           DCT         LearnedGain        jerk  (gamma * J)

This is the minimal structure that still expresses every ablation, which is the
point: ``raw_vib`` vs ``sib`` is *only* the transform; ``jerk`` vs
``gain_no_rate`` is *only* the regulariser.  Nothing else moves.

----------------------------------------------------------------------------
Rate units.  The spec pins the rate term verbatim as

    R = 0.5 * log1p(lambda / sigma2).sum()          # natural log -> nats

``log1p`` is the natural logarithm, so the optimised quantity is in **nats**.
The loss is ``mse + beta * R``; multiplying R by any constant (e.g. converting
nats->bits) is absorbed by ``beta``, so the choice is irrelevant to the
optimum -- but it matters for *reporting*.  We therefore keep the loss in nats
(``rate_nats``) and report the per-band allocation heatmap in **bits**
(``bits_per_band``, base-2), clearly labelled.  ``R`` is documented as the
mutual information ``I(X; X_tilde)`` of the Gaussian channel, i.e. a variational
*upper bound* on the true coding rate -- never a closed-form optimum.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .lambda_estimator import LambdaEstimator
from .transforms import SpectralTransform

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Rate (information) functionals -- free functions so tests can probe limits
# directly without routing through softplus.
# ---------------------------------------------------------------------------
def rate_nats(lam: Tensor, sigma2: Tensor) -> Tensor:
    """Per-element channel rate ``0.5 * ln(1 + lambda/sigma2)`` in nats.

    Equals ``I(X; X_tilde)`` for ``X ~ N(0, lambda)`` through an additive
    Gaussian channel of noise variance ``sigma2``.  Monotonically decreasing in
    ``sigma2``; ``-> +inf`` as ``sigma2 -> 0`` and ``-> 0`` as ``sigma2 -> inf``.
    """
    return 0.5 * torch.log1p(lam / sigma2)


def bits_per_band(lam: Tensor, sigma2: Tensor) -> Tensor:
    """Per-element rate in **bits** (base-2), for reporting/heatmaps."""
    return rate_nats(lam, sigma2) / LN2


def wiener_gain(lam: Tensor, sigma2: Tensor) -> Tensor:
    """MMSE / Wiener gain ``lambda / (lambda + sigma2)`` in ``[0, 1]``.

    This is the correct estimator gain, *not* ``1 - theta/lambda``.
    """
    return lam / (lam + sigma2)


def _reduce_rate(rate_el: Tensor) -> Tensor:
    """Scalar rate: sum over (band, dim); mean over batch if sigma2 is per-sample."""
    if rate_el.dim() == 2:                       # (H, d) global sigma2
        return rate_el.sum()
    if rate_el.dim() == 3:                        # (B, H, d) context sigma2
        return rate_el.sum(dim=(1, 2)).mean()
    raise ValueError(f"unexpected rate shape {tuple(rate_el.shape)}")


# ---------------------------------------------------------------------------
# Operators.  Common interface:  __call__(X, lam, context) -> (X_hat, info)
#   info: dict possibly containing 'rate' (scalar nats), 'gain', 'sigma2'.
# ---------------------------------------------------------------------------
class GaussianChannel(nn.Module):
    """Per-band additive-Gaussian channel with learned noise and Wiener decode.

    ``log_var`` is learned (shape ``(H, d)``); ``sigma2 = softplus(log_var)+eps``.
    In ``sigma_mode='context'`` a small head adds a per-sample perturbation
    ``(B, H, d)`` predicted from a context embedding, so the bit allocation can
    adapt to the observation.

    Train: reparameterised stochastic channel ``X_hat = gain * (X + noise)``.
    Eval : ``decode='mmse'`` -> ``X_hat = gain * X`` (drop the noise);
           ``decode='stochastic'`` -> keep injecting noise (ablation).
    """

    def __init__(
        self,
        H: int,
        d: int,
        sigma_mode: str = "global",
        decode: str = "mmse",
        eps: float = 1e-6,
        context_dim: Optional[int] = None,
        context_hidden: int = 128,
        init_log_var: float = -2.0,
        adaptive_sigma: bool = False,
        noise_band_frac: float = 0.5,
    ) -> None:
        super().__init__()
        if sigma_mode not in ("global", "context"):
            raise ValueError(f"unknown sigma_mode {sigma_mode!r}")
        if decode not in ("mmse", "stochastic"):
            raise ValueError(f"unknown decode {decode!r}")
        self.H, self.d = H, d
        self.sigma_mode = sigma_mode
        self.decode = decode
        self.eps = eps

        self.log_var = nn.Parameter(torch.full((H, d), float(init_log_var)))
        # Input-adaptive noise floor: estimate per-(sample,dim) noise variance from
        # the signal-free high bands (k >= noise_k0, where clean lambda ~ 0 so the
        # observed energy IS the noise) and ADD it to the base sigma2. Clean input ->
        # tiny term -> base smoothing; noisy input -> large term -> suppress harder.
        # This is the lever that lets the learned bottleneck beat a fixed low-pass
        # under action noise. adapt_log_scale is learned (softplus -> >=0).
        self.adaptive_sigma = adaptive_sigma
        self.noise_k0 = max(1, int(round(H * noise_band_frac)))
        self.adapt_log_scale = nn.Parameter(torch.zeros(d)) if adaptive_sigma else None
        if sigma_mode == "context":
            if context_dim is None:
                raise ValueError("sigma_mode='context' requires context_dim")
            self.ctx_head = nn.Sequential(
                nn.Linear(context_dim, context_hidden),
                nn.SiLU(),
                nn.Linear(context_hidden, H * d),
            )
            # start as a no-op perturbation
            nn.init.zeros_(self.ctx_head[-1].weight)
            nn.init.zeros_(self.ctx_head[-1].bias)
        else:
            self.ctx_head = None

    def sigma2(self, batch: int, context: Optional[Tensor]) -> Tensor:
        log_var = self.log_var                                   # (H, d)
        if self.sigma_mode == "context" and context is not None:
            delta = self.ctx_head(context).view(-1, self.H, self.d)
            log_var = log_var.unsqueeze(0) + delta               # (B, H, d)
        return F.softplus(log_var) + self.eps

    def forward(self, X: Tensor, lam: Tensor, context: Optional[Tensor] = None):
        sigma2 = self.sigma2(X.shape[0], context)                # (H,d) or (B,H,d)
        if self.adaptive_sigma:
            noise_est = X[:, self.noise_k0:, :].pow(2).mean(dim=1, keepdim=True)  # (B,1,d)
            sigma2 = sigma2 + F.softplus(self.adapt_log_scale) * noise_est        # -> (B,H,d)
        gain = wiener_gain(lam, sigma2)
        inject = self.training or (self.decode == "stochastic")
        if inject:
            noise = torch.randn_like(X) * sigma2.sqrt()          # reparameterised
            X_hat = gain * (X + noise)
        else:
            X_hat = gain * X                                     # MMSE
        R = _reduce_rate(rate_nats(lam, sigma2))
        info = {"rate": R, "gain": gain, "sigma2": sigma2}
        return X_hat, info

    @torch.no_grad()
    def bits_per_band(self, lam: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Per-(band, dim) rate in bits, shape ``(H, d)`` (batch-averaged)."""
        batch = 1 if context is None else context.shape[0]
        sigma2 = self.sigma2(batch, context)
        b = bits_per_band(lam, sigma2)
        return b.mean(dim=0) if b.dim() == 3 else b


class LearnedGain(nn.Module):
    """Free per-band gain in ``[0, 1]`` -- no channel, no rate (gain_no_rate / jerk).

    The gain is a raw parameter clamped to ``[0, 1]`` (spec wording); initialised
    at 1.0 so the module starts as exact reconstruction and descends.
    """

    def __init__(self, H: int, d: int) -> None:
        super().__init__()
        self.gain_param = nn.Parameter(torch.ones(H, d))

    def gain(self) -> Tensor:
        return self.gain_param.clamp(0.0, 1.0)

    def forward(self, X: Tensor, lam: Optional[Tensor] = None, context=None):
        g = self.gain()
        return g * X, {"rate": None, "gain": g, "sigma2": None}


class LowpassMask(nn.Module):
    """Hard DCT truncation: keep bands ``k < cutoff``, zero the rest.

    No parameters; ``cutoff`` is grid-searched at the config level.
    """

    def __init__(self, H: int, d: int, cutoff: int) -> None:
        super().__init__()
        if not 1 <= cutoff <= H:
            raise ValueError(f"cutoff must be in [1, H={H}], got {cutoff}")
        self.cutoff = cutoff
        mask = torch.zeros(H, 1)
        mask[:cutoff] = 1.0
        self.register_buffer("mask", mask)                       # (H, 1)

    def forward(self, X: Tensor, lam: Optional[Tensor] = None, context=None):
        g = self.mask.to(X.dtype)
        return g * X, {"rate": None, "gain": g.expand(self.mask.shape[0], X.shape[-1]),
                       "sigma2": None}


# ---------------------------------------------------------------------------
# Composed module: transform -> operator -> inverse transform.
# ---------------------------------------------------------------------------
_GAUSSIAN = ("sib", "raw_vib")


class SpectralActionModule(nn.Module):
    """Wraps an action chunk ``A (B,H,d)`` through the spectral bottleneck.

    Returns a dict with ``A_hat`` (reconstructed chunk) and the regulariser
    ingredients (``rate``, ``gain``, ``sigma2``, plus ``X``/``X_hat`` for
    diagnostics).  The frozen policy lives in :class:`sib.wrapper.SIBPolicy`;
    this module is the only trainable part.
    """

    def __init__(
        self,
        module_type: str,
        H: int,
        d: int,
        *,
        rotation: str = "none",
        sigma_mode: str = "global",
        decode: str = "mmse",
        cutoff: Optional[int] = None,
        context_dim: Optional[int] = None,
        lambda_momentum: float = 0.99,
        eps: float = 1e-6,
        preserve_energy: bool = False,
        adaptive_sigma: bool = False,
        noise_band_frac: float = 0.5,
    ) -> None:
        super().__init__()
        if module_type not in ("sib", "raw_vib", "lowpass", "gain_no_rate", "jerk"):
            raise ValueError(f"unknown module_type {module_type!r}")
        self.module_type = module_type
        self.H, self.d = H, d
        self.eps = eps
        # Magnitude-preserving variant: the Wiener/MMSE gain is <=1 and shrinks the
        # chunk (optimal under MSE, but it under-actuates the robot -- fatal at low
        # n_action_steps). When set, rescale A_hat per (sample, action-dim) so its
        # over-time RMS matches the input: keeps the learned spectral SHAPE (the
        # smoothing) but restores the action magnitude needed for control.
        self.preserve_energy = preserve_energy

        identity = module_type == "raw_vib"
        self.transform = SpectralTransform(H, rotation=rotation, identity=identity)

        if module_type in _GAUSSIAN:
            self.operator = GaussianChannel(
                H, d, sigma_mode=sigma_mode, decode=decode,
                eps=eps, context_dim=context_dim,
                adaptive_sigma=adaptive_sigma, noise_band_frac=noise_band_frac,
            )
            self.lambda_estimator = LambdaEstimator(H, d, momentum=lambda_momentum)
            self.uses_rate = True
        elif module_type == "lowpass":
            if cutoff is None:
                raise ValueError("lowpass requires cutoff")
            self.operator = LowpassMask(H, d, cutoff)
            self.lambda_estimator = None
            self.uses_rate = False
        else:  # gain_no_rate, jerk
            self.operator = LearnedGain(H, d)
            self.lambda_estimator = None
            self.uses_rate = False

    def forward(
        self,
        A: Tensor,
        context: Optional[Tensor] = None,
        update_lambda: Optional[bool] = None,
    ) -> dict:
        X = self.transform.analyze(A)
        lam = None
        if self.lambda_estimator is not None:
            if update_lambda is None:
                update_lambda = self.training
            if update_lambda:
                self.lambda_estimator.update(X)
            lam = self.lambda_estimator.value
        X_hat, info = self.operator(X, lam, context)
        A_hat = self.transform.synthesize(X_hat)
        if self.preserve_energy:
            rms_in = A.pow(2).mean(dim=-2, keepdim=True).clamp_min(self.eps).sqrt()
            rms_out = A_hat.pow(2).mean(dim=-2, keepdim=True).clamp_min(self.eps).sqrt()
            A_hat = A_hat * (rms_in / rms_out)
        return {"A_hat": A_hat, "X": X, "X_hat": X_hat, "lam": lam, **info}

    @torch.no_grad()
    def bits_per_band(self, context: Optional[Tensor] = None) -> Optional[Tensor]:
        """Per-(band, dim) bit allocation ``(H, d)``; None for non-rate modules."""
        if not self.uses_rate:
            return None
        return self.operator.bits_per_band(self.lambda_estimator.value, context)


def build_module(cfg: dict, H: int, d: int):
    """Construct the bottleneck module from a resolved config dict.

    ``module: rasf`` selects the redesigned Residual Adaptive Spectral Filter
    (identity-initialised, denoising-trained); everything else builds the original
    :class:`SpectralActionModule`.
    """
    if cfg.get("module") == "rasf":
        from sib.adaptive_filter import build_rasf
        return build_rasf(cfg, H, d)
    return SpectralActionModule(
        module_type=cfg["module"],
        H=H,
        d=d,
        rotation=cfg.get("rotation", "none"),
        sigma_mode=cfg.get("sigma_mode", "global"),
        decode=cfg.get("decode", "mmse"),
        cutoff=cfg.get("cutoff"),
        context_dim=cfg.get("context_dim"),
        lambda_momentum=cfg.get("lambda_momentum", 0.99),
        eps=cfg.get("sigma_eps", 1e-6),
        preserve_energy=cfg.get("preserve_energy", False),
        adaptive_sigma=cfg.get("adaptive_sigma", False),
        noise_band_frac=cfg.get("noise_band_frac", 0.5),
    )
