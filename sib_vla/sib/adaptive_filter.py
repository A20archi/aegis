"""RASF — Residual Adaptive Spectral Filter (SIB v2, *conservative denoiser* design).

History / why this exists
-------------------------
v1 (rate-distortion SIB): a lossy bottleneck — compresses the chunk, degrades even
clean actions, SR collapsed to ~15%.

v2-aggressive (denoise-toward-GT): trained the filter to match the *ground-truth
demonstrated* chunk on clean inputs. Diagnosed failure — it learned to crush the
spectrum on CLEAN inputs (mean band-gain g≈0.08, rms|A_hat−A|≈0.23, ~18% of action
energy removed) because the policy's good-but-different predictions were dragged
toward the single smooth demo. Clean SR collapsed (first LIBERO-Spatial tasks ~28%).

v2-conservative (THIS FILE): a pure *denoiser* whose fixed point on clean input is
the identity. Trained to reproduce the policy's OWN clean prediction (not the demo),
so it has zero incentive to touch clean actions and only learns to strip injected
perturbations. Structural caps make over-aggression impossible:

    A_hat = A + (gate_max · tanh(gate)) · ( IDCT( g(A) ⊙ DCT(A) ) − A )
            └──────── residual strength ≤ gate_max ────────┘

Design guarantees
-----------------
1. **Identity at init** — gate=0 ⇒ A_hat=A exactly.
2. **Bounded residual** — `gate_max` (default 0.6) caps how much of the spectral
   correction can ever be applied per dim, so the filter can never fully replace A.
3. **Gain floor** — every band keeps ≥ `gain_floor` (default 0.30) of its energy, so
   no band can be nulled (prevents the v2-aggressive g→0 collapse).
4. **Input-adaptive** — per-band gain g∈[floor,1] = base ⊙ sigmoid(MLP(log-energy)).
   On clean inputs band energies are nominal ⇒ g≈1 (all-pass). When a band carries
   anomalous (perturbation) energy the MLP pulls its gain down ⇒ denoise.
5. **Per-dim gate** — gripper (bang-bang) vs smooth pose dims handled independently.

Trained by train_rasf.py with target = the policy's clean prediction (see there).
Interface matches SpectralActionModule: forward(A) -> {"A_hat": ...}; uses_rate=False.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


def dct_matrix(N: int) -> Tensor:
    """Orthonormal DCT-II matrix M (N×N), rows = frequencies. IDCT = M.T."""
    n = torch.arange(N, dtype=torch.float64)
    k = n.view(-1, 1)
    M = torch.cos(math.pi / N * (n.view(1, -1) + 0.5) * k)
    M = M * math.sqrt(2.0 / N)
    M[0] = M[0] / math.sqrt(2.0)
    return M.to(torch.float32)


class AdaptiveSpectralFilter(nn.Module):
    """Residual, identity-initialised, input-adaptive spectral *denoiser* (bounded).

    Parameters
    ----------
    H, d : chunk horizon and action dim.
    hidden : width of the band-energy modulation MLP.
    gain_floor : minimum per-band gain (default 0.30) — a band can be attenuated to
        at most this fraction, never nulled. This is the main structural guard against
        the v2-aggressive collapse.
    base_gain_logit : init logit for the per-band base gain; large ⇒ g≈1 (all-pass).
    gate_max : hard cap on per-dim residual strength (default 0.6). The applied
        correction is gate_max·tanh(gate)·(Af−A), so |residual| ≤ gate_max·|Af−A|.
    """

    def __init__(self, H: int, d: int, *, hidden: int = 32,
                 gain_floor: float = 0.30, base_gain_logit: float = 4.0,
                 gate_max: float = 0.6) -> None:
        super().__init__()
        self.H, self.d = H, d
        self.gain_floor = float(gain_floor)
        self.gate_max = float(gate_max)
        self.uses_rate = False

        self.register_buffer("dct", dct_matrix(H))            # (H,H) freq×time
        self.register_buffer("idct", dct_matrix(H).t().contiguous())

        # Per-band base gain (init → all-pass).
        self.band_logit = nn.Parameter(torch.full((H,), float(base_gain_logit)))
        # Input-adaptive modulation from per-band log-energy (init → 0 contribution).
        self.mod = nn.Sequential(
            nn.Linear(H, hidden), nn.GELU(), nn.Linear(hidden, H),
        )
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        # Residual master gate, per action dim, init 0 → identity output.
        self.gate = nn.Parameter(torch.zeros(d))

    # --------------------------------------------------------------- forward
    def forward(self, A: Tensor, context=None, update_lambda=None) -> dict:
        """A: (B,H,d) normalised action chunk → {"A_hat": (B,H,d), ...}."""
        squeeze = A.dim() == 2
        if squeeze:
            A = A.unsqueeze(0)

        C = torch.einsum("kh,bhd->bkd", self.dct, A)          # (B,H,d) spectral
        e = (C.pow(2).mean(dim=-1) + 1e-8).log()              # (B,H) per-band log-energy
        logit = self.band_logit.unsqueeze(0) + self.mod(e)    # (B,H)
        g = torch.sigmoid(logit)                              # (B,H) ∈ (0,1)
        g = self.gain_floor + (1.0 - self.gain_floor) * g     # ∈ [floor,1]

        Cf = g.unsqueeze(-1) * C
        Af = torch.einsum("hk,bkd->bhd", self.idct, Cf)       # (B,H,d) filtered
        strength = self.gate_max * torch.tanh(self.gate)      # (d,), |·| ≤ gate_max, 0 at init
        A_hat = A + strength.view(1, 1, -1) * (Af - A)

        if squeeze:
            A_hat = A_hat.squeeze(0)
        return {"A_hat": A_hat, "gain": g, "gate": strength.view(1, 1, -1)}

    def bits_per_band(self, context=None):
        return None

    @property
    def gate_strength(self) -> float:
        """Mean applied residual strength gate_max·|tanh(gate)| (0 = identity)."""
        return float((self.gate_max * torch.tanh(self.gate).abs()).mean().item())


def build_rasf(cfg: dict, H: int, d: int) -> "AdaptiveSpectralFilter":
    return AdaptiveSpectralFilter(
        H, d,
        hidden=cfg.get("rasf_hidden", 32),
        gain_floor=cfg.get("rasf_gain_floor", 0.30),
        base_gain_logit=cfg.get("rasf_base_gain_logit", 4.0),
        gate_max=cfg.get("rasf_gate_max", 0.6),
    )
