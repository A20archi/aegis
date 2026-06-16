"""AEGIS injection for NanoVLA-S (language-conditioned ACT) — cross-architecture port.

This mirrors the SmolVLA AEGIS stack onto the ACT backbone WITHOUT touching either
module's internals — RIB and RASF are architecture-agnostic operators, so we only
re-route where they attach:

  * RIB (perception leg, sib/robust_ib.py::RobustIB) wraps ``act.encoder_img_feat_input_proj``
    — the 1x1 conv ``W_img`` that projects ResNet feature maps into the encoder
    (the FRONT point). The conv output (B, C, h, w) is tokenized to (B, h*w, C) and
    passed through the SAME RobustIB residual used for SmolVLA (here D = dim_model = 512).
    Identity-initialised (zero-init decoder + residual gate) so clean behaviour is
    unchanged at step 0 — clean SR cannot degrade.
  * RASF (action leg, sib/adaptive_filter.py::AdaptiveSpectralFilter) post-filters the
    action chunk A (B, H, d_act) emerging from the action head (the BACK point), again
    identity at init (master gate = 0 -> A_hat == A).

Nothing in ACT is modified: we swap one conv module and post-filter the chunk. RIB at
d_z=256 is ~0.65M params (smaller model -> smaller bottleneck); RASF a few-k. Both train
by the SAME recipes as SmolVLA — RIB by corruption-augmented consistency, RASF by
clean-self-prediction denoising — so the only architecture-specific code is here.

Temporal ensembling, the third AEGIS leg, is NATIVE to ACT (ACTConfig.temporal_ensemble_coeff
+ ACTTemporalEnsembler), so it is configured at eval time rather than re-implemented here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sib.robust_ib import RobustIB
from sib.adaptive_filter import AdaptiveSpectralFilter


class FusedRIBConv(nn.Module):
    """Wrap a 1x1 Conv2d (W_img) with an identity-initialised RIB residual over tokens.

        feat = conv(x)                          # (B, C, h, w)
        tok  = feat.flatten(2).transpose(1, 2)  # (B, h*w, C)  -> RIB's (B, N, D) interface
        tok  = tok + tanh(fusion_coeff) * RIB(tok)
        return tok.transpose(1, 2).reshape(B, C, h, w)

    fusion_coeff init positive (tanh(.5493)=0.5) so the decoder receives gradient from
    step 1 (avoids the dead-start dormancy), while RIB's zero-init decoder makes the
    correction 0 at init -> output == conv(x) exactly (clean-safe identity).
    """

    def __init__(self, conv: nn.Conv2d, d_z: int = 256, n_heads: int = 8) -> None:
        super().__init__()
        self.conv = conv
        D = conv.out_channels
        self.rib = RobustIB(D, d_z=d_z, n_heads=n_heads)
        self.fusion_coeff = nn.Parameter(torch.full((1,), 0.5493))

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv(x)
        B, C, h, w = feat.shape
        tok = feat.flatten(2).transpose(1, 2)               # (B, N, C)
        tok = tok + torch.tanh(self.fusion_coeff) * self.rib(tok)
        return tok.transpose(1, 2).reshape(B, C, h, w)

    @property
    def ib_contribution(self) -> float:
        return float(torch.tanh(self.fusion_coeff).item())

    @property
    def last_kl(self) -> Tensor:
        return self.rib.last_kl

    def n_params(self) -> int:
        return self.rib.n_params() + self.fusion_coeff.numel()


def inject_fused_rib_conv(nanovla, d_z: int = 256, n_heads: int = 8) -> FusedRIBConv:
    """Replace NanoVLA's ACT image-projection conv with a FusedRIBConv in-place."""
    act = nanovla.act
    conv = act.encoder_img_feat_input_proj
    assert isinstance(conv, nn.Conv2d), \
        f"expected encoder_img_feat_input_proj to be Conv2d, got {type(conv)}"
    fused = FusedRIBConv(conv, d_z=d_z, n_heads=n_heads)
    ref = next(conv.parameters())
    fused.rib.to(device=ref.device, dtype=ref.dtype)
    fused.fusion_coeff.data = fused.fusion_coeff.data.to(device=ref.device, dtype=ref.dtype)
    act.encoder_img_feat_input_proj = fused
    print(f"[aegis-nano] injected RIB into encoder_img_feat_input_proj  "
          f"params={fused.rib.n_params()/1e6:.2f}M  d_z={d_z}  C={conv.out_channels}  "
          f"fusion_coeff init={fused.ib_contribution:.3f} (vanilla at init)", flush=True)
    return fused


class AEGISNanoVLA(nn.Module):
    """NanoVLA-S + AEGIS dual plug-in (RIB in the image conv, RASF on the action chunk).

    Wraps a built :class:`nanovla.modeling_nanovla.NanoVLAS`. RIB is injected in-place
    into the wrapped net (active automatically on every forward); RASF post-filters the
    predicted chunk at inference. The training forward returns the net's own (loss, parts)
    with RIB live — RASF is trained separately by clean-self-prediction (train_rasf-style),
    so it is NOT inserted into the training loss path here.
    """

    def __init__(self, nanovla, *, rib_d_z: int = 256, rib_heads: int = 8,
                 rasf_gate_max: float = 0.95, rasf_gain_floor: float = 0.05) -> None:
        super().__init__()
        self.net = nanovla
        self.config = nanovla.config
        self.rib = inject_fused_rib_conv(nanovla, d_z=rib_d_z, n_heads=rib_heads)
        H = nanovla.config.chunk_size
        d = nanovla.config.action_feature.shape[0]
        self.rasf = AdaptiveSpectralFilter(H, d, gate_max=rasf_gate_max,
                                           gain_floor=rasf_gain_floor)
        ref = next(nanovla.act.parameters())
        self.rasf.to(device=ref.device, dtype=ref.dtype)
        print(f"[aegis-nano] attached RASF  H={H} d={d}  gate_max={rasf_gate_max} "
              f"gain_floor={rasf_gain_floor}  gate_strength={self.rasf.gate_strength:.3f} "
              f"(identity at init)", flush=True)

    # --- training: RIB live in the conv; RASF trained separately -------------
    def forward(self, batch: dict):
        return self.net.forward(batch)

    # --- inference: RIB live + RASF denoise on the chunk ---------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> Tensor:
        A = self.net.predict_action_chunk(batch)            # (B, H, d), RIB active
        A_hat = self.rasf(A.float())["A_hat"].to(A.dtype)
        return A_hat

    @torch.no_grad()
    def select_action(self, batch: dict) -> Tensor:
        return self.predict_action_chunk(batch)[:, 0]
