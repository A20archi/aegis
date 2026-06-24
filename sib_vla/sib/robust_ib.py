"""RIB — Robust Information Bottleneck (perception-locus leg of AEGIS).

The StableVLA IB-Adapter (sib/ib_adapter.py) inserts a channel-covariance sigmoid
gate at the vision->LLM projection and is trained by the clean task loss alone. With
no corruption to defend against and no rate objective, its fusion coefficient stays
at ~0 (dormant) — it never engages (measured: coeff = -0.006 after 10k steps).

RIB fixes this the same way the conservative RASF fixed the action leg:

  * Same insertion point as StableVLA (connector modality_projection.proj output,
    D=960 LLM-embedding space), fused residually and IDENTITY-INITIALISED:
        out = Z_mlp + tanh(fusion_coeff) * decoder(z),  fusion_coeff=0, decoder=0-init
    so at init it is exactly vanilla SmolVLA and clean SR cannot degrade.
  * An EXPLICIT variational rate term  beta * KL(q(z|x) || N(0,I))  — the information
    bottleneck objective StableVLA lacks (its gate is heuristic).
  * Trained by CORRUPTION-AUGMENTED CONSISTENCY (see scripts/finetune_rib.py): the
    input image is corrupted (blur/noise/lighting/crop/perspective) and RIB must
    recover the GT action — so the bottleneck learns to keep task-relevant info and
    drop the corruption-induced nuisance subspace = visual robustness.

Capacity (default d_z=448): ~2.3M params, within the <3M budget. Layout:
    Z_mlp(960) -> enc(960->d_z) -> 1x token self-attention (spatial corruption context)
              -> VIB latent (mu, logvar; rate=KL) -> decoder(d_z->960) -> residual fuse
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RobustIB(nn.Module):
    """Variational information bottleneck on the visual connector features.

    Parameters
    ----------
    D : feature dim (960 for SmolVLA connector output).
    d_z : bottleneck working/latent dim (capacity knob; 448 ~= 2.3M params).
    n_heads : attention heads for the token-mixing block.
    """

    def __init__(self, D: int = 960, d_z: int = 448, n_heads: int = 7) -> None:
        super().__init__()
        self.D, self.d_z = D, d_z
        self.enc = nn.Linear(D, d_z)
        self.enc_ln = nn.LayerNorm(d_z)
        # token-mixing: gives spatial context so localized corruption is detectable
        self.attn = nn.MultiheadAttention(d_z, n_heads, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_z)
        # variational bottleneck
        self.to_z = nn.Linear(d_z, d_z)          # deterministic latent projection
        # decoder back to feature space
        self.dec = nn.Sequential(nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, D))
        # zero-init the decoder output so Z_ib starts at 0 (extra identity safety)
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)
        self.last_kl: Tensor | None = None   # stashed each forward for the rate loss

    def forward(self, X: Tensor) -> Tensor:
        """X: (B,N,D) connector features -> corrected (B,N,D). Stores KL in self.last_kl."""
        squeeze = X.dim() == 2
        if squeeze:
            X = X.unsqueeze(0)
        # encode (compute in fp32 for stable KL / sampling, cast back at the end)
        in_dtype = X.dtype
        h = F.gelu(self.enc_ln(self.enc(X.float())))
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_ln(h + a)
        z = self.to_z(h)                             # DETERMINISTIC latent (no sampling)
        Z_ib = self.dec(z)
        # explicit rate term: L2 on the latent = deterministic information-bottleneck
        # compression proxy. Drives the encoder to drop nuisance/corruption-sensitive
        # dims (invariance) without the VIB posterior-collapse pathology that sampling
        # induced (task loss avoided the injected noise by ignoring the latent).
        self.last_kl = z.pow(2).mean()
        out = Z_ib.to(in_dtype)
        if squeeze:
            out = out.squeeze(0)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FusedRobustIBProjector(nn.Module):
    """Wrap SmolVLA's connector linear with an identity-initialised RIB residual.

        out = linear(x) + tanh(fusion_coeff) * RIB(linear(x))

    fusion_coeff=0 and RIB decoder zero-init  =>  out == linear(x) at init (vanilla).
    """

    def __init__(self, original_linear: nn.Linear, D_out: int = 960,
                 d_z: int = 448, n_heads: int = 7, gate_hidden: int = 128) -> None:
        super().__init__()
        self.linear = original_linear
        self.rib = RobustIB(D_out, d_z=d_z, n_heads=n_heads)
        # IMPORTANT: init fusion_coeff POSITIVE (tanh(0.549)=0.5), NOT zero. The decoder
        # is zero-init so Z_ib=0 at step 0 -> output == linear(x) (exact identity, clean
        # SR safe). But a zero fusion_coeff would also zero the decoder's gradient
        # (grad ∝ tanh(coeff)*dZ_ib), a DEAD-START that keeps RIB dormant forever
        # (StableVLA's failure). A positive coeff lets the decoder receive gradient and
        # learn the correction from step 1, while identity-at-init still holds via Z_ib=0.
        self.fusion_coeff = nn.Parameter(torch.full((1,), 0.5493))
        # INPUT-ADAPTIVE GATE (AEGIS v2): a per-sample scalar g(x) in (0,1) that scales
        # the RIB residual, so RIB can BACK OFF on inputs where its bottleneck doesn't
        # help — the high-baseline axes (lighting / layout / robot-init) where the frozen
        # base is already strong and RIB's information loss was pure downside. Trained
        # (scripts/finetune_rib.py) to OPEN on corrupted inputs, CLOSE on clean ones.
        # Closed gate (g->0) zeroes the residual => connector == original linear == base,
        # exactly (the deployable, per-input version of "close the gate to recover base").
        self.gate_head = nn.Sequential(nn.Linear(D_out, gate_hidden), nn.GELU(),
                                       nn.Linear(gate_hidden, 1))
        # zero-weight + positive-bias init: gate starts ~sigmoid(2)=0.88 (mostly open,
        # input-independent) so the decoder keeps its gradient; input-dependence is learned.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 2.0)
        self.last_gate: Tensor | None = None   # per-sample gate, stashed for the gate loss
        self.gate_trace: list = []             # per-view gate means (one entry per forward call)
        self.zmlp_trace: list = []             # per-view PRE-RIB linear output (clean-target src)
        self.out_trace: list = []              # per-view RIB output (denoising prediction)
        self.gate_sample_trace: list = []      # per-view per-sample gate g (B,), grad-carrying (BCE)
        self.gate_raw_trace: list = []         # per-view per-sample raw sigmoid (B,), pre-floor (observ.)
        self.gate_logit_trace: list = []       # per-view per-sample gate LOGIT (B,), for BCE-with-logits
        # ADVANTAGE-GATE PROBE: when set to a float, the gate is FORCED to that constant
        # (0.0 = baseline path, 1.0 = full RIB) regardless of gate_head/gate_floor. Used by
        # finetune_rib.py to measure, per sample, whether RIB BEATS the baseline on THIS input —
        # the gate is then trained to open only where it helps and close (=> emit baseline) else.
        self.gate_override: float | None = None
        # effective gate = gate_floor + (1-gate_floor)*sigmoid(.). gate_floor=1.0 => always-on
        # (proven RIB, decoder always receives gradient, no dead-start). Lower it (<1) to
        # re-enable adaptive per-input back-off — but only with a decoder warmup, else the gate
        # shuts before the zero-init decoder learns and freezes RIB. v3 territory.
        self.gate_floor: float = 1.0

    def forward(self, x: Tensor) -> Tensor:
        z_mlp = self.linear(x)
        z_rib = self.rib(z_mlp)
        is3d = z_mlp.dim() == 3
        B = z_mlp.shape[0] if is3d else 1
        if self.gate_override is not None:
            # PROBE PATH: force a constant gate (0=baseline, 1=full RIB), bypass gate_head/floor.
            ov = float(self.gate_override)
            g = z_mlp.new_full((B, 1, 1) if is3d else (1, 1), ov)
            self.last_gate = z_mlp.new_full((B,) if is3d else (1,), ov)
        else:
            # per-sample gate from pooled connector features (token-mean if 3D)
            if is3d:                                            # (B, N, D)
                pooled = z_mlp.float().mean(dim=1)              # (B, D)
                logit = self.gate_head(pooled).squeeze(-1)     # (B,)
                raw = torch.sigmoid(logit)                     # (B,)
                gvec = self.gate_floor + (1.0 - self.gate_floor) * raw    # (B,)
                self.last_gate = gvec
                g = gvec.view(B, 1, 1)                          # broadcast
            else:                                              # (N, D) single sample
                pooled = z_mlp.float().mean(dim=0, keepdim=True)          # (1, D)
                logit = self.gate_head(pooled).squeeze(-1)     # (1,)
                raw = torch.sigmoid(logit)                     # (1,)
                gvec = self.gate_floor + (1.0 - self.gate_floor) * raw    # (1,)
                self.last_gate = gvec
                g = gvec.view(1, 1)
            self.gate_logit_trace.append(logit)     # pre-sigmoid logit (BCE-with-logits target)
            self.gate_raw_trace.append(raw)         # pre-floor sigmoid (== deployed gate at floor=0)
            self.gate_sample_trace.append(gvec)     # per-sample effective gate (grad-carrying)
        self.gate_trace.append(self.last_gate.mean().detach())  # per-view observability
        g = g.to(z_rib.dtype)
        out = z_mlp + torch.tanh(self.fusion_coeff) * g * z_rib
        self.zmlp_trace.append(z_mlp)   # pre-RIB linear output (clean-feature target source)
        self.out_trace.append(out)      # RIB output (denoising prediction, for consistency loss)
        return out

    @property
    def ib_contribution(self) -> float:
        return float(torch.tanh(self.fusion_coeff).item())

    @property
    def last_kl(self) -> Tensor:
        return self.rib.last_kl


def inject_fused_rib(policy, D_out: int = 960, d_z: int = 448, n_heads: int = 7) -> FusedRobustIBProjector:
    """Replace SmolVLA's connector linear with a FusedRobustIBProjector in-place."""
    conn = policy.model.vlm_with_expert.vlm.model.connector
    original_linear = conn.modality_projection.proj
    fused = FusedRobustIBProjector(original_linear, D_out=D_out, d_z=d_z, n_heads=n_heads)
    ref = next(original_linear.parameters())
    fused.rib.to(device=ref.device, dtype=ref.dtype)
    fused.gate_head.to(device=ref.device, dtype=ref.dtype)
    fused.fusion_coeff.data = fused.fusion_coeff.data.to(device=ref.device, dtype=ref.dtype)
    conn.modality_projection.proj = fused
    print(f"[rib] injected RobustIB  params={fused.rib.n_params()/1e6:.2f}M  d_z={d_z}  "
          f"fusion_coeff init={fused.ib_contribution:.3f} (vanilla at init)", flush=True)
    return fused


def save_rib_checkpoint(fused: FusedRobustIBProjector, lm_expert, out_path: str,
                        config: dict, step: int, loss: float) -> None:
    torch.save({
        "rib_state": fused.rib.state_dict(),
        "gate_head_state": fused.gate_head.state_dict(),
        "fusion_coeff": fused.fusion_coeff.data.clone(),
        "lm_expert_state": lm_expert.state_dict(),
        "config": config, "step": step, "loss": loss,
    }, out_path)


def load_rib_checkpoint(policy, weights_path: str, device: str = "cuda") -> None:
    """Load a saved RIB checkpoint into a policy already patched with inject_fused_rib()."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    conn = policy.model.vlm_with_expert.vlm.model.connector
    fused = conn.modality_projection.proj
    assert isinstance(fused, FusedRobustIBProjector), \
        "Call inject_fused_rib(policy) before load_rib_checkpoint()"
    fused.rib.load_state_dict(ckpt["rib_state"])
    if "gate_head_state" in ckpt:                       # AEGIS v2 adaptive gate
        fused.gate_head.load_state_dict(ckpt["gate_head_state"])
    ref = next(fused.linear.parameters())
    fused.rib.to(device=ref.device, dtype=ref.dtype)
    fused.gate_head.to(device=ref.device, dtype=ref.dtype)
    fused.fusion_coeff.data = ckpt["fusion_coeff"].to(device=ref.device, dtype=ref.dtype)
    policy.model.vlm_with_expert.lm_expert.load_state_dict(ckpt["lm_expert_state"])
    print(f"[rib] loaded {weights_path}  fusion_coeff={fused.ib_contribution:.3f}  "
          f"step={ckpt.get('step','?')}", flush=True)
