"""Fused IB-Adapter for SmolVLA — channel-wise information bottleneck in the visual projector.

Based on StableVLA (Fu et al., 2026), adapted for SmolVLA's SmolVLMConnector.

Architecture:
    connector.forward(X_vis):
        X_ps = pixel_shuffle(X_vis)                    # [B, N, 12288]
        Z_mlp = linear(X_ps)                           # [B, N, 960]  ← existing Linear
        Z_ib  = ib_adapter(Z_mlp)                      # [B, N, 960]  ← new
        return Z_mlp + tanh(fusion_coeff) * Z_ib       # Fused output

IBAdapter operates at D=960 (LLM embedding dim) with n_heads=8 (d=120 per head):
    G_h = Q_h.T @ K_h              # Gram matrix [d, d], captures channel covariance
    A_h = sigmoid(G_h * τ_h)       # Sigmoid gate, suppresses uncorrelated noise
    V_h = Norm(GELU(X_h @ W_v1) @ W_v2)
    Z_h = V_h @ A_h                # gated feature reconstruction

Fusion coefficient initialized to 0: tanh(0)=0 → model starts as vanilla SmolVLA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IBAdapter(nn.Module):
    """Channel-wise Information Bottleneck Adapter.

    Operates on features X ∈ R^{B×N×D}, using multi-head Gram matrix attention
    with Sigmoid gating (independent Bernoulli latent structure).

    Parameters
    ----------
    D : int
        Feature dimension (960 for SmolVLA connector output).
    n_heads : int
        Number of attention heads; D must be divisible by n_heads.
    """

    def __init__(self, D: int, n_heads: int = 8) -> None:
        super().__init__()
        assert D % n_heads == 0, f"D={D} not divisible by n_heads={n_heads}"
        self.D = D
        self.n_heads = n_heads
        self.d = D // n_heads

        d = self.d
        # Per-head Q projection (identity key — no W_k, raw features used as K)
        self.W_q = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_heads)])
        # Per-head value: two-layer MLP with GELU + LayerNorm
        self.W_v1 = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_heads)])
        self.W_v2 = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_heads)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_heads)])
        # Learnable temperature τ_h per head (initialized to 1)
        self.tau = nn.Parameter(torch.ones(n_heads))

        self._init_weights()

    def _init_weights(self) -> None:
        for mod in [self.W_q, self.W_v1, self.W_v2]:
            for m in mod:
                nn.init.xavier_uniform_(m.weight)

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        X : Tensor, shape (B, N, D) or (N, D)
            Input visual features in LLM embedding space.

        Returns
        -------
        Tensor, same shape as X
        """
        batched = X.dim() == 3
        if not batched:
            X = X.unsqueeze(0)          # (1, N, D)

        B, N, D = X.shape
        # Split into heads: each (B, N, d)
        heads = X.split(self.d, dim=-1)

        out_heads = []
        for h in range(self.n_heads):
            X_h = heads[h]                               # (B, N, d)
            Q_h = self.W_q[h](X_h)                      # (B, N, d)
            K_h = X_h                                    # identity key: (B, N, d)

            # Gram matrix: G_h[b, i, j] = sum_n Q[b,n,i] * K[b,n,j]
            # = Q.T @ K in the N dimension → shape (B, d, d)
            G_h = torch.bmm(Q_h.transpose(1, 2), K_h)   # (B, d, d)

            # Sigmoid gating
            A_h = torch.sigmoid(G_h * self.tau[h])       # (B, d, d)

            # Value: 2-layer MLP + LayerNorm
            V_h = self.norm[h](F.gelu(self.W_v1[h](X_h)) @ self.W_v2[h].weight.T)  # (B, N, d)

            # Gated reconstruction: Z_h = V_h @ A_h
            Z_h = torch.bmm(V_h, A_h)                   # (B, N, d)
            out_heads.append(Z_h)

        Z = torch.cat(out_heads, dim=-1)                 # (B, N, D)
        if not batched:
            Z = Z.squeeze(0)
        return Z


class FusedIBProjector(nn.Module):
    """Wraps the existing SmolVLA connector linear with a Fused IB-Adapter path.

    Z = linear(X) + tanh(fusion_coeff) * ib_adapter(linear(X))

    The fusion_coeff is initialized to 0 so tanh(0)=0 and the model starts
    identical to vanilla SmolVLA.  Only ib_adapter params + fusion_coeff need
    to be trained; the original linear can be frozen or fine-tuned separately.

    Parameters
    ----------
    original_linear : nn.Linear
        The existing SmolVLA modality_projection.proj (Linear 12288 → 960).
    D_out : int
        Output dimension of original_linear (LLM embedding dim = 960).
    n_heads : int
        Number of IB-Adapter heads.
    """

    def __init__(self, original_linear: nn.Linear, D_out: int = 960,
                 n_heads: int = 8) -> None:
        super().__init__()
        self.linear = original_linear
        self.ib = IBAdapter(D_out, n_heads=n_heads)
        # Init at 0: tanh(0) = 0 → IB contribution is off initially
        self.fusion_coeff = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        z_mlp = self.linear(x)                             # (B, N, D_out)
        z_ib  = self.ib(z_mlp)                            # (B, N, D_out)
        return z_mlp + torch.tanh(self.fusion_coeff) * z_ib

    @property
    def ib_contribution(self) -> float:
        """Current tanh(fusion_coeff) as a scalar (for logging)."""
        return float(torch.tanh(self.fusion_coeff).item())


def inject_fused_ib_adapter(policy, D_out: int = 960, n_heads: int = 8) -> FusedIBProjector:
    """Replace SmolVLA's connector linear with a FusedIBProjector in-place.

    Returns the newly created FusedIBProjector.
    """
    conn = policy.model.vlm_with_expert.vlm.model.connector
    original_linear = conn.modality_projection.proj
    fused = FusedIBProjector(original_linear, D_out=D_out, n_heads=n_heads)
    # Match device + dtype of the existing linear (model may be bfloat16 on GPU)
    ref = next(original_linear.parameters())
    fused.ib.to(device=ref.device, dtype=ref.dtype)
    fused.fusion_coeff.data = fused.fusion_coeff.data.to(device=ref.device, dtype=ref.dtype)
    conn.modality_projection.proj = fused
    return fused


def save_ib_checkpoint(fused: FusedIBProjector, lm_expert, out_path: str,
                       config: dict, step: int, loss: float) -> None:
    """Save IB-Adapter weights + adapted action head weights."""
    torch.save({
        "ib_state": fused.ib.state_dict(),
        "fusion_coeff": fused.fusion_coeff.data.clone(),
        "lm_expert_state": lm_expert.state_dict(),
        "config": config,
        "step": step,
        "loss": loss,
    }, out_path)


def load_ib_checkpoint(policy, weights_path: str, device: str = "cuda") -> None:
    """Load a saved IB checkpoint into a policy that has been patched with
    inject_fused_ib_adapter()."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    conn = policy.model.vlm_with_expert.vlm.model.connector
    fused = conn.modality_projection.proj
    assert isinstance(fused, FusedIBProjector), \
        "Call inject_fused_ib_adapter(policy) before load_ib_checkpoint()"

    fused.ib.load_state_dict(ckpt["ib_state"])
    # Cast to match the model's dtype (checkpoint saved as float32, model may be bfloat16)
    ref = next(fused.linear.parameters())
    fused.ib.to(device=ref.device, dtype=ref.dtype)
    fused.fusion_coeff.data = ckpt["fusion_coeff"].to(device=ref.device, dtype=ref.dtype)
    policy.model.vlm_with_expert.lm_expert.load_state_dict(ckpt["lm_expert_state"])
    print(f"[ib] loaded weights from {weights_path}  "
          f"fusion_coeff={fused.ib_contribution:.3f}  step={ckpt.get('step', '?')}")
