"""AEGIS modules for GR00T N1.5 — self-contained (no sib_vla deps, Modal-portable).

Dual-locus, identity-residual plug-ins for a FROZEN GR00T N1.5 (Eagle VLM + flow-
matching action head). Hook points verified in Isaac-GR00T v1.1.0 source:

  RIB  (perception) -> wrap  model.backbone.eagle_linear  (Linear 2048->1536)
                       eagle_backbone.py:54 / forward_eagle :112  -> features (B,T,1536)
  RASF (action)     -> filter the final denoised chunk in the action head get_action,
                       flow_matching_action_head.py:407-408     -> actions (B,16,d)
                       (applied at the normalised-action level, where it was trained)

Both are EXACT pass-throughs at init (RIB fusion decoder zero-init; RASF gate=0), so
clean success cannot change until the modules are trained. Mirrors the SmolVLA AEGIS
design (sib/robust_ib.py, sib/adaptive_filter.py) at GR00T dims.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ======================================================================== RIB
class RobustIB(nn.Module):
    """Deterministic robust information bottleneck on connector features (B,N,D)."""
    def __init__(self, D: int = 1536, d_z: int = 448, n_heads: int = 8) -> None:
        super().__init__()
        self.D, self.d_z = D, d_z
        self.enc = nn.Linear(D, d_z)
        self.enc_ln = nn.LayerNorm(d_z)
        self.attn = nn.MultiheadAttention(d_z, n_heads, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_z)
        self.to_z = nn.Linear(d_z, d_z)
        self.dec = nn.Sequential(nn.Linear(d_z, d_z), nn.GELU(), nn.Linear(d_z, D))
        nn.init.zeros_(self.dec[-1].weight)          # zero-init -> Z_ib=0 at step 0
        nn.init.zeros_(self.dec[-1].bias)
        self.last_kl: Tensor | None = None

    def forward(self, X: Tensor) -> Tensor:
        squeeze = X.dim() == 2
        if squeeze:
            X = X.unsqueeze(0)
        in_dtype = X.dtype
        h = F.gelu(self.enc_ln(self.enc(X.float())))
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_ln(h + a)
        z = self.to_z(h)
        Z_ib = self.dec(z)
        self.last_kl = z.pow(2).mean()               # L2 rate proxy (anti-collapse)
        out = Z_ib.to(in_dtype)
        return out.squeeze(0) if squeeze else out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FusedRobustIBProjector(nn.Module):
    """Wrap an existing Linear with an identity-initialised RIB residual:
        out = linear(x) + tanh(fusion_coeff) * RIB(linear(x))
    fusion_coeff>0 (live gradient) + RIB decoder zero-init (Z_ib=0) => out==linear(x).
    """
    def __init__(self, original_linear: nn.Linear, D_out: int = 1536,
                 d_z: int = 448, n_heads: int = 8) -> None:
        super().__init__()
        self.linear = original_linear
        self.rib = RobustIB(D_out, d_z=d_z, n_heads=n_heads)
        self.fusion_coeff = nn.Parameter(torch.full((1,), 0.5493))   # tanh=0.5, not dead

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        return z + torch.tanh(self.fusion_coeff) * self.rib(z)

    @property
    def ib_contribution(self) -> float:
        return float(torch.tanh(self.fusion_coeff).item())


def inject_rib(model, d_z: int = 384, n_heads: int = 8) -> FusedRobustIBProjector:
    """Wrap GR00T's REAL vision->LLM connector with a FusedRobustIBProjector in place.

    NOTE: in GR00T N1.5, backbone.eagle_linear is nn.Identity (verified by structure
    dump). The actual perception projector is backbone.eagle_model.mlp1 (a Sequential
    that maps SigLIP vision tokens -> LLM hidden 2048). That is the correct RIB locus
    (the analog of SmolVLA's connector hook). D_out is read from the connector's last
    Linear so this self-adapts. Identity at init (RIB decoder zero-init)."""
    bb = model.backbone
    conn = bb.eagle_model.mlp1
    if isinstance(conn, FusedRobustIBProjector):
        return conn
    lins = [m for m in conn.modules() if isinstance(m, nn.Linear)]
    assert lins, f"no nn.Linear inside connector mlp1 ({type(conn).__name__})"
    D_out = lins[-1].out_features                              # LLM hidden (2048)
    fused = FusedRobustIBProjector(conn, D_out=D_out, d_z=d_z, n_heads=n_heads)
    ref = next(conn.parameters())
    # Keep RIB weights in fp32: RIB.forward casts activations to float() for a stable
    # bottleneck and re-casts the output to the input dtype. Casting RIB weights to the
    # model's bf16 would clash (fp32 acts x bf16 weights). Only fusion_coeff (which scales
    # the bf16 RIB output) must match the model dtype.
    fused.rib.to(device=ref.device)                              # fp32 weights, model device
    fused.fusion_coeff.data = fused.fusion_coeff.data.to(device=ref.device, dtype=ref.dtype)
    bb.eagle_model.mlp1 = fused
    return fused


# ======================================================================= RASF
def dct_matrix(N: int) -> Tensor:
    n = torch.arange(N, dtype=torch.float64)
    k = n.view(-1, 1)
    M = torch.cos(math.pi / N * (n.view(1, -1) + 0.5) * k) * math.sqrt(2.0 / N)
    M[0] = M[0] / math.sqrt(2.0)
    return M.to(torch.float32)


class AdaptiveSpectralFilter(nn.Module):
    """Residual, identity-initialised, input-adaptive spectral denoiser (bounded).
    Verbatim port of sib/adaptive_filter.py. A_hat = A + gate_max*tanh(gate)*(IDCT(g*DCT(A))-A)."""
    def __init__(self, H: int, d: int, *, hidden: int = 32,
                 gain_floor: float = 0.30, base_gain_logit: float = 4.0,
                 gate_max: float = 0.6) -> None:
        super().__init__()
        self.H, self.d = H, d
        self.gain_floor = float(gain_floor)
        self.gate_max = float(gate_max)
        self.register_buffer("dct", dct_matrix(H))
        self.register_buffer("idct", dct_matrix(H).t().contiguous())
        self.band_logit = nn.Parameter(torch.full((H,), float(base_gain_logit)))
        self.mod = nn.Sequential(nn.Linear(H, hidden), nn.GELU(), nn.Linear(hidden, H))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        self.gate = nn.Parameter(torch.zeros(d))                # 0 -> identity at init

    def forward(self, A: Tensor, context=None) -> dict:
        squeeze = A.dim() == 2
        if squeeze:
            A = A.unsqueeze(0)
        nd = A.dtype
        Af32 = A.float()
        C = torch.einsum("kh,bhd->bkd", self.dct, Af32)
        e = (C.pow(2).mean(dim=-1) + 1e-8).log()
        logit = self.band_logit.unsqueeze(0) + self.mod(e)
        g = torch.sigmoid(logit)
        g = self.gain_floor + (1.0 - self.gain_floor) * g
        Cf = g.unsqueeze(-1) * C
        Af = torch.einsum("hk,bkd->bhd", self.idct, Cf)
        strength = self.gate_max * torch.tanh(self.gate)
        A_hat = Af32 + strength.view(1, 1, -1) * (Af - Af32)
        A_hat = A_hat.to(nd)
        return {"A_hat": A_hat.squeeze(0) if squeeze else A_hat,
                "gain": g, "gate": strength.view(1, 1, -1)}

    @property
    def gate_strength(self) -> float:
        return float((self.gate_max * torch.tanh(self.gate).abs()).mean().item())


def attach_rasf(policy, rasf: "AdaptiveSpectralFilter"):
    """Monkeypatch a Gr00tPolicy so RASF filters the NORMALISED action chunk
    (B,H,d) before unnormalisation — the regime RASF is trained in. Identity at init.
    Wraps policy._get_action_from_normalized_input (returns normalized (B,H,d))."""
    if getattr(policy, "_rasf_attached", False):
        return policy
    orig = policy._get_action_from_normalized_input
    rasf_dev = next(rasf.parameters()).device

    def wrapped(normalized_input):
        a = orig(normalized_input)               # (B, H, d) normalised torch tensor
        out = rasf(a.to(rasf_dev).float(), context=None)["A_hat"]
        return out.to(a.device, a.dtype)
    policy._get_action_from_normalized_input = wrapped
    policy._rasf_attached = True
    policy._rasf = rasf
    return policy
