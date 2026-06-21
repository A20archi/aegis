"""Tier-1 smoke: validate GR00T AEGIS modules in isolation (CPU, tiny tensors).
Zero model load -> a few MB RAM. Checks identity-at-init, shapes, dtype, params.
Run under a hard RAM cap, e.g.:  ulimit -v 4000000; python smoke_modules.py
"""
import torch
from gr00t_aegis import (RobustIB, FusedRobustIBProjector, inject_rib,
                         AdaptiveSpectralFilter, attach_rasf)
import torch.nn as nn

torch.manual_seed(0)
ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

print("== RIB (perception, D_out=1536) ==")
lin = nn.Linear(2048, 1536)                      # GR00T eagle_linear dims
fused = FusedRobustIBProjector(lin, D_out=1536, d_z=448, n_heads=8).eval()
x = torch.randn(2, 5, 2048)                       # (B,T,2048) tiny
with torch.no_grad():
    base = lin(x)
    out = fused(x)
check("output shape == base (B,T,1536)", out.shape == base.shape == (2, 5, 1536))
check("IDENTITY at init: ||fused - base|| ~ 0",
      torch.allclose(out, base, atol=1e-5))
check("fusion_coeff live (tanh=0.5, not dead-start)", abs(fused.ib_contribution - 0.5) < 0.02)
check("RIB params < 3M (budget)", fused.rib.n_params() < 3_000_000)
print(f"     RIB params = {fused.rib.n_params()/1e6:.2f}M")
# non-dormant after a perturbation to decoder (sanity: it CAN deviate once trained)
with torch.no_grad():
    fused.rib.dec[-1].weight.add_(1e-2 * torch.randn_like(fused.rib.dec[-1].weight))
    out2 = fused(x)
check("RIB CAN engage after weights change (not structurally dead)",
      not torch.allclose(out2, base, atol=1e-5))

print("== RASF (action, H=16, d=8) ==")
rasf = AdaptiveSpectralFilter(H=16, d=8, gate_max=0.6).eval()
A = torch.randn(3, 16, 8)
with torch.no_grad():
    Ah = rasf(A)["A_hat"]
check("output shape == (B,16,8)", Ah.shape == (3, 16, 8))
check("IDENTITY at init: ||A_hat - A|| ~ 0 (gate=0)", torch.allclose(Ah, A, atol=1e-6))
check("gate_strength == 0 at init", rasf.gate_strength == 0.0)
# engage once gate>0 -> deviates but stays bounded
with torch.no_grad():
    rasf.gate.add_(1.0)
    Ah2 = rasf(A)["A_hat"]
    resid = (Ah2 - A).abs().max().item()
    bound = rasf.gate_max * (rasf(A)["gate"].abs().max().item() / rasf.gate_max + 0)  # gate<=gate_max
check("RASF engages when gate>0 (not dead)", not torch.allclose(Ah2, A, atol=1e-4))
check("residual bounded (no blow-up)", torch.isfinite(Ah2).all() and resid < 50)

print("== dtype passthrough (bf16 in -> bf16 out) ==")
with torch.no_grad():
    Ah_bf = rasf.float()  # filter math in fp32 internally
    rasf.gate.zero_()
    o = rasf(A.to(torch.bfloat16))["A_hat"]
check("RASF returns input dtype", o.dtype == torch.bfloat16)

print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
import sys; sys.exit(0 if ok else 1)
