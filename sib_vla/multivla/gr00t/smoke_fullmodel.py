"""Tier-2 smoke: wire AEGIS into the REAL GR00T N1.5 — RAM-bounded, CPU-only.
Avoids a full VLM forward (no flash-attn/GPU needed): only checks the eagle_linear
submodule identity + attribute paths + action-head config + RASF on synthetic chunk.

Run RAM-capped:  ulimit -v 35000000; GR00T_PATH=nvidia/GR00T-N1.5-3B python smoke_fullmodel.py
"""
import os, sys, torch, torch.nn as nn
from gr00t_aegis import inject_rib, FusedRobustIBProjector, AdaptiveSpectralFilter

PATH = os.environ.get("GR00T_PATH", "nvidia/GR00T-N1.5-3B")
ok = True
def check(n, c):
    global ok; print(f"  [{'PASS' if c else 'FAIL'}] {n}"); ok = ok and c

print(f"== loading GR00T N1.5 on CPU (low-mem) from {PATH} ==", flush=True)
from gr00t.model.gr00t_n1 import GR00T_N1_5
model = GR00T_N1_5.from_pretrained(PATH, torch_dtype=torch.float32,
                                   low_cpu_mem_usage=True).eval().cpu()
print("  loaded.", flush=True)

# --- RIB hook: real connector ---
print("== RIB wiring (model.backbone.eagle_linear) ==")
el = model.backbone.eagle_linear
check("eagle_linear is nn.Linear", isinstance(el, nn.Linear))
if isinstance(el, nn.Linear):
    check("dims 2048 -> 1536", (el.in_features, el.out_features) == (2048, 1536))
    x = torch.randn(1, 4, el.in_features)
    with torch.no_grad():
        base = el(x)
    fused = inject_rib(model)                          # in-place swap
    check("eagle_linear replaced by FusedRobustIBProjector",
          isinstance(model.backbone.eagle_linear, FusedRobustIBProjector))
    with torch.no_grad():
        out = model.backbone.eagle_linear(x)
    check("IDENTITY at init on REAL connector (||fused-base||~0)",
          out.shape == base.shape and torch.allclose(out, base, atol=1e-4))
    check("RIB params < 3M", fused.rib.n_params() < 3_000_000)
    print(f"     RIB params = {fused.rib.n_params()/1e6:.2f}M; fusion tanh={fused.ib_contribution:.3f}")

# --- RASF hook: action-head dims ---
print("== RASF dims (action head) ==")
ah = model.action_head
H = getattr(model, "action_horizon", None) or ah.config.action_horizon
d = getattr(model, "action_dim", None) or ah.config.action_dim
print(f"     action_horizon={H}  action_dim={d}")
check("action_horizon == 16", int(H) == 16)
rasf = AdaptiveSpectralFilter(H=int(H), d=int(d), gate_max=0.6).eval()
A = torch.randn(2, int(H), int(d))
with torch.no_grad():
    Ah = rasf(A)["A_hat"]
check("RASF identity at init on (B,H,d)", torch.allclose(Ah, A, atol=1e-6))

print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
sys.exit(0 if ok else 1)
