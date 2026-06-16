"""CPU smoke test for the AEGIS-on-NanoVLA injection (nanovla/aegis_nanovla.py).

Runs entirely on CPU with a DUMMY batch (no dataset, no GPU) so it can run alongside
a training job with zero contention. Verifies the two guarantees that matter:

  1. IDENTITY AT INIT  — AEGIS(model) at step 0 reproduces vanilla NanoVLA's action
     chunk exactly (RIB zero-init decoder + RASF zero gate). Clean SR cannot degrade.
  2. GRADIENT REACHES RIB — after one training backward, RIB params + fusion_coeff
     receive non-zero gradient (no StableVLA dead-start dormancy).

    CUDA_VISIBLE_DEVICES="" python scripts/smoke_aegis_nanovla.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch
from lerobot.utils.constants import OBS_STATE, OBS_IMAGES, ACTION
from nanovla.modeling_nanovla import NanoVLAS, make_nanovla_config
from nanovla.aegis_nanovla import AEGISNanoVLA

torch.manual_seed(0)
DEV = "cpu"
B, H, A_DIM, S_DIM = 2, 100, 7, 8


def dummy_batch():
    return {
        OBS_STATE: torch.randn(B, S_DIM),
        OBS_IMAGES: [torch.rand(B, 3, 224, 224)],
        ACTION: torch.randn(B, H, A_DIM),
        "task": ["pick up the black bowl", "open the top drawer"],
        "action_is_pad": torch.zeros(B, H, dtype=torch.bool),
    }


def main():
    print("=== AEGIS-on-NanoVLA CPU smoke ===", flush=True)
    cfg = make_nanovla_config(state_dim=S_DIM, action_dim=A_DIM, chunk_size=H, n_action_steps=10)
    # distilbert is cached & 768-d like bert-base -> matches env_state slot, no download
    model = NanoVLAS(cfg, lang_model="distilbert-base-uncased").to(DEV)
    model.eval()
    batch = dummy_batch()

    # 1) vanilla chunk BEFORE injection -------------------------------------
    with torch.no_grad():
        A_vanilla = model.predict_action_chunk(batch).clone()
    print(f"[smoke] vanilla chunk shape={tuple(A_vanilla.shape)}", flush=True)

    # 2) inject AEGIS (in-place into model.act) + re-predict -----------------
    aegis = AEGISNanoVLA(model, rib_d_z=256, rib_heads=8,
                         rasf_gate_max=0.95, rasf_gain_floor=0.05).to(DEV)
    aegis.eval()
    with torch.no_grad():
        A_aegis = aegis.predict_action_chunk(batch)
    diff = (A_aegis - A_vanilla).abs().max().item()
    n_rib = aegis.rib.n_params() / 1e6
    n_rasf = sum(p.numel() for p in aegis.rasf.parameters())
    print(f"[smoke] RIB params={n_rib:.2f}M  RASF params={n_rasf}  "
          f"fusion_coeff={aegis.rib.ib_contribution:.3f}  rasf_gate={aegis.rasf.gate_strength:.3f}")
    print(f"[smoke] |AEGIS - vanilla|_max at init = {diff:.2e}", flush=True)
    assert diff < 1e-4, f"IDENTITY-AT-INIT VIOLATED: max diff {diff:.2e} (clean SR would shift)"
    print("[smoke] (1) identity-at-init: PASS", flush=True)

    # 3) training backward -> RIB must get gradient (no dead-start dormancy).
    #    fusion_coeff grad is 0 at the EXACT init point by design (zero-init decoder
    #    => rib(tok)=0 => d out/d coeff = sech^2(coeff)*rib(tok) = 0). It becomes
    #    reachable once the decoder moves off zero, so we take one optimizer step on
    #    the RIB params and confirm fusion_coeff then receives gradient.
    model.train()
    rib_params = list(aegis.rib.rib.parameters())
    opt = torch.optim.SGD(rib_params + [aegis.rib.fusion_coeff], lr=0.1)

    loss, parts = aegis.forward(batch)
    loss.backward()
    g_rib = max((p.grad.abs().max().item() for p in rib_params if p.grad is not None), default=0.0)
    gc0 = aegis.rib.fusion_coeff.grad
    gc0 = float(gc0.abs().max().item()) if gc0 is not None else 0.0
    print(f"[smoke] step0: loss={parts['loss']:.4f}  max|grad RIB|={g_rib:.2e}  "
          f"max|grad coeff|={gc0:.2e} (0 by design: zero-init decoder)", flush=True)
    assert g_rib > 0, "RIB received NO gradient (dead-start dormancy)"
    opt.step()                                    # decoder now off zero

    aegis.zero_grad(set_to_none=True)
    loss2, _ = aegis.forward(batch)
    loss2.backward()
    gc1 = aegis.rib.fusion_coeff.grad
    gc1 = float(gc1.abs().max().item()) if gc1 is not None else 0.0
    print(f"[smoke] step1: max|grad coeff|={gc1:.2e} (reachable once decoder != 0)", flush=True)
    assert gc1 > 0, "fusion_coeff UNREACHABLE (detached from graph)"
    print("[smoke] (2) gradient reaches RIB (init) + fusion_coeff (step1): PASS", flush=True)

    print("\n[smoke] ALL CHECKS PASSED — AEGIS ports cleanly onto NanoVLA/ACT.", flush=True)


if __name__ == "__main__":
    main()
