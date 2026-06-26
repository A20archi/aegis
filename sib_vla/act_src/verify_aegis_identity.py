"""
verify_aegis_identity.py — prove AEGIS ≡ base at init on the colleague's ACT.

AEGIS adds RIB (identity-init residual at encoder_img_feat_input_proj) + RASF
(identity-init SpectralActionModule on the action chunk). At zero strength the
wrapped policy must produce a BIT-IDENTICAL action chunk to the bare base — the
no-harm safety property. Runs on CPU (no GPU contention with a running eval).
"""
from __future__ import annotations
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIBVLA = os.path.dirname(HERE)
for p in (HERE, SIBVLA):
    if p not in sys.path:
        sys.path.insert(0, p)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from act_common import load_ckpt
from sib.robust_ib_act import inject_rib_act
from sib.bottleneck import SpectralActionModule
from sib.wrapper import SIBPolicy

CKPT = os.path.join(SIBVLA, "act_ckpts", "Spatial", "act", "30000")
DATASET = "lerobot/libero_spatial_image"


def dummy_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "observation.images.image": torch.rand(1, 3, 256, 256, generator=g),
        "observation.images.wrist_image": torch.rand(1, 3, 256, 256, generator=g),
        "observation.state": torch.randn(1, 8, generator=g),
        "task": ["pick up the black bowl and place it on the plate"],
    }


def main():
    torch.manual_seed(0)
    meta = LeRobotDatasetMetadata(DATASET)

    # --- bare base ---
    base, _ = load_ckpt(meta, CKPT)
    base = base.float().eval()
    batch = dummy_batch()
    with torch.no_grad():
        chunk_base = base.predict_action_chunk(batch).float()   # (1,100,7)

    H, d = base.config.chunk_size, base.config.action_feature.shape[0]

    # --- Leg 1: RIB only (untrained, identity-init residual) — must be BIT-EXACT ---
    rib_base, _ = load_ckpt(meta, CKPT)
    rib_base = rib_base.float().eval()
    fused = inject_rib_act(rib_base)                              # identity at init
    with torch.no_grad():
        chunk_rib = rib_base.predict_action_chunk(batch).float()
    d_rib = (chunk_base - chunk_rib).abs()

    # --- Leg 2: full AEGIS = RIB + identity RASF (DCT→gain(=1)→iDCT) ---
    rasf = SpectralActionModule("gain_no_rate", H, d).float().eval()  # identity at init
    aegis = SIBPolicy(rib_base, rasf, n_action_steps=5).float().eval()
    with torch.no_grad():
        chunk_aegis = aegis.correct_chunk(chunk_rib, batch)
    d_aegis = (chunk_base - chunk_aegis).abs()

    print(f"chunk shape              : {tuple(chunk_base.shape)}  (H={H}, d={d})")
    print(f"RIB fusion at init       : {fused.ib_contribution:.4f}  (residual = 0 via zero-init decoder)")
    print(f"[RIB leg]   max|Δ| base   : {d_rib.max().item():.3e}   (expect 0: bit-exact residual)")
    print(f"[RIB+RASF]  max|Δ| base   : {d_aegis.max().item():.3e}   (expect ~1e-6: DCT round-trip fp)")
    rib_exact = torch.allclose(chunk_base, chunk_rib, atol=1e-6, rtol=0)
    aegis_ok = torch.allclose(chunk_base, chunk_aegis, atol=1e-5, rtol=0)
    print(f"RIB bit-exact (atol 1e-6)         : {'PASS ✅' if rib_exact else 'FAIL ❌'}")
    print(f"RIB+RASF identity (atol 1e-5)     : {'PASS ✅' if aegis_ok else 'FAIL ❌'}")
    print(f"\nIDENTITY-AT-INIT: {'PASS ✅ — AEGIS ≡ base (RIB exact; RASF identity to fp precision)' if (rib_exact and aegis_ok) else 'FAIL ❌'}")
    sys.exit(0 if (rib_exact and aegis_ok) else 1)


if __name__ == "__main__":
    main()
