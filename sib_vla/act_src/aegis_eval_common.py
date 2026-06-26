"""aegis_eval_common.py — load the colleague's ACT base, optionally with the AEGIS RIB leg.

AEGIS arm = base + RIB (loaded via load_rib_act_checkpoint, which injects at
encoder_img_feat_input_proj). RIB then applies INSIDE policy.predict_action_chunk, so the
stock select_action queue routes through it transparently. RASF optional (off by default —
RIB is the LIBERO-Plus lever). Base arm = identical load with rib_weights=None.
"""
from __future__ import annotations
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIBVLA = os.path.dirname(HERE)
for p in (HERE, SIBVLA):
    if p not in sys.path:
        sys.path.insert(0, p)

from lerobot.policies.act.processor_act import make_act_pre_post_processors
from act_common import load_ckpt
from sib.robust_ib_act import load_rib_act_checkpoint


def load_aegis_policy(meta, base_ckpt, replan, rib_weights=None, device="cuda", fusion_mult=1.0):
    policy, variant = load_ckpt(meta, base_ckpt)
    assert variant == "act", f"expected variant 'act', got {variant}"
    policy = policy.to(device).eval()
    policy.config.n_action_steps = replan
    if rib_weights:
        load_rib_act_checkpoint(policy, rib_weights, device=device)  # injects + loads RIB
        policy = policy.to(device).eval()
        if fusion_mult != 1.0:
            # de-strength the RIB residual at eval (no retrain): scale tanh(fusion_coeff) by mult.
            # mult=0 -> identity (== base exactly); mult=1 -> trained strength.
            m = policy.model.encoder_img_feat_input_proj
            orig = torch.tanh(m.fusion_coeff.data)
            scaled = torch.clamp(fusion_mult * orig, -0.999, 0.999)
            m.fusion_coeff.data = torch.atanh(scaled)
            print(f"[aegis] fusion de-strength mult={fusion_mult} -> contribution {m.ib_contribution:.3f}", flush=True)
    pre, post = make_act_pre_post_processors(policy.config, dataset_stats=meta.stats)
    return policy, pre, post
