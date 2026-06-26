#!/usr/bin/env python
"""
finetune_rib_act_v2.py — train the AEGIS RIB perception leg on the COLLEAGUE's frozen ACT.

Loads the colleague's per-suite ACT baseline (act_common.load_ckpt: model.* + lang_encoder),
injects an identity-init RIB at encoder_img_feat_input_proj, FREEZES everything except RIB,
and trains it with view-asymmetric corruption-augmented consistency:

    L = L1(GT action | agentview corrupted, wrist clean, z=0) + beta * KL(RIB)

Base runs in eval() so the CVAE latent z=0 (NO ground-truth-action leak through the VAE) and
BatchNorm stays frozen; only the RIB submodule is in train() so its VIB sampling + KL are live.
RIB is identity at init (zero-init decoder) so clean SR is protected; clean samples in each batch
pull RIB(clean)->0 while corrupted agentviews teach it to drop the nuisance subspace.

    python finetune_rib_act_v2.py \
        --ckpt-dir ../act_ckpts/Spatial/act/30000 --dataset lerobot/libero_spatial_image \
        --out ../results/aegis_act_v2/Spatial/rib.pt --steps 6000
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIBVLA = os.path.dirname(HERE)
for p in (HERE, SIBVLA):
    if p not in sys.path:
        sys.path.insert(0, p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from torch.utils.data import DataLoader

from act_common import load_ckpt
from sib.robust_ib_act import inject_rib_act, save_rib_act_checkpoint
from scripts.finetune_rib import _augment_images, _image_keys  # corruption aug (B,C,H,W) in [0,1]

DEV = "cuda"
WRIST_HINTS = ("wrist", "image2", "eye_in_hand")


def to_dev(b):
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="colleague ACT ckpt dir (model.safetensors+meta.json)")
    ap.add_argument("--dataset", required=True, help="lerobot/libero_<suite>_image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr-rib", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--d-z", type=int, default=384)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--free-bits", type=float, default=0.1)
    ap.add_argument("--corrupt-frac", type=float, default=0.6)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- frozen colleague baseline -----
    print(f"[rib-act-v2] loading baseline {args.ckpt_dir}", flush=True)
    meta = LeRobotDatasetMetadata(args.dataset)
    policy, variant = load_ckpt(meta, args.ckpt_dir)
    assert variant == "act", f"expected variant 'act', got {variant}"
    policy = policy.to(DEV)
    cfg = policy.config
    H = cfg.chunk_size
    pre, _ = make_act_pre_post_processors(cfg, dataset_stats=meta.stats)

    # ----- inject RIB, freeze all but RIB -----
    fused = inject_rib_act(policy, d_z=args.d_z, n_heads=args.n_heads)
    policy.to(DEV)
    for p in policy.parameters():
        p.requires_grad_(False)
    rib_params = list(fused.rib.parameters()) + [fused.fusion_coeff]
    for p in rib_params:
        p.requires_grad_(True)
    n_rib = sum(p.numel() for p in rib_params)
    print(f"[rib-act-v2] trainable {n_rib/1e6:.3f}M @ lr={args.lr_rib:.1e} beta={args.beta} "
          f"corrupt_frac={args.corrupt_frac} H={H} (ACT FROZEN, base eval/z=0, RIB train)", flush=True)

    # ----- data (action chunk + action_is_pad + task, same as base training) -----
    dt = resolve_delta_timestamps(cfg, meta)
    ds = LeRobotDataset(args.dataset, delta_timestamps=dt)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=(args.num_workers > 0),
                        prefetch_factor=(4 if args.num_workers > 0 else None))
    print(f"[rib-act-v2] dataset {args.dataset} frames={ds.num_frames}", flush=True)

    opt = torch.optim.AdamW(rib_params, lr=args.lr_rib, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr_rib * 0.1)
    gen = torch.Generator(device=DEV).manual_seed(1000)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # base eval() -> z=0, BN frozen; RIB submodule train() -> VIB sampling + KL live
    policy.eval()
    fused.rib.train()

    history, step, t0 = [], 0, time.perf_counter()
    print(f"[rib-act-v2] training {args.steps} steps bs={args.batch_size}", flush=True)
    while step < args.steps:
        for raw_batch in loader:
            raw = to_dev(raw_batch)
            B = raw["action"].shape[0]
            # view-asymmetric corruption: agentview only, wrist stays clean
            cmask = torch.rand(B, generator=gen, device=DEV) < args.corrupt_frac
            if cmask.any():
                for k in _image_keys(raw):
                    if any(h in k for h in WRIST_HINTS):
                        continue
                    raw[k] = raw[k].clone()
                    raw[k][cmask] = _augment_images(raw[k][cmask], gen).to(raw[k].dtype)
            batch = to_dev(pre(raw))

            with amp:
                task_loss, info = policy.forward(batch)            # L1 only (eval -> z=0, no CVAE KL)
                kl = fused.last_kl.float() if fused.last_kl is not None else torch.zeros((), device=DEV)
                loss = task_loss + args.beta * torch.clamp(kl, min=args.free_bits)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rib_params, 1.0)
            opt.step(); sched.step()

            if step % 100 == 0:
                print(f"[rib-act-v2] step={step:5d}/{args.steps} loss={loss.item():.4f} "
                      f"l1={task_loss.item():.4f} kl={kl.item():.3f} fusion={fused.ib_contribution:+.3f} "
                      f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)
                history.append({"step": step, "loss": float(loss.item()), "l1": float(task_loss.item()),
                                "kl": float(kl.item()), "fusion": float(fused.ib_contribution)})
            if step % args.ckpt_every == 0 and step > 0:
                save_rib_act_checkpoint(fused, str(out_path),
                                        config={"d_z": args.d_z, "n_heads": args.n_heads,
                                                "base_checkpoint": args.ckpt_dir, "dataset": args.dataset},
                                        step=step, loss=float(loss.item()))
                fused.rib.train()
            step += 1
            if step >= args.steps:
                break

    save_rib_act_checkpoint(fused, str(out_path),
                            config={"d_z": args.d_z, "n_heads": args.n_heads,
                                    "base_checkpoint": args.ckpt_dir, "dataset": args.dataset},
                            step=step, loss=history[-1]["loss"] if history else float("nan"))
    wall = (time.perf_counter() - t0) / 60
    print(f"[rib-act-v2] DONE steps={step} wall={wall:.1f}min -> {out_path}  "
          f"final_fusion={fused.ib_contribution:+.3f} "
          f"({'ENGAGED' if abs(fused.ib_contribution) > 0.05 else '~identity'})", flush=True)
    json.dump({"steps": args.steps, "lr_rib": args.lr_rib, "beta": args.beta,
               "corrupt_frac": args.corrupt_frac, "d_z": args.d_z, "dataset": args.dataset,
               "wall_min": wall, "final_fusion": fused.ib_contribution, "history": history},
              open(str(out_path).replace(".pt", ".train.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
