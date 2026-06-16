"""Fine-tune SmolVLA with the Forge recipe (Levers 1, 3, 4).

Levers applied:
  L1 (EMA=0.9999)          -- weight averaging during training; EMA weights saved
  L3 (percentile norm)     -- 1-percentile soft-clamp on action output
  L4 (Tier-2 aug, p=0.3)   -- aggressive lighting augmentation on training images

Architecture is unchanged (chunk_size=50).  At inference use n=1 + SIB on top.

Usage:
    python scripts/finetune_forge.py --config configs/sib.yaml \\
        --steps 10000 --lr-head 2e-5 --lr-backbone 2e-6

Pipeline after this:
    python scripts/estimate_lambda.py --config configs/forge_ft.yaml
    python scripts/train.py --config configs/forge_ft.yaml --beta 1e-4 --tag forge_sib_b1e-4
    python scripts/eval.py  --config configs/forge_ft_n1.yaml \\
        --weights results/forge_ft/forge_sib_b1e-4.pt --tag forge_sib_n1
"""

from __future__ import annotations

import contextlib
import shutil
import sys as _sys
import pathlib as _pathlib

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sib.data import build_lerobot_dataset
from sib.utils import GpuHourLogger, load_config, resolve_device, save_json, set_seed


# --------------------------------------------------------------------------- #
# Lever 4: Tier-2 aggressive lighting augmentation (inline from forge_vla)
# --------------------------------------------------------------------------- #
class AggressiveLightingAugment:
    """Applies one of 5 extreme lighting transforms with probability p.
    Operates on [C, H, W] float tensors in [0, 1].
    """
    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(1)) > self.p:
            return img
        img = img.clone().clamp(0.0, 1.0)
        aug = int(torch.randint(0, 5, (1,)).item())
        if aug == 0:                                           # dark room
            img = img * float(torch.empty(1).uniform_(0.1, 0.3))
        elif aug == 1:                                         # warm cast
            img[0] = img[0] * float(torch.empty(1).uniform_(1.0, 1.4))
            img[2] = img[2] * float(torch.empty(1).uniform_(0.5, 0.8))
        elif aug == 2:                                         # cool cast
            img[0] = img[0] * float(torch.empty(1).uniform_(0.5, 0.8))
            img[2] = img[2] * float(torch.empty(1).uniform_(1.0, 1.4))
        elif aug == 3:                                         # colored LED
            ch = int(torch.randint(0, 3, (1,)).item())
            img[ch] = img[ch] * float(torch.empty(1).uniform_(1.3, 1.8))
            for c in range(3):
                if c != ch:
                    img[c] = img[c] * float(torch.empty(1).uniform_(0.3, 0.6))
        else:                                                  # gamma/exposure
            img = img.pow(float(torch.empty(1).uniform_(0.4, 2.5)))
        return img.clamp(0.0, 1.0)


IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def augment_batch(batch: dict, aug: AggressiveLightingAugment) -> dict:
    """Apply augmentation to all image keys in a training batch."""
    out = dict(batch)
    for key in IMAGE_KEYS:
        if key not in out or not torch.is_tensor(out[key]):
            continue
        t = out[key]                    # [B, C, H, W]  float32 in [0,1]
        augmented = torch.stack([aug(t[i]) for i in range(t.shape[0])])
        out[key] = augmented
    return out


# --------------------------------------------------------------------------- #
# Lever 1: EMA (inline, no forge_vla dependency)
# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and p.is_floating_point():
                    self.shadow[name] = p.detach().clone()

    def update(self, model: nn.Module) -> None:
        d, one_minus = self.decay, 1.0 - self.decay
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(d).add_(p.detach(), alpha=one_minus)

    def apply_shadow(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    p.data.copy_(self.shadow[name].to(p.device))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr-head", type=float, default=2e-5,
                    help="LR for action expert + projections")
    ap.add_argument("--lr-backbone", type=float, default=2e-6,
                    help="LR for VLM backbone (lower = safer)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--aug-prob", type=float, default=0.3,
                    help="Tier-2 augmentation probability")
    ap.add_argument("--out-name", default="smolvla_forge",
                    help="subdirectory under output_dir for the saved model")
    ap.add_argument("--lever3-clamp", type=float, default=0.98,
                    help="L3: percentile soft-clamp on normalised actions (0-1)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    base_out = Path(cfg["output_dir"])
    model_out = base_out / args.out_name
    model_out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load SmolVLA
    # ------------------------------------------------------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print("[forge] loading SmolVLA ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg["checkpoint"]).to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, cfg["checkpoint"])
    H = policy.config.chunk_size          # 50

    # ------------------------------------------------------------------
    # 2. Differential parameter groups
    #    backbone (vlm): slower LR — preserve pre-trained features
    #    action head (lm_expert + projections): faster LR — adapt to Forge
    # ------------------------------------------------------------------
    backbone_params, head_params = [], []
    backbone_prefix = "model.vlm_with_expert.vlm"
    for name, p in policy.named_parameters():
        if name.startswith(backbone_prefix):
            backbone_params.append(p)
        else:
            head_params.append(p)
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"[forge] backbone {n_backbone/1e6:.1f}M @ lr={args.lr_backbone:.1e}  "
          f"head {n_head/1e6:.1f}M @ lr={args.lr_head:.1e}", flush=True)

    # ------------------------------------------------------------------
    # 3. Dataset + augmentation
    # ------------------------------------------------------------------
    print("[forge] loading dataset ...", flush=True)
    dataset = build_lerobot_dataset(cfg["repo_id"], H, cfg["fps"],
                                    root=cfg.get("dataset_root"))
    aug = AggressiveLightingAugment(p=args.aug_prob)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, drop_last=True, pin_memory=True,
                        persistent_workers=True)

    # ------------------------------------------------------------------
    # 4. Optimiser + EMA
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps,
        eta_min=min(args.lr_backbone, args.lr_head) * 0.1)
    ema = EMA(policy, decay=args.ema_decay)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history, step = [], 0
    policy.train()
    print(f"[forge] training {args.steps} steps  aug_p={args.aug_prob}  "
          f"ema={args.ema_decay}  batch={args.batch_size}", flush=True)

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with GpuHourLogger("finetune_forge", base_out, cfg.get("n_gpus", 1)):
        while step < args.steps:
            for raw_batch in loader:
                # L4: Tier-2 lighting augmentation on raw images
                raw_batch = augment_batch(raw_batch, aug)

                batch = preprocessor(
                    {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in raw_batch.items()}
                )
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

                # L3: percentile soft-clamp on normalised action targets
                if "action" in batch and args.lever3_clamp < 1.0:
                    batch["action"] = batch["action"].clamp(
                        -args.lever3_clamp, args.lever3_clamp)

                with amp_ctx:
                    loss, _ = policy.forward(batch)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()

                # L1: EMA update after every gradient step
                ema.update(policy)

                if step % 100 == 0:
                    lrs = [g["lr"] for g in opt.param_groups]
                    print(f"[forge] step={step:6d}/{args.steps}  "
                          f"loss={loss.item():.4f}  "
                          f"lr_bb={lrs[0]:.1e} lr_hd={lrs[1]:.1e}",
                          flush=True)
                    history.append({"step": step, "loss": float(loss.item()),
                                    "lr_backbone": float(lrs[0]),
                                    "lr_head": float(lrs[1])})
                step += 1
                if step >= args.steps:
                    break

    # ------------------------------------------------------------------
    # 6. Apply EMA shadow weights and save
    # ------------------------------------------------------------------
    print("[forge] applying EMA shadow weights ...", flush=True)
    ema.apply_shadow(policy)
    policy.eval()

    policy.save_pretrained(str(model_out))
    print(f"[forge] EMA model saved -> {model_out}", flush=True)

    # Copy preprocessor companion files from the original checkpoint so that
    # from_pretrained + make_pre_post_processors work without 404 errors.
    orig = Path(cfg["checkpoint"])
    companion_globs = [
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
    ]
    for fname in companion_globs:
        src = orig / fname
        if src.exists():
            shutil.copy2(src, model_out / fname)
            print(f"[forge] copied {fname}", flush=True)
        else:
            # Try HF cache snapshot
            import glob as _glob
            hits = _glob.glob(str(Path.home() /
                               ".cache/huggingface/hub/**/snapshots/**" / fname),
                              recursive=True)
            if hits:
                shutil.copy2(hits[0], model_out / fname)
                print(f"[forge] copied {fname} (from HF cache)", flush=True)

    # Inject draccus discriminator into config.json (same fix as smolvla_lora)
    import json
    cfg_path = model_out / "config.json"
    if cfg_path.exists():
        d = json.load(open(cfg_path))
        if "type" not in d:
            json.dump({"type": "smolvla", **d}, open(cfg_path, "w"))
            print("[forge] injected 'type': 'smolvla' into config.json", flush=True)

    save_json(
        {"steps": args.steps, "lr_head": args.lr_head,
         "lr_backbone": args.lr_backbone, "batch_size": args.batch_size,
         "ema_decay": args.ema_decay, "aug_prob": args.aug_prob,
         "lever3_clamp": args.lever3_clamp,
         "out": str(model_out), "history": history},
        base_out / "finetune_forge.json",
    )
    print(f"[forge] done. final loss={history[-1]['loss']:.4f}" if history else "[forge] done.")


if __name__ == "__main__":
    main()
