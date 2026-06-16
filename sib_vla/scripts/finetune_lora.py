"""LoRA fine-tune SmolVLA on LIBERO-Spatial training data.

Applies PEFT LoRA to SmolVLA's action-expert attention (q/v) and all action
projection layers (SmolVLA's built-in default PEFT targets).  Base VLM
vision encoder and language backbone weights stay frozen by PEFT; only the
LoRA adapters (~1-2% of params) are trained.  After training the adapters
are merged back into the base weights and the full model is saved so that
the existing load_smolvla / from_pretrained path works without modification.

    python scripts/finetune_lora.py --config configs/sib.yaml
    python scripts/finetune_lora.py --config configs/sib.yaml --steps 4000 --rank 16 --lr 2e-4

Outputs:
    results/smolvla_lora/          merged checkpoint (from_pretrained compatible)
    results/finetune_lora.json     loss curve + hyperparams

Next steps after this script:
    python scripts/estimate_lambda.py  --config configs/sib_lora.yaml
    python scripts/train.py            --config configs/sib_lora.yaml --beta 1e-4 --tag sib_lora_b1e-4
    python scripts/eval.py             --config configs/vanilla_lora_n4.yaml
    python scripts/eval.py             --config configs/sib_lora_n4.yaml \\
        --weights results/sib_lora/sib_lora_b1e-4.pt
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sib.data import build_lerobot_dataset, make_action_delta_timestamps
from sib.utils import GpuHourLogger, load_config, resolve_device, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=4000,
                    help="gradient steps (model is already LIBERO-adapted; ~3-5k is enough)")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    ap.add_argument("--alpha", type=int, default=32,  help="LoRA alpha (typically 2*rank)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-name", default="smolvla_lora",
                    help="subdirectory under output_dir for the saved model")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    base_out = Path(cfg["output_dir"])
    model_out = base_out / args.out_name

    # ------------------------------------------------------------------
    # 1. Load SmolVLA policy and its preprocessor
    # ------------------------------------------------------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print("[lora] loading SmolVLA...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg["checkpoint"])
    policy = policy.to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, cfg["checkpoint"])
    H = policy.config.chunk_size   # 50 for SmolVLA

    # ------------------------------------------------------------------
    # 2. Apply PEFT LoRA using SmolVLA's built-in target_modules
    # ------------------------------------------------------------------
    from peft import get_peft_model, LoraConfig

    peft_targets = policy._get_default_peft_targets()
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=peft_targets["target_modules"],
        lora_dropout=0.05,
        bias="none",
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.print_trainable_parameters()
    policy = policy.to(device)

    # ------------------------------------------------------------------
    # 3. Load LIBERO training dataset
    # ------------------------------------------------------------------
    print("[lora] loading dataset...", flush=True)
    dataset = build_lerobot_dataset(
        cfg["repo_id"], H, cfg["fps"], root=cfg.get("dataset_root")
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # 4. Train LoRA adapters
    # ------------------------------------------------------------------
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    history = []
    step = 0
    policy.train()

    print(f"[lora] training for {args.steps} steps  lr={args.lr}  rank={args.rank}  "
          f"batch={args.batch_size}", flush=True)

    with GpuHourLogger("finetune_lora", base_out, cfg.get("n_gpus", 1)):
        while step < args.steps:
            for raw_batch in loader:
                # Preprocessor: tokenizes language, normalizes obs + action
                batch = preprocessor(
                    {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in raw_batch.items()}
                )
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss, _ = policy.forward(batch)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                scheduler.step()

                if step % 50 == 0:
                    print(f"[lora] step={step:5d}/{args.steps}  "
                          f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
                          flush=True)
                    history.append({"step": step, "loss": float(loss.item()),
                                    "lr": float(scheduler.get_last_lr()[0])})

                step += 1
                if step >= args.steps:
                    break

    # ------------------------------------------------------------------
    # 5. Merge LoRA adapters back into base weights and save
    # ------------------------------------------------------------------
    print("[lora] merging adapters...", flush=True)
    merged = policy.merge_and_unload()
    model_out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(model_out))
    print(f"[lora] merged model saved -> {model_out}", flush=True)

    save_json(
        {"steps": args.steps, "rank": args.rank, "alpha": args.alpha,
         "lr": args.lr, "batch_size": args.batch_size,
         "out": str(model_out), "history": history},
        base_out / "finetune_lora.json",
    )
    print(f"[lora] done. final loss={history[-1]['loss']:.4f}" if history else "[lora] done.")


if __name__ == "__main__":
    main()
