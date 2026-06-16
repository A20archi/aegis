"""Fine-tune SmolVLA with the Fused IB-Adapter in the visual connector.

IB-Adapter (StableVLA) inserts a channel-wise information bottleneck adapter
into the visual projector (SmolVLMConnector) of SmolVLA. The Fused design:
    Z = MLP(X) + tanh(λ) * IB-Adapter(MLP(X))

Only the IB-Adapter params (~350K) + fusion_coeff + lm_expert (action head)
are trained.  Vision encoder and LLM backbone are frozen.

Usage:
    python scripts/finetune_ib.py --config configs/ib_adapter.yaml \\
        --steps 10000 --lr-ib 2e-4 --lr-head 2e-5 --tag smolvla_ib
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sib.data import build_lerobot_dataset
from sib.ib_adapter import inject_fused_ib_adapter, save_ib_checkpoint
from sib.utils import GpuHourLogger, load_config, resolve_device, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr-ib",   type=float, default=2e-4,
                    help="LR for IB-Adapter params + fusion_coeff")
    ap.add_argument("--lr-head", type=float, default=2e-5,
                    help="LR for lm_expert (action head)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-heads",    type=int, default=8,
                    help="Number of IB-Adapter heads (D=960 → d=120)")
    ap.add_argument("--tag", default="smolvla_ib",
                    help="Output filename stem under output_dir/")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{args.tag}.pt")

    # ------------------------------------------------------------------
    # 1. Load SmolVLA and inject Fused IB-Adapter into connector
    # ------------------------------------------------------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print("[ib] loading SmolVLA ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg["checkpoint"]).to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, cfg["checkpoint"])

    print("[ib] injecting Fused IB-Adapter into connector ...", flush=True)
    fused = inject_fused_ib_adapter(policy, D_out=960, n_heads=args.n_heads)
    policy.to(device)

    # ------------------------------------------------------------------
    # 2. Parameter groups
    #    IB params + fusion_coeff : lr_ib  (new)
    #    lm_expert (action head)  : lr_head (adapt to IB-filtered features)
    #    Everything else          : frozen
    # ------------------------------------------------------------------
    for p in policy.parameters():
        p.requires_grad_(False)

    # IB-Adapter params (new) — must train
    ib_params = list(fused.ib.parameters()) + [fused.fusion_coeff]
    for p in ib_params:
        p.requires_grad_(True)

    # Action head — fine-tune to adapt to new connector features
    lm_expert = policy.model.vlm_with_expert.lm_expert
    for p in lm_expert.parameters():
        p.requires_grad_(True)

    n_ib   = sum(p.numel() for p in ib_params)
    n_head = sum(p.numel() for p in lm_expert.parameters())
    print(f"[ib] IB-Adapter {n_ib/1e3:.1f}K @ lr={args.lr_ib:.1e}  "
          f"lm_expert {n_head/1e6:.1f}M @ lr={args.lr_head:.1e}", flush=True)
    print(f"[ib] fusion_coeff init = {fused.ib_contribution:.3f}  "
          f"(tanh(0)=0 → starts as vanilla SmolVLA)", flush=True)

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    H = policy.config.chunk_size     # 50
    print("[ib] loading dataset ...", flush=True)
    dataset = build_lerobot_dataset(cfg["repo_id"], H, cfg["fps"],
                                    root=cfg.get("dataset_root"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, drop_last=True, pin_memory=True,
                        persistent_workers=True)

    # ------------------------------------------------------------------
    # 4. Optimiser
    # ------------------------------------------------------------------
    opt = torch.optim.AdamW(
        [
            {"params": ib_params,                  "lr": args.lr_ib},
            {"params": list(lm_expert.parameters()), "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=min(args.lr_ib, args.lr_head) * 0.1)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history, step = [], 0
    policy.train()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    print(f"[ib] training {args.steps} steps  batch={args.batch_size}  "
          f"n_heads={args.n_heads}", flush=True)

    with GpuHourLogger("finetune_ib", out_dir, cfg.get("n_gpus", 1)):
        while step < args.steps:
            for raw_batch in loader:
                batch = preprocessor(
                    {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in raw_batch.items()}
                )
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

                with amp_ctx:
                    loss, _ = policy.forward(batch)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()

                if step % 100 == 0:
                    lrs = [g["lr"] for g in opt.param_groups]
                    print(f"[ib] step={step:6d}/{args.steps}  "
                          f"loss={loss.item():.4f}  "
                          f"coeff={fused.ib_contribution:.3f}  "
                          f"lr_ib={lrs[0]:.1e} lr_hd={lrs[1]:.1e}",
                          flush=True)
                    history.append({
                        "step": step, "loss": float(loss.item()),
                        "fusion_coeff": float(fused.fusion_coeff.item()),
                        "ib_contribution": fused.ib_contribution,
                    })
                step += 1
                if step >= args.steps:
                    break

    # ------------------------------------------------------------------
    # 6. Save IB checkpoint (IB weights + adapted action head)
    # ------------------------------------------------------------------
    policy.eval()
    save_ib_checkpoint(
        fused, lm_expert, out_path,
        config={"n_heads": args.n_heads, "D_out": 960,
                "base_checkpoint": cfg["checkpoint"]},
        step=step,
        loss=history[-1]["loss"] if history else float("nan"),
    )
    print(f"[ib] saved -> {out_path}", flush=True)

    save_json(
        {"steps": args.steps, "lr_ib": args.lr_ib, "lr_head": args.lr_head,
         "n_heads": args.n_heads, "batch_size": args.batch_size,
         "out": out_path, "history": history},
        out_dir / "finetune_ib.json",
    )
    print(f"[ib] done. final loss={history[-1]['loss']:.4f}  "
          f"fusion_coeff={fused.ib_contribution:.3f}" if history else "[ib] done.")


if __name__ == "__main__":
    main()
