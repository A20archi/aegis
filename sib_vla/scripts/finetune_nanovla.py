"""Train NanoVLA-S baseline (language-conditioned ACT) on LIBERO.

NanoVLA-S = ACT (ResNet18 + transformer enc/dec + CVAE + action-chunk regression)
+ frozen BERT instruction token (via ACT's env_state slot). Vision backbone + BERT
frozen; only the ~73M enc/dec/heads train. We normalize state/action ourselves
(ACT config uses IDENTITY) and save the stats with the checkpoint for eval.

    python scripts/finetune_nanovla.py --repo-id HuggingFaceVLA/libero \
        --steps 100000 --batch-size 64 --lr 1e-4 --out results/nanovla_s
    python scripts/finetune_nanovla.py --smoke      # 3-step real-data smoke
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.utils.constants import OBS_STATE, OBS_IMAGES, ACTION
from sib.data import build_lerobot_dataset, action_norm_from_lerobot
from sib.utils import GpuHourLogger, save_json, set_seed
from nanovla.modeling_nanovla import NanoVLAS, make_nanovla_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--chunk-size", type=int, default=100)   # NanoVLA H_train
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--image-key", default="observation.images.image")  # single 3rd-person cam
    ap.add_argument("--lang-model", default="google-bert/bert-base-uncased")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--out", default="results/nanovla_s")
    ap.add_argument("--smoke", action="store_true", help="3 steps on real data, then exit")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.smoke:
        args.steps, args.batch_size, args.num_workers = 3, 4, 2
        args.lang_model = "distilbert-base-uncased"   # cached; avoids a download in smoke
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("[nanovla] loading dataset ...", flush=True)
    ds = build_lerobot_dataset(args.repo_id, args.chunk_size, args.fps)
    stats = ds.meta.stats
    action_norm = action_norm_from_lerobot(stats[ACTION], "mean_std")
    state_norm = action_norm_from_lerobot(stats[OBS_STATE], "mean_std")
    action_norm.a = action_norm.a.to(device); action_norm.b = action_norm.b.to(device)
    state_norm.a = state_norm.a.to(device); state_norm.b = state_norm.b.to(device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    cfg = make_nanovla_config(state_dim=stats[OBS_STATE]["mean"].shape[-1] if hasattr(stats[OBS_STATE]["mean"], "shape") else 8,
                              action_dim=stats[ACTION]["mean"].shape[-1] if hasattr(stats[ACTION]["mean"], "shape") else 7,
                              image_keys=(args.image_key,), chunk_size=args.chunk_size,
                              n_action_steps=args.n_action_steps)
    model = NanoVLAS(cfg, lang_model=args.lang_model).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable) / 1e6
    print(f"[nanovla] trainable params: {n_tr:.1f}M  | chunk={args.chunk_size} exec={args.n_action_steps}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

    def to_dev(x):
        return x.to(device) if torch.is_tensor(x) else x

    history, step = [], 0
    model.train()
    with GpuHourLogger("finetune_nanovla", out, 1):
        while step < args.steps:
            for raw in loader:
                img = to_dev(raw[args.image_key]).float()
                if img.dim() == 4 and img.shape[-1] == 3:      # HWC -> CHW
                    img = img.permute(0, 3, 1, 2)
                if img.max() > 1.5:                            # uint8 -> [0,1]
                    img = img / 255.0
                state = state_norm.normalize(to_dev(raw[OBS_STATE]).float())
                action = action_norm.normalize(to_dev(raw[ACTION]).float())
                batch = {OBS_STATE: state, OBS_IMAGES: [img], ACTION: action,
                         "task": raw["task"]}
                if "action_is_pad" in raw:
                    batch["action_is_pad"] = to_dev(raw["action_is_pad"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss, parts = model(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step()
                if step % (1 if args.smoke else 200) == 0:
                    print(f"[nanovla] step {step:6d}/{args.steps}  loss={parts['loss']:.4f} "
                          f"l1={parts['l1']:.4f} kld={parts.get('kld', 0):.4f} lr={sched.get_last_lr()[0]:.2e}",
                          flush=True)
                    history.append({"step": step, **parts})
                step += 1
                if step >= args.steps:
                    break

    if args.smoke:
        print("[nanovla] SMOKE OK (real-data train path works)"); return

    torch.save({"model_state": model.state_dict(), "config": cfg,
                "action_norm": action_norm.state_dict(), "state_norm": state_norm.state_dict(),
                "image_key": args.image_key, "lang_model": args.lang_model,
                "chunk_size": args.chunk_size, "n_action_steps": args.n_action_steps},
               out / "nanovla_s.pt")
    save_json({"steps": args.steps, "lr": args.lr, "batch_size": args.batch_size,
               "trainable_M": n_tr, "history": history}, out / "finetune_nanovla.json")
    print(f"[nanovla] saved -> {out / 'nanovla_s.pt'}")


if __name__ == "__main__":
    main()
