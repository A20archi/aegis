"""Train the spectral bottleneck on cached (A, A_star) pairs.

The frozen policy is *not* in this loop: its outputs were cached once by
``estimate_lambda.py``.  We optimise only the bottleneck against the ground-truth
chunks, which is identical to training end-to-end with the sampler under no_grad
(the sampler has no parameters to update and never receives gradient).

    python scripts/train.py --config configs/sib.yaml
    python scripts/train.py --config configs/sib.yaml --beta 1e-3 --tag sib_b1e-3

Outputs: ``results/<name>.pt`` (module weights + resolved config + final lambda
+ bits-per-band heatmap) and ``results/<name>.train.json`` (loss curve summary).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sib.bottleneck import build_module
from sib.data import CachedPairDataset, ChunkCache
from sib.losses import total_loss
from sib.utils import (GpuHourLogger, load_config, resolve_device, save_json,
                       save_resolved_config, set_seed)


def build_and_warm_module(cfg, H, d, out_dir: Path, device):
    """Construct the module and apply warm-start lambda / PCA rotation if present."""
    module = build_module(cfg, H, d).to(device)
    if module.uses_rate and (out_dir / "lambda.pt").exists():
        module.lambda_estimator.load_estimate(torch.load(out_dir / "lambda.pt"))
    if cfg.get("rotation") == "pca":
        rot_path = out_dir / "pca_rotation.pt"
        if not rot_path.exists():
            raise FileNotFoundError(f"rotation=pca but {rot_path} missing; run decorrelation_check")
        module.transform.set_pca_rotation(torch.load(rot_path).to(device))
    return module


def run_name(cfg, args) -> str:
    if args.tag:
        return args.tag
    base = cfg.get("name", Path(args.config).stem)
    return f"{base}_beta{cfg['beta']}" if cfg["module"] in ("sib", "raw_vib") else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--beta", type=float, default=None, help="override cfg.beta (sweep)")
    ap.add_argument("--tag", default=None, help="override the run name")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    if args.beta is not None:
        cfg["beta"] = args.beta
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    set_seed(seed)
    cfg["seed_used"] = seed
    device = resolve_device(cfg)
    out = Path(cfg["output_dir"])
    name = run_name(cfg, args)

    cache = ChunkCache.load(out / "chunk_cache.pt")
    ds = CachedPairDataset(cache)
    H, d = cache.A.shape[1], cache.A.shape[2]
    tcfg = cfg.get("train", {})
    loader = DataLoader(ds, batch_size=tcfg.get("batch_size", 256),
                        shuffle=True, drop_last=True)

    module = build_and_warm_module(cfg, H, d, out, device)
    has_params = any(p.requires_grad for p in module.parameters())

    history = []
    with GpuHourLogger(f"train_{name}", out, cfg.get("n_gpus", 1)):
        if has_params:
            module.train()
            opt = torch.optim.Adam(module.parameters(), lr=tcfg.get("lr", 1e-3))
            total_steps = tcfg.get("steps", 30000)
            step = 0
            while step < total_steps:
                for batch in loader:
                    A = batch[0].to(device)
                    A_star = batch[1].to(device)
                    context = batch[2].to(device) if len(batch) > 2 else None
                    # Noise-augmented (denoiser) training: inject action-space noise on
                    # the INPUT chunk with a per-sample std ~ U(0, train_action_noise),
                    # keep the clean A_star as target. Teaches the input-adaptive sigma
                    # to estimate and suppress action noise across severities.
                    tn = float(cfg.get("train_action_noise", 0.0))
                    if tn > 0.0:
                        s = torch.rand(A.shape[0], 1, 1, device=device) * tn
                        A = A + torch.randn_like(A) * s
                    out_d = module(A, context=context)
                    loss, parts = total_loss(cfg, out_d, A_star)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(module.parameters(),
                                                   tcfg.get("grad_clip", 1.0))
                    opt.step()
                    if step % tcfg.get("log_every", 500) == 0:
                        history.append({"step": step,
                                        **{k: float(v) for k, v in parts.items()}})
                    step += 1
                    if step >= total_steps:
                        break
        else:  # lowpass: no parameters, nothing to optimise
            module.eval()

    # Persist module + provenance.
    bpb = module.bits_per_band()
    torch.save({"module_state": module.state_dict(),
                "config": cfg, "name": name, "H": H, "d": d,
                "lambda": module.lambda_estimator.value.cpu() if module.uses_rate else None,
                "bits_per_band": None if bpb is None else bpb.cpu()},
               out / f"{name}.pt")
    save_resolved_config(cfg, out / f"{name}.config.yaml")
    save_json({"name": name, "seed": seed, "history": history,
               "final": history[-1] if history else {}}, out / f"{name}.train.json")
    print(f"[train] {name}: {len(history)} logged steps; saved -> {out / (name + '.pt')}")


if __name__ == "__main__":
    main()
