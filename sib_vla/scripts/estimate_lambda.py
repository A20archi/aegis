"""Stage 1 precompute: cache (A, A_star) pairs and estimate source variance.

Runs the frozen SmolVLA policy over the train set *once* (no_grad), caches the
predicted chunks ``A`` (normalised action space) alongside the raw ground-truth
chunks, and estimates the EMA-warm-start ``lambda_k`` = per-(band, dim) variance
of the predicted DCT coefficients.

    python scripts/estimate_lambda.py --config configs/sib.yaml

Outputs (under ``results/``):
    chunk_cache.pt     (A, A_star_raw, action-norm)        -> training data
    lambda.pt          (H, d) source variance              -> warm start
    stage1_lambda.json (per-band energy spectrum summary)
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from sib.data import (action_norm_from_lerobot, build_lerobot_dataset, ChunkCache)
from sib.transforms import SpectralTransform
from sib.utils import GpuHourLogger, load_config, resolve_device, save_json, set_seed
from sib.wrapper import load_smolvla


def _normalization_mode(policy) -> str:
    """Read the action normalisation mode from the policy config.

    LeRobot maps feature types to {MEAN_STD, MIN_MAX, IDENTITY}.  Verify this
    matches your checkpoint; we fall back to mean_std.
    """
    try:
        from lerobot.configs.types import FeatureType
        mode = policy.config.normalization_mapping.get(FeatureType.ACTION)
        name = getattr(mode, "name", str(mode)).lower()
        return "min_max" if "min" in name else "mean_std"
    except Exception:
        return "mean_std"


@torch.no_grad()
def collect_chunks(policy, preprocessor, dataset, n_samples, batch_size, device,
                   collect_context: bool = False):
    """Return ``(A, A_star_raw[, context])`` of shape ``(N, H, d)`` [and ``(N, C)``].

    ``A`` is the policy's predicted chunk (normalised space); ``A_star_raw`` is
    the raw ground-truth chunk captured *before* preprocessing.
    If ``collect_context`` is True, also returns the proprioceptive state vector
    ``observation.state`` (shape ``(N, state_dim)``) for context-conditioned training.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    A_all, star_all, ctx_all, seen = [], [], [], 0
    for batch in loader:
        # Drop chunks that run past the episode end (padded targets corrupt A_star).
        if "action_is_pad" in batch:
            valid = ~batch["action_is_pad"].any(dim=1)
        else:
            valid = torch.ones(batch["action"].shape[0], dtype=torch.bool)
        if not valid.any():
            continue
        raw_action = batch["action"].float()                     # (B, H, d) raw
        proc = preprocessor({k: (v.to(device) if torch.is_tensor(v) else v)
                             for k, v in batch.items()})
        proc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in proc.items()}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            A = policy.predict_action_chunk(proc).float().cpu()  # BF16 compute → FP32 out
        A_all.append(A[valid]); star_all.append(raw_action[valid])
        if collect_context and "observation.state" in batch:
            state = batch["observation.state"].float()           # (B, C)
            ctx_all.append(state[valid])
        seen += int(valid.sum())
        if seen >= n_samples:
            break
    A = torch.cat(A_all)[:n_samples]
    star = torch.cat(star_all)[:n_samples]
    if collect_context and ctx_all:
        ctx = torch.cat(ctx_all)[:n_samples]
        return A, star, ctx
    return A, star


def estimate_lambda(A: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Per-(band, dim) variance of the predicted DCT coefficients, + spectrum."""
    H = A.shape[1]
    transform = SpectralTransform(H)
    X = transform.analyze(A)                                     # (N, H, d)
    lam = X.var(dim=0, unbiased=False)                          # (H, d)
    band_energy = (X ** 2).mean(dim=0).sum(dim=1)               # (H,)
    frac = (band_energy / band_energy.sum()).tolist()
    return lam, {"band_energy_fraction": frac,
                 "cumulative_low3": float(sum(frac[:3]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out = Path(cfg["output_dir"])

    with GpuHourLogger("estimate_lambda", out, cfg.get("n_gpus", 1)):
        policy, preprocessor, _, H, d = load_smolvla(cfg["checkpoint"], str(device))
        lam_cfg = cfg.get("lambda", {})
        ep_filter = None
        if "episode_indices_file" in lam_cfg:
            import json as _json
            ep_filter = _json.load(open(lam_cfg["episode_indices_file"]))
            print(f"[estimate_lambda] filtering to {len(ep_filter)} episodes from {lam_cfg['episode_indices_file']}")
        dataset = build_lerobot_dataset(cfg["repo_id"], H, cfg["fps"],
                                        root=cfg.get("dataset_root"),
                                        episodes=ep_filter)
        need_ctx = cfg.get("sigma_mode", "global") == "context"
        result = collect_chunks(policy, preprocessor, dataset,
                                n_samples=lam_cfg.get("n_samples", 5000),
                                batch_size=cfg.get("batch_size", 32), device=device,
                                collect_context=need_ctx)
        if need_ctx:
            A, star, ctx = result
            print(f"[estimate_lambda] context collected: state shape {tuple(ctx.shape)}")
        else:
            A, star = result
            ctx = None

        norm = action_norm_from_lerobot(dataset.meta.stats["action"],
                                        _normalization_mode(policy))
        ChunkCache(A=A, A_star_raw=star, norm=norm, context=ctx).save(out / "chunk_cache.pt")

        lam, spectrum = estimate_lambda(A)
        torch.save(lam, out / "lambda.pt")
        spectrum["n_samples"] = int(A.shape[0])
        spectrum["H"], spectrum["d"] = int(H), int(d)
        save_json(spectrum, out / "stage1_lambda.json")
    print(f"[estimate_lambda] cached {A.shape[0]} chunks, lambda {tuple(lam.shape)} "
          f"-> {out}; low-3-band energy = {spectrum['cumulative_low3']:.3f}")


if __name__ == "__main__":
    main()
