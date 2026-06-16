"""Train RASF (Residual Adaptive Spectral Filter, SIB v2) as a CONSERVATIVE denoiser.

KEY FIX vs the aggressive v2 (which collapsed clean SR to ~28%):
  * OLD target = ground-truth demo chunk  -> filter crushed the clean spectrum
    (g≈0.08, 18% of action energy removed) trying to match the smooth demo.
  * NEW target = the policy's OWN clean prediction A_pred. Identity is now the exact
    optimum on clean input, so the filter has ZERO incentive to touch clean actions;
    it only learns to remove the perturbation we inject. Result: near-identity on
    clean (SR preserved, temporal-ensemble compatible), denoising under noise.

    x_clean = A_pred
    x_noisy = A_pred + eps              (eps: per-sample gaussian + sparse spikes)
    loss = W_CLEAN  * MSE(RASF(x_clean), x_clean)    # identity on clean (DOMINANT)
         + W_DENOISE* MSE(RASF(x_noisy), x_clean)    # strip injected noise
         + W_GAIN   * (1 - g_clean)^2                # keep clean band-gains ≈ all-pass
         + W_GATE   * tanh(gate)^2                   # mild residual regulariser

No ground truth, no jerk penalty. The module is identity at init and the residual is
hard-capped (gate_max), so it cannot start by degrading and cannot over-correct.

    python scripts/train_rasf.py --config configs/rasf_on86.yaml \
        --cache results/sib_on86/chunk_cache.pt --steps 8000 --tag rasf_on86
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from sib.bottleneck import build_module
from sib.utils import load_config, resolve_device, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default=None, help="chunk_cache.pt (default <output_dir>/chunk_cache.pt)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-clean", type=float, default=5.0, help="clean identity-preservation weight (dominant)")
    ap.add_argument("--w-denoise", type=float, default=1.0, help="denoise-to-clean weight")
    ap.add_argument("--w-gain", type=float, default=0.05, help="clean band-gain -> all-pass regulariser")
    ap.add_argument("--w-gate", type=float, default=1e-3, help="residual-strength regulariser")
    ap.add_argument("--noise-std-max", type=float, default=0.30, help="max aug noise std (matches eval sweep)")
    ap.add_argument("--spike-prob", type=float, default=0.1, help="per-sample chance of an impulse spike")
    ap.add_argument("--tag", default="rasf")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = load_config(args.config)
    cfg["module"] = "rasf"
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or str(out_dir / "chunk_cache.pt")

    # ------------------------------------------------------------------ data
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    A = c["A"].float()                              # (N,H,d) policy pred, normalised
    N, H, d = A.shape
    A = A.to(device)
    print(f"[rasf] cache={cache_path}  N={N} H={H} d={d}  "
          f"(target = policy's OWN clean prediction; no ground truth)", flush=True)

    # ------------------------------------------------------------------ model
    module = build_module(cfg, H, d).to(device)
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"[rasf] RASF params={n_params/1e3:.1f}K  gate_max={module.gate_max}  "
          f"gain_floor={module.gain_floor}  gate init strength={module.gate_strength:.3f}", flush=True)

    opt = torch.optim.Adam(module.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    gen = torch.Generator(device=device).manual_seed(cfg["seeds"][0])

    def make_noise(x):
        """Per-sample gaussian (random std up to noise_std_max) + sparse impulse spikes."""
        B = x.shape[0]
        std = torch.rand(B, 1, 1, device=device, generator=gen) * args.noise_std_max
        eps = std * torch.randn(x.shape, device=device, generator=gen)
        spike_mask = (torch.rand(x.shape, device=device, generator=gen) < (args.spike_prob / H))
        eps = eps + spike_mask * torch.randn(x.shape, device=device, generator=gen) * (args.noise_std_max * 3)
        return eps

    # ------------------------------------------------------------------ train
    module.train()
    history = []
    print(f"[rasf] training {args.steps} steps  bs={args.batch_size}  "
          f"W_clean={args.w_clean} W_denoise={args.w_denoise} W_gain={args.w_gain} "
          f"noise_std_max={args.noise_std_max}", flush=True)
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,), device=device, generator=gen)
        Ac = A[idx]                                  # clean target == clean input
        Xn = Ac + make_noise(Ac)                     # noisy input

        out_c = module(Ac)
        out_n = module(Xn)
        Ahc, g_clean = out_c["A_hat"], out_c["gain"]
        Ahn = out_n["A_hat"]

        loss_clean = F.mse_loss(Ahc, Ac)             # preserve clean (target = clean)
        loss_denoise = F.mse_loss(Ahn, Ac)           # remove injected noise
        loss_gain = (1.0 - g_clean).pow(2).mean()    # clean gains -> all-pass
        loss_gate = torch.tanh(module.gate).pow(2).mean()
        loss = (args.w_clean * loss_clean + args.w_denoise * loss_denoise
                + args.w_gain * loss_gain + args.w_gate * loss_gate)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 250 == 0 or step == args.steps - 1:
            with torch.no_grad():
                clean_dev = (Ahc - Ac).pow(2).mean().sqrt().item()   # MUST stay tiny
            print(f"[rasf] step={step:5d}/{args.steps}  loss={loss.item():.4f}  "
                  f"clean_rms|dA|={clean_dev:.4f}  denoise={loss_denoise.item():.4f}  "
                  f"g_clean={g_clean.mean().item():.3f}  gate={module.gate_strength:.3f}", flush=True)
            history.append({"step": step, "loss": float(loss.item()),
                            "clean_dev": clean_dev, "denoise": float(loss_denoise.item()),
                            "g_clean": float(g_clean.mean().item()),
                            "gate_strength": module.gate_strength})

    # ------------------------------------------------------------------ save + validate
    module.eval()
    with torch.no_grad():
        out = module(A)
        clean_rms = (out["A_hat"] - A).pow(2).mean().sqrt().item()
        clean_max = (out["A_hat"] - A).abs().max().item()
        g_mean = out["gain"].mean().item()
        energy_kept = (out["A_hat"].pow(2).mean() / A.pow(2).mean()).item() * 100
    out_path = out_dir / f"{args.tag}.pt"
    torch.save({"module_state": module.state_dict(), "config": cfg,
                "H": H, "d": d, "name": args.tag, "history": history}, out_path)
    strength = (module.gate_max * torch.tanh(module.gate)).detach().cpu().tolist()
    print(f"[rasf] saved -> {out_path}", flush=True)
    print(f"[rasf] === CLEAN-PRESERVATION VALIDATION (full cache) ===", flush=True)
    print(f"[rasf]   rms|A_hat-A| = {clean_rms:.4f}  (want < 0.02)", flush=True)
    print(f"[rasf]   max|A_hat-A| = {clean_max:.4f}", flush=True)
    print(f"[rasf]   mean g_clean = {g_mean:.3f}  (want > 0.90, all-pass)", flush=True)
    print(f"[rasf]   energy kept  = {energy_kept:.1f}%  (want ~100%)", flush=True)
    print(f"[rasf]   per-dim residual strength = {[round(x,3) for x in strength]}", flush=True)
    verdict = "PASS (conservative)" if (clean_rms < 0.02 and g_mean > 0.90) else "FAIL (still aggressive)"
    print(f"[rasf]   VERDICT: {verdict}", flush=True)
    save_json({"steps": args.steps, "tag": args.tag, "out": str(out_path),
               "clean_rms": clean_rms, "clean_max": clean_max, "g_mean": g_mean,
               "energy_kept_pct": energy_kept, "residual_strength": strength,
               "verdict": verdict, "history": history},
              out_dir / f"{args.tag}.train.json")


if __name__ == "__main__":
    main()
