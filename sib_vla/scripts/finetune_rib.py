"""Fine-tune SmolVLA with the RIB (Robust Information Bottleneck) — AEGIS perception leg.

Unlike finetune_ib.py (StableVLA IB, trained on the clean task loss -> dormant), RIB
is trained by CORRUPTION-AUGMENTED CONSISTENCY: a random fraction of each batch's
images is corrupted (photometric + geometric, approximating the LIBERO-V axes) and
the model must still predict the GT action. The bottleneck therefore learns to drop
the corruption-induced nuisance subspace -> visual robustness. Clean samples in the
same batch preserve clean SR; an explicit beta*KL rate term gives the IB objective
that StableVLA lacks.

    L = task_loss(GT | images[corrupted-frac]) + beta * KL(q(z|x)||N(0,I))

Only RIB (~2.3M) + fusion_coeff + lm_expert (action head) train; everything else frozen.

    python scripts/finetune_rib.py --config configs/ib_on86.yaml --steps 12000 \
        --lr-rib 3e-4 --lr-head 2e-5 --d-z 448 --beta 1e-3 --corrupt-frac 0.6 --tag rib_on86
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sib import corruptions as corr
from sib.data import build_lerobot_dataset
from sib.robust_ib import inject_fused_rib, save_rib_checkpoint
from sib.utils import GpuHourLogger, load_config, resolve_device, save_json, set_seed

# photometric / sensor corruptions available as image-space augmentation
_PHOTO = ["gaussian_noise", "gaussian_blur", "motion_blur", "fog", "glass_blur",
          "brightness_contrast"]


def _augment_images(img: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random corruption of a (B,C,H,W) float image batch in [0,1] (in-place-safe copy).

    Mixes one photometric corruption (approx. lighting/blur/sensor-noise/fog axes) with
    an optional geometric warp (random resized crop + shift, approximating viewpoint).
    """
    x = img.clone()
    B = x.shape[0]
    dev = x.device
    # 1) photometric: random family, severity sampled {0,1,2} centered on the EVAL severity
    #    (1) and clamped to the family's range — span, don't overfit, the test point.
    #    (severity=0 alone is too mild: it barely moves the flow loss -> no learning signal.)
    fam = _PHOTO[int(torch.randint(0, len(_PHOTO), (1,), generator=gen, device=gen.device).item())]
    if fam == "gaussian_noise":
        # denoiser must see the FULL eval noise sweep incl HIGH severity: std 0.05..0.70
        # (severities 0-5; skip 0.75/1.0 — image destroyed, nothing to learn to recover).
        sev = int(torch.randint(0, 6, (1,), generator=gen, device=gen.device).item())
    else:
        r = torch.rand(1, generator=gen, device=gen.device).item()
        sev = 0 if r < 0.2 else (1 if r < 0.7 else 2)
    sev = min(sev, len(corr.SEVERITY[fam]) - 1)
    try:
        x = corr.apply(x, fam, severity=sev, generator=gen)
    except Exception:
        pass
    # 2) geometric (approx viewpoint — the headline axis): zoom-crop + translate, prob 0.6,
    #    wider scale 0.75..1.15 so the bottleneck sees stronger spatial shifts.
    if torch.rand(1, generator=gen, device=gen.device).item() < 0.6:
        H, W = x.shape[-2:]
        scale = 0.75 + 0.40 * torch.rand(1, generator=gen, device=gen.device).item()   # 0.75..1.15
        nh, nw = max(8, int(H * scale)), max(8, int(W * scale))
        xr = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        # center-pad or center-crop back to (H,W)
        out = torch.zeros_like(x)
        dh, dw = (H - nh) // 2, (W - nw) // 2
        sy = int((torch.rand(1, generator=gen, device=gen.device).item() - 0.5) * 0.16 * H)  # shift
        sx = int((torch.rand(1, generator=gen, device=gen.device).item() - 0.5) * 0.16 * W)
        if scale <= 1.0:
            y0, x0 = max(0, dh + sy), max(0, dw + sx)
            out[..., y0:y0 + nh, x0:x0 + nw] = xr[..., :H - y0, :W - x0]
            x = out
        else:
            y0, x0 = (nh - H) // 2, (nw - W) // 2
            x = xr[..., y0:y0 + H, x0:x0 + W]
    return x.clamp(0.0, 1.0)


def _image_keys(batch: dict) -> list[str]:
    return [k for k, v in batch.items()
            if k.startswith("observation.images") and torch.is_tensor(v) and v.dim() == 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr-rib", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-z", type=int, default=448, help="RIB bottleneck dim (capacity knob, <3M)")
    ap.add_argument("--n-heads", type=int, default=7)
    ap.add_argument("--beta", type=float, default=1e-3, help="KL rate weight (VIB)")
    ap.add_argument("--free-bits", type=float, default=0.1,
                    help="per-elem KL floor: no compression pressure below this (anti-collapse)")
    ap.add_argument("--corrupt-frac", type=float, default=0.6, help="fraction of batch images corrupted")
    ap.add_argument("--lambda-gate", type=float, default=0.05,
                    help="weight on the ADVANTAGE-gate BCE (open where RIB beats baseline, else close)")
    ap.add_argument("--adv-margin", type=float, default=0.0,
                    help="advantage gate: target=open(1) only if baseline_loss - rib_loss > margin "
                         "(RIB must BEAT baseline by this much; else gate closes -> emit baseline). "
                         ">0 biases toward safety: don't act unless clearly better than the base.")
    ap.add_argument("--lambda-cons", type=float, default=0.0,
                    help="weight on self-referential feature-consistency (RIB(corrupt)->clean feats)")
    ap.add_argument("--gate-warmup", type=int, default=0,
                    help="if >0: gate_floor=1 for [0,W) (decoder warmup, no dead-start), anneal 1->0 "
                         "over [W,2W), then 0 (per-input gate learnable). The consistency loss teaches "
                         "the gate to open where the correction helps, close where it doesn't.")
    ap.add_argument("--tag", default="rib_on86")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{args.tag}.pt")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print("[rib] loading SmolVLA ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg["checkpoint"]).to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, cfg["checkpoint"])

    fused = inject_fused_rib(policy, D_out=960, d_z=args.d_z, n_heads=args.n_heads)
    policy.to(device)

    for p in policy.parameters():
        p.requires_grad_(False)
    rib_params = (list(fused.rib.parameters()) + list(fused.gate_head.parameters())
                  + [fused.fusion_coeff])
    for p in rib_params:
        p.requires_grad_(True)
    lm_expert = policy.model.vlm_with_expert.lm_expert
    for p in lm_expert.parameters():
        p.requires_grad_(True)

    n_rib = sum(p.numel() for p in rib_params)
    print(f"[rib] RIB {n_rib/1e6:.2f}M @ lr={args.lr_rib:.1e}  "
          f"lm_expert @ lr={args.lr_head:.1e}  beta={args.beta}  corrupt_frac={args.corrupt_frac}", flush=True)

    H = policy.config.chunk_size
    print("[rib] loading dataset ...", flush=True)
    dataset = build_lerobot_dataset(cfg["repo_id"], H, cfg["fps"], root=cfg.get("dataset_root"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)

    opt = torch.optim.AdamW(
        [{"params": rib_params, "lr": args.lr_rib},
         {"params": list(lm_expert.parameters()), "lr": args.lr_head}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=min(args.lr_rib, args.lr_head) * 0.1)
    gen = torch.Generator(device=device).manual_seed(cfg["seeds"][0])

    history, step = [], 0
    policy.train()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    print(f"[rib] training {args.steps} steps  bs={args.batch_size}", flush=True)

    with GpuHourLogger("finetune_rib", out_dir, cfg.get("n_gpus", 1)):
        while step < args.steps:
            for raw_batch in loader:
                raw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw_batch.items()}
                B = next(v.shape[0] for v in raw.values() if torch.is_tensor(v) and v.dim() >= 1)
                # GATE WARMUP: decoder learns first (floor=1), then the per-input gate becomes
                # learnable (floor->0) so it can't shut before the decoder is useful (no dead-start).
                if args.gate_warmup > 0:
                    W = args.gate_warmup
                    fused.gate_floor = (1.0 if step < W else
                                        max(0.0, 1.0 - (step - W) / W) if step < 2 * W else 0.0)
                # CLEAN batch (built BEFORE corruption) -> source of the denoising target features
                clean_batch = preprocessor({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()})
                clean_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in clean_batch.items()}
                # CORRUPT batch: VIEW-ASYMMETRIC (LIBERO-Plus: lighting wrecks the third-person view,
                # the wrist stays stable). Corrupt the AGENTVIEW only, keep the WRIST clean.
                cmask = torch.rand(B, generator=gen, device=device) < args.corrupt_frac
                WRIST_HINTS = ("image2", "wrist", "eye_in_hand")
                for k in _image_keys(raw):
                    if any(h in k for h in WRIST_HINTS):
                        continue                       # keep wrist view clean
                    if cmask.any():
                        sub = raw[k][cmask]
                        raw[k] = raw[k].clone()
                        raw[k][cmask] = _augment_images(sub, gen).to(raw[k].dtype)
                batch = preprocessor({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()})
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                # CLEAN target features (no grad): per-view PRE-RIB linear output on clean images.
                # MUST run under autocast (same as the main forward) or dtypes mismatch.
                with torch.no_grad(), amp_ctx:
                    fused.zmlp_trace.clear()
                    policy.forward(clean_batch)
                    clean_zmlp = [z.detach() for z in fused.zmlp_trace]

                # ADVANTAGE GATE: once the gate is learnable (floor<1), probe the SAME corrupted batch
                # with RIB OFF (g=0, baseline) and RIB FULLY ON (g=1) under ONE SHARED flow-matching
                # noise/time draw, per sample. target=1 where RIB BEATS the baseline (by adv_margin),
                # else 0. The gate is BCE-supervised to open only where it helps and CLOSE (-> emit the
                # baseline value) elsewhere — the honest, per-input "AEGIS >= baseline" mechanism.
                gate_target = None; adv_open = float("nan"); adv_mean = float("nan")
                do_adv = args.lambda_gate > 0 and fused.gate_floor < 1.0
                if do_adv:
                    acts_shape = (B, policy.config.chunk_size, policy.config.max_action_dim)
                    noise = policy.model.sample_noise(acts_shape, device)
                    time = policy.model.sample_time(B, device)
                    with torch.no_grad(), amp_ctx:
                        fused.gate_override = 0.0
                        L_base, _ = policy.forward(batch, noise=noise, time=time, reduction="none")
                        fused.gate_override = 1.0
                        L_rib, _ = policy.forward(batch, noise=noise, time=time, reduction="none")
                        fused.gate_override = None
                        adv = (L_base.float() - L_rib.float())          # >0 => RIB helps THIS sample
                        gate_target = (adv > args.adv_margin).float()    # (B,) 1=open, 0=stay baseline
                        adv_open = float(gate_target.mean()); adv_mean = float(adv.mean())
                else:
                    noise = time = None

                fused.gate_trace.clear(); fused.out_trace.clear()
                fused.gate_raw_trace.clear(); fused.gate_sample_trace.clear()
                fused.gate_logit_trace.clear()
                fused.gate_override = None
                with amp_ctx:
                    task_loss, _ = policy.forward(batch, noise=noise, time=time)
                    kl = fused.last_kl.float() if fused.last_kl is not None else torch.zeros((), device=device)
                    kl_pen = torch.clamp(kl, min=args.free_bits)   # free-bits: no pressure below floor
                    # (1) SELF-REFERENTIAL FEATURE CONSISTENCY — the direct fix for the flat task loss.
                    # RIB's per-view output on the CORRUPTED images is pulled toward the clean pre-RIB
                    # features: denoises the corrupted agentview toward its clean target, and reinforces
                    # pass-through on the clean wrist. Gives RIB a denoising gradient that does NOT
                    # depend on whether the action loss needs it (the wrist crutch can't starve it).
                    outs = fused.out_trace
                    if outs and clean_zmlp and len(outs) == len(clean_zmlp):
                        cons = sum(F.mse_loss(o.float(), t.float())
                                   for o, t in zip(outs, clean_zmlp)) / len(outs)
                    else:
                        cons = torch.zeros((), device=device)
                    # (2) ADVANTAGE-GATE BCE — supervise the PRE-FLOOR sigmoid (== deployed gate at
                    # floor=0) per view toward the per-sample target. Gradient reaches ONLY gate_head
                    # (everything upstream is frozen), so it trains the gate without disturbing RIB.
                    if do_adv and gate_target is not None and fused.gate_logit_trace:
                        # BCE-with-logits (autocast-safe) on the per-view pre-sigmoid gate logits
                        gate_loss = sum(F.binary_cross_entropy_with_logits(l.float(), gate_target)
                                        for l in fused.gate_logit_trace) / len(fused.gate_logit_trace)
                        gate_mode = "advantage"
                    else:
                        gate_loss = torch.zeros((), device=device)
                        gate_mode = "warmup"
                    loss = (task_loss + args.beta * kl_pen
                            + args.lambda_gate * gate_loss + args.lambda_cons * cons)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()

                if step % 100 == 0:
                    # per-view gate means: [agentview, wrist] in embed order. We WANT agentview to
                    # stay open under corruption and wrist to close (lean on the stable wrist view).
                    gv = [round(float(x), 3) for x in fused.gate_trace]
                    print(f"[rib] step={step:6d}/{args.steps}  loss={loss.item():.4f}  "
                          f"task={task_loss.item():.4f}  kl={kl.item():.3f}  cons={float(cons):.4f}  "
                          f"gloss={float(gate_loss):.4f}({gate_mode})  open={adv_open:.2f}  adv={adv_mean:+.4f}  "
                          f"floor={fused.gate_floor:.2f}  gate_views={gv}  "
                          f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)
                    history.append({"step": step, "loss": float(loss.item()),
                                    "task": float(task_loss.item()), "kl": float(kl.item()),
                                    "fusion_coeff": float(fused.fusion_coeff.item())})
                if step % 3000 == 0 and step > 0:   # periodic ckpt -> can eval before full run ends
                    policy.eval()
                    save_rib_checkpoint(fused, lm_expert, out_path,
                                        config={"d_z": args.d_z, "n_heads": args.n_heads, "D_out": 960,
                                                "base_checkpoint": cfg["checkpoint"]},
                                        step=step, loss=float(loss.item()))
                    policy.train()
                    print(f"[rib] >>> checkpoint saved at step {step} -> {out_path}", flush=True)
                step += 1
                if step >= args.steps:
                    break

    policy.eval()
    save_rib_checkpoint(fused, lm_expert, out_path,
                        config={"d_z": args.d_z, "n_heads": args.n_heads, "D_out": 960,
                                "base_checkpoint": cfg["checkpoint"]},
                        step=step, loss=history[-1]["loss"] if history else float("nan"))
    print(f"[rib] saved -> {out_path}", flush=True)
    print(f"[rib] FINAL fusion_coeff = {fused.ib_contribution:+.3f}  "
          f"({'ENGAGED' if abs(fused.ib_contribution) > 0.05 else 'still dormant!'})", flush=True)
    save_json({"steps": args.steps, "lr_rib": args.lr_rib, "beta": args.beta,
               "corrupt_frac": args.corrupt_frac, "d_z": args.d_z, "out": out_path,
               "final_fusion_coeff": fused.ib_contribution, "history": history},
              out_dir / f"{args.tag}.train.json")


if __name__ == "__main__":
    main()
