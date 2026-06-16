"""Evaluate a policy on the LIBERO-V (Visual) robustness benchmark — the 4-axis
sweep (camera viewpoint, lighting, background texture, sensor noise) from "VLA
Models Are More Generalizable Than You Think".

Unifies all three method families under eval.py's rollout_task:
  * vanilla : SIBPolicy(policy, identity)              (== eval.py vanilla)
  * sib     : SIBPolicy(policy, trained spectral mod)  (--weights <sib>.pt)
  * ib      : inject Fused IB-Adapter into the connector + load weights, then
              SIBPolicy(policy, identity) — the IB filtering is transparent to
              predict_action_chunk, so the same rollout path applies.

Sim axes (viewpoint/lighting/texture) are applied via sib.libero_v's reset
monkeypatch; the sensor-noise axis is applied via eval's image-corruption path.

    python scripts/eval_libero_v.py --config configs/ib_adapter_repro.yaml \
        --method ib --weights results/ib_repro/smolvla_ib_repro.pt --n-heads 8 \
        --n-action-steps 1 --episodes 10

Results -> <output_dir>/libero_v/<method>/eval_<condition>.json  (+ a summary).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from sib import libero_v as lv
from sib.bottleneck import SpectralActionModule
from sib.ib_adapter import inject_fused_ib_adapter, load_ib_checkpoint
from sib.robust_ib import inject_fused_rib, load_rib_checkpoint
from sib.metrics import success_rate, wilson_ci
from sib.utils import load_config, resolve_device, save_json, set_seed
from sib.wrapper import ForgeActionHeadPolicy, SIBPolicy, load_smolvla

# reuse eval.py's rollout + trained-module loader (load by path; avoids dup)
_eval_path = _pathlib.Path(__file__).resolve().parent / "eval.py"
_spec = importlib.util.spec_from_file_location("sib_eval_mod", _eval_path)
_eval = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_eval)
rollout_task = _eval.rollout_task
load_trained_module = _eval.load_trained_module


def build_policy(args, cfg, device):
    """Build the eval policy for any method, optionally with temporal ensembling.

    Loci:
      perception -> optionally inject RIB (aegis) or StableVLA IB-Adapter (ib)
      action     -> spectral module: RASF/SIB (--rasf-weights/--weights) or identity
      receding   -> ForgeActionHeadPolicy (temporal ensembling) if --forge-ensemble,
                    else plain SIBPolicy.
    Methods: vanilla | sib | ib | baseline(=SmolVLA, usually +TE) | aegis(=RIB+RASF+TE)
    """
    method = args.method
    policy, preprocessor, postprocessor, H, d = load_smolvla(cfg["checkpoint"], str(device))

    # --- perception locus ------------------------------------------------
    if method == "aegis":
        if args.rib_weights is None:
            raise ValueError("--method aegis requires --rib-weights (RIB checkpoint)")
        rc = torch.load(args.rib_weights, map_location="cpu", weights_only=False).get("config", {})
        inject_fused_rib(policy, D_out=rc.get("D_out", 960),
                         d_z=rc.get("d_z", 512), n_heads=rc.get("n_heads", 8))
        policy.to(device)
        load_rib_checkpoint(policy, args.rib_weights, device=str(device))
        # de-strength override: scale RIB fusion_coeff (0 => RIB off, for RASF-only ablation)
        if args.rib_fusion_scale != 1.0:
            n=0
            for m in policy.modules():
                if hasattr(m, "fusion_coeff") and isinstance(getattr(m, "fusion_coeff"), torch.nn.Parameter):
                    with torch.no_grad(): m.fusion_coeff.mul_(args.rib_fusion_scale); n+=1
            print(f"[aegis] RIB fusion_coeff scaled x{args.rib_fusion_scale} on {n} module(s)", flush=True)
        policy.eval()
    elif method == "ib":
        if args.weights is None:
            raise ValueError("--method ib requires --weights (IB checkpoint)")
        inject_fused_ib_adapter(policy, D_out=cfg.get("ib_adapter", {}).get("D_out", 960),
                                n_heads=args.n_heads)
        policy.to(device)
        load_ib_checkpoint(policy, args.weights, device=str(device))
        policy.eval()

    # --- action locus ----------------------------------------------------
    act_w = args.rasf_weights if method == "aegis" else (args.weights if method == "sib" else None)
    if act_w is not None:
        module = load_trained_module(act_w, H, d, device)
        # de-strength overrides: dial the RASF filter toward identity without retraining
        for m in module.modules() if hasattr(module, "modules") else [module]:
            if hasattr(m, "gate_max") and args.rasf_gate_max is not None:
                m.gate_max = float(args.rasf_gate_max)
            if hasattr(m, "gain_floor") and args.rasf_gain_floor is not None:
                m.gain_floor = float(args.rasf_gain_floor)
        if args.rasf_gate_max is not None or args.rasf_gain_floor is not None:
            print(f"[aegis] RASF override gate_max={args.rasf_gate_max} gain_floor={args.rasf_gain_floor}", flush=True)
    else:
        module = SpectralActionModule("gain_no_rate", H, d).to(device).eval()   # identity

    # --- receding horizon: temporal ensembling or plain ------------------
    if args.forge_ensemble:
        sib = ForgeActionHeadPolicy(policy, module, n_action_steps=args.n_action_steps,
                                    ensemble_coeff=args.ensemble_coeff, lever3_clamp=1.0).to(device)
    else:
        sib = SIBPolicy(policy, module, n_action_steps=args.n_action_steps).to(device)
    return policy, sib, preprocessor, postprocessor, H, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["vanilla", "sib", "ib", "baseline", "aegis"], required=True)
    ap.add_argument("--weights", default=None, help="sib/ib checkpoint")
    ap.add_argument("--rib-weights", default=None, help="RIB checkpoint (perception leg, method=aegis)")
    ap.add_argument("--rasf-weights", default=None, help="RASF module .pt (action leg, method=aegis)")
    ap.add_argument("--forge-ensemble", action="store_true", help="temporal ensembling (TE)")
    ap.add_argument("--ensemble-coeff", type=float, default=0.01)
    # --- inference-time de-strength overrides (NO retraining) for surgical ablation ---
    ap.add_argument("--rasf-gate-max", type=float, default=None,
                    help="override RASF gate_max at eval (trained=0.95). Lower => gentler action filter; 0 => identity")
    ap.add_argument("--rasf-gain-floor", type=float, default=None,
                    help="override RASF gain_floor at eval (trained=0.05). Higher => less band attenuation")
    ap.add_argument("--rib-fusion-scale", type=float, default=1.0,
                    help="multiply RIB fusion_coeff at eval (1=trained, 0=RIB off => RASF-only ablation)")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-action-steps", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=10, help="episodes/task per condition")
    ap.add_argument("--tasks", default=None, help="comma task ids; default all 10")
    ap.add_argument("--full", action="store_true", help="full grid (more noise/light/texture severities)")
    ap.add_argument("--only", default=None, help="comma condition labels to restrict to")
    ap.add_argument("--record", action="store_true", help="save rollout videos per condition")
    ap.add_argument("--videos-per-task", type=int, default=2)
    ap.add_argument("--camera-key", default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out = Path(cfg["output_dir"]) / "libero_v" / args.method
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(cfg["seeds"][0])

    policy, sib, preprocessor, postprocessor, H, d = build_policy(args, cfg, device)
    print(f"[libero_v] method={args.method}  TE={args.forge_ensemble}  "
          f"n_action_steps={args.n_action_steps}  H={H} d={d}", flush=True)

    import gymnasium as gym
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.configs import LiberoEnv

    suite = cfg["suite"]
    env_cfg = LiberoEnv(task=suite, fps=cfg["fps"],
                        episode_length=cfg.get("episode_length", 300))
    n_envs = cfg.get("eval_n_envs", 20)
    eps = min(args.episodes, n_envs)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    from libero.libero import benchmark
    all_tids = list(range(len(benchmark.get_benchmark_dict()[suite]().tasks)))
    task_ids = ([int(t) for t in args.tasks.split(",")] if args.tasks else all_tids)

    def build_one(tid):
        gk = dict(env_cfg.gym_kwargs); gk["task_ids"] = [tid]
        return create_libero_envs(
            task=suite, n_envs=n_envs, camera_name=env_cfg.camera_name,
            init_states=env_cfg.init_states, gym_kwargs=gk,
            env_cls=gym.vector.SyncVectorEnv, control_mode=env_cfg.control_mode,
            episode_length=env_cfg.episode_length)[suite][tid]

    grid = lv.libero_v_grid(compact=not args.full)
    grid = [("clean", {})] + grid   # clean (no perturbation) — plug-in SR measurement
    if args.only:
        keep = set(args.only.split(","))
        grid = [(lbl, c) for lbl, c in grid if lbl in keep]

    summary = []
    for label, cond in grid:
        all_succ, all_jerk = [], []
        per_task = []
        recorder = None
        if args.record:
            from sib.recording import RolloutRecorder
            recorder = RolloutRecorder(
                "results", f"libv_{args.method}_{suite}", condition=label,
                fps=10, video=True, max_videos_per_task=args.videos_per_task,
                camera_key=args.camera_key)
        for tid in task_ids:
            env = build_one(tid)
            if "sim" in cond:
                lv.install_sim_perturbations(env, cond["sim"])
            corruption = cond.get("corruption")
            sib.reset()
            try:
                succ, jerk, hfe = rollout_task(
                    env, sib, env_pre, env_post, preprocessor, postprocessor,
                    n_episodes=eps, corruption=corruption, device=device,
                    generator=generator, recorder=recorder,
                    task_label=f"{label}_t{tid}",
                    seed_base=cfg["seeds"][0])
            finally:
                try: env.close()
                except Exception: pass
            all_succ.extend(succ); all_jerk.extend(jerk)
            per_task.append({"task_id": tid, "n": len(succ),
                             "success_rate": float(np.mean(succ)) if succ else 0.0})
            print(f"[libero_v] {label} t{tid}: {int(np.sum(succ))}/{len(succ)}", flush=True)
        if recorder is not None:
            mani = recorder.finalize()
            print(f"[libero_v] videos -> results/videos/libv_{args.method}_{suite}/{label}/  (manifest {mani})", flush=True)
        n = len(all_succ); nsucc = int(np.sum(all_succ))
        p, lo, hi = wilson_ci(nsucc, n)
        res = {"name": f"libero_v_{args.method}_{label}", "method": args.method,
               "condition": label, "axis": cond.get("sim", {}).get("axis", "noise"),
               "n_episodes": n, "n_success": nsucc, "success_rate": success_rate(all_succ),
               "success_wilson95": [lo, hi],
               "rms_jerk_mean": float(np.nanmean(all_jerk)) if all_jerk else 0.0,
               "per_task": per_task}
        save_json(res, out / f"eval_{label}.json")
        summary.append((label, p, lo, hi, n))
        print(f"[libero_v] == {label}: SR={p*100:.1f}% [{lo*100:.1f},{hi*100:.1f}] n={n}", flush=True)

    print("\n==== LIBERO-V summary (%s) ====" % args.method)
    for label, p, lo, hi, n in summary:
        print(f"  {label:18s} {p*100:5.1f}%  [{lo*100:4.1f},{hi*100:4.1f}]  n={n}")
    if summary:
        avg = float(np.mean([p for _, p, _, _, _ in summary]))
        print(f"  {'AVG across conditions':18s} {avg*100:5.1f}%")
    save_json({"method": args.method,
               "conditions": [{"label": l, "sr": p, "n": n} for l, p, _, _, n in summary]},
              out / "summary.json")


if __name__ == "__main__":
    main()
