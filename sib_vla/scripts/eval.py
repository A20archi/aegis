"""Evaluate a trained policy on LIBERO (optionally under an observation corruption).

    python scripts/eval.py --config configs/sib.yaml --weights results/sib_beta0.001.pt
    python scripts/eval.py --config configs/vanilla.yaml --corruption gaussian_noise:1

Builds the LIBERO env via ``make_env(LiberoEnv(...))``, wraps the frozen policy
with the trained bottleneck, rolls out ``episodes_per_task`` episodes per task,
and records success (Wilson CI), RMS jerk, and HF energy fraction of the
*executed* (post-processed) action stream.

Integration points to confirm against your LeRobot build are marked ``VERIFY``;
the math/metrics this feeds are fully tested in ``sib/``.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch

from sib import corruptions as corr
from sib.bottleneck import build_module
from sib.metrics import (hf_energy_fraction, rms_jerk, success_rate,
                         two_proportion_ztest, wilson_ci)
from sib.utils import (GpuHourLogger, load_config, resolve_device, save_json,
                       set_seed)
from sib.wrapper import EMABlendPolicy, ForgeActionHeadPolicy, SIBPolicy, load_smolvla


def load_trained_module(weights: str, H, d, device):
    """Rebuild the module from a train checkpoint (or identity for vanilla)."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    module = build_module(ckpt["config"], ckpt["H"], ckpt["d"]).to(device)
    module.load_state_dict(ckpt["module_state"])
    module.eval()
    return module


def _apply_corruption(obs_batch: dict, spec, generator):
    """Apply ``name:severity`` to every image key in a preprocessed obs batch."""
    if spec is None:
        return obs_batch
    name, sev = spec.split(":")
    for k, v in obs_batch.items():
        if "image" in k and torch.is_tensor(v) and v.dim() >= 3:
            obs_batch[k] = corr.apply(v, name, int(sev), generator=generator)
    return obs_batch


def _read_success(info: dict, n_envs: int):
    """Per-env (success, valid-mask) from step info, robust to both conventions.

    Gymnasium >=1.0 vector envs may report success either directly as
    ``info['is_success']`` (+ ``_is_success`` mask) -- as this LIBERO env does --
    or inside ``info['final_info']['is_success']`` (+ ``_final_info``) on the
    termination step. Verified empirically: this env uses the direct form.
    """
    fi = info.get("final_info")
    if isinstance(fi, dict) and "is_success" in fi:
        succ = np.asarray(fi["is_success"]).reshape(-1)
        mask = np.asarray(info.get("_final_info", np.ones(n_envs, bool))).reshape(-1)
        return succ, mask
    succ = np.asarray(info.get("is_success", np.zeros(n_envs))).reshape(-1)
    mask = np.asarray(info.get("_is_success", np.ones(n_envs, bool))).reshape(-1)
    return succ, mask


@torch.no_grad()
def rollout_task(env, sib, env_preprocessor, env_postprocessor, preprocessor,
                 postprocessor, n_episodes, corruption, device, generator,
                 action_noise=0.0, recorder=None, task_label="task", seed_base=0):
    """Roll out one task, mirroring LeRobot's canonical eval flow with the
    bottleneck inserted at chunk prediction.

    Flow (per step):
        preprocess_observation -> add_envs_task -> env_preprocessor
        (LIBERO remap: pixels/robot_state -> observation.images.image/image2 +
        observation.state) -> [corruption + frame capture] -> policy preprocessor
        -> sib.select_action (predict_action_chunk + bottleneck, queued) ->
        postprocessor -> env_postprocessor -> env.step.

    Success is read from the SyncVectorEnv ``is_success`` / ``_is_success``
    arrays (verified present in this env's step info).
    """
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION
    from sib.recording import extract_frames

    n_envs = env.num_envs
    max_steps = int(env.call("_max_episode_steps")[0])
    successes, jerks, hfes = [], [], []
    done_episodes = 0
    batch_ix = 0
    while done_episodes < n_episodes:
        # NOTE (LIBERO): each sub-env i is permanently tied to init_state[i]
        # (see lerobot/envs/libero.py: _init_state_id = episode_index). So one
        # batch == n_envs DISTINCT init states; running extra batches would just
        # re-evaluate the same states. main() therefore caps n_episodes <= n_envs.
        # Seeds make the MuJoCo dynamics reproducible across runs.
        seeds = [seed_base + batch_ix * n_envs + i for i in range(n_envs)]
        obs, info = env.reset(seed=seeds)
        sib.reset()
        traj = [[] for _ in range(n_envs)]
        rec = recorder is not None and recorder.video and done_episodes < recorder.max_videos_per_task
        frames = [[] for _ in range(n_envs)] if rec else None
        ep_success = [False] * n_envs
        # Per-env done mask: each env contributes exactly ONE episode (its first).
        # Gymnasium vector envs auto-reset on termination, so once env i is done we
        # must stop recording it -- otherwise its trace/success leak into the next
        # (auto-reset) episode. Mirrors lerobot_eval's `done` handling.
        env_done = np.zeros(n_envs, dtype=bool)
        for _ in range(max_steps):
            observation = preprocess_observation(obs)
            observation = add_envs_task(env, observation)          # inject task string
            observation = env_preprocessor(observation)            # LIBERO -> policy keys
            observation = _apply_corruption(observation, corruption, generator)
            if rec:
                sf = extract_frames(observation, recorder.camera_key)
                for i in range(min(n_envs, len(sf))):
                    if not env_done[i]:
                        frames[i].append(sf[i])
            observation = preprocessor(observation)                # normalise + tokenise
            observation = {k: (v.to(device) if torch.is_tensor(v) else v)
                           for k, v in observation.items()}
            # Noise BEFORE bottleneck (the filterable signal) for SIB modules;
            # vanilla has no bottleneck so pre/post is equivalent for it.
            action = sib.select_action(observation,
                                       pre_noise_std=action_noise,
                                       generator=generator)        # bottleneck-corrected, normalised
            action = postprocessor(action)                         # -> robot space
            action = env_postprocessor({ACTION: action})[ACTION]
            # post-bottleneck actuator jitter: intentionally 0 (noise already injected pre-bottleneck)
            np_action = action.to("cpu").numpy()
            assert np_action.ndim == 2 and np_action.shape[0] == n_envs, \
                f"action must be (n_envs, d), got {np_action.shape}"
            obs, reward, terminated, truncated, info = env.step(np_action)
            succ, mask = _read_success(info, n_envs)
            term = np.asarray(terminated).reshape(-1)
            trunc = np.asarray(truncated).reshape(-1)
            for i in range(n_envs):
                if env_done[i]:
                    continue
                traj[i].append(np_action[i])
                if i < succ.size and bool(mask[i]) and bool(succ[i]):
                    ep_success[i] = True                           # sticky within the episode
                if bool(term[i]) or bool(trunc[i]):
                    env_done[i] = True                             # finalize; ignore auto-reset
            if env_done.all():
                break
        for i in range(n_envs):
            if done_episodes >= n_episodes:
                break
            seq = np.asarray(traj[i])
            successes.append(ep_success[i])
            jerks.append(rms_jerk(seq))
            hfes.append(hf_energy_fraction(seq))
            if recorder is not None:
                recorder.write_episode(task_label, done_episodes,
                                       frames[i] if rec else None, seq, ep_success[i],
                                       rms_jerk=rms_jerk(seq),
                                       hf_energy=hf_energy_fraction(seq))
            done_episodes += 1
        batch_ix += 1
    return successes, jerks, hfes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None, help="train checkpoint; omit for vanilla")
    ap.add_argument("--corruption", default=None, help="name:severity, e.g. gaussian_blur:0")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="std of additive Gaussian noise on the executed action")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task ids to evaluate (shard subset); "
                         "overrides cfg.tasks. Enables concurrent task-sharded runs.")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override execution cadence (open-loop horizon); for "
                         "cadence sweeps without per-n config files.")
    ap.add_argument("--ema-alpha", type=float, default=0.0,
                    help="EMA blend weight at chunk boundary (0=off, e.g. 0.5)")
    ap.add_argument("--ema-k", type=int, default=5,
                    help="number of steps to blend at each chunk boundary")
    ap.add_argument("--forge-ensemble", action="store_true",
                    help="Forge recipe: L2+L3+L5 on action head, SIB after")
    ap.add_argument("--ensemble-coeff", type=float, default=0.01,
                    help="Temporal ensemble exp weight coeff (0.01 = ACT default)")
    ap.add_argument("--lever3-clamp", type=float, default=1.0,
                    help="L3 percentile soft-clamp in normalised action space (1.0=off)")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override flow-matching inference denoising steps (default 10)")
    ap.add_argument("--ib-weights", default=None,
                    help="Fused IB-Adapter checkpoint; injects visual bottleneck into connector")
    ap.add_argument("--n-heads", type=int, default=8,
                    help="IB-Adapter heads (only used with --ib-weights)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    if args.tasks is not None:
        cfg["tasks"] = [int(t) for t in args.tasks.split(",") if t.strip() != ""]
    if args.n_action_steps is not None:
        cfg["n_action_steps"] = args.n_action_steps
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out = Path(cfg["output_dir"])
    generator = torch.Generator(device=device).manual_seed(cfg["seeds"][0])

    policy, preprocessor, postprocessor, H, d = load_smolvla(cfg["checkpoint"], str(device))

    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps
        print(f"[eval] flow-matching num_steps overridden -> {args.num_steps}")

    if args.ib_weights is not None:
        from sib.ib_adapter import inject_fused_ib_adapter, load_ib_checkpoint
        inject_fused_ib_adapter(policy, D_out=960, n_heads=args.n_heads)
        policy.to(device)
        load_ib_checkpoint(policy, args.ib_weights, device=str(device))
        policy.eval()
        conn = policy.model.vlm_with_expert.vlm.model.connector
        print(f"[eval] IB-Adapter injected  fusion_coeff(tanh)={conn.modality_projection.proj.ib_contribution:.3f}")

    if args.weights is not None:
        module = load_trained_module(args.weights, H, d, device)
    else:  # vanilla: an identity passthrough (gain_no_rate starts at gain=1)
        from sib.bottleneck import SpectralActionModule
        module = SpectralActionModule("gain_no_rate", H, d).to(device).eval()

    # Receding-horizon length: honor the checkpoint's value unless the config overrides.
    n_action_steps = cfg.get("n_action_steps") or policy.config.n_action_steps
    if args.forge_ensemble:
        sib = ForgeActionHeadPolicy(
            policy, module,
            n_action_steps=n_action_steps,
            ensemble_coeff=args.ensemble_coeff,
            lever3_clamp=args.lever3_clamp,
        ).to(device)
        print(f"[eval] Forge recipe: L2 n={n_action_steps}  "
              f"L3 clamp={args.lever3_clamp}  L5 coeff={args.ensemble_coeff}  "
              f"then SIB module")
    elif args.ema_alpha > 0:
        sib = EMABlendPolicy(policy, module, n_action_steps=n_action_steps,
                             ema_alpha=args.ema_alpha, ema_k=args.ema_k).to(device)
        print(f"[eval] EMA blend: alpha={args.ema_alpha}, k={args.ema_k}")
    else:
        sib = SIBPolicy(policy, module, n_action_steps=n_action_steps).to(device)
    print(f"[eval] n_action_steps={n_action_steps} (chunk H={H}, d={d})")

    import gymnasium as gym
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.configs import LiberoEnv
    env_cfg = LiberoEnv(task=cfg["suite"], fps=cfg["fps"],
                        episode_length=cfg.get("episode_length", 300))
    n_envs = cfg.get("eval_n_envs", 1)
    suite = cfg["suite"]
    task_subset = cfg.get("tasks")             # list[int] or None -> all tasks
    per_task_build = cfg.get("per_task_build", False)

    # Environment-specific obs/action remap (LIBERO: pixels/robot_state -> policy
    # keys). Independent of which tasks are built, so construct once.
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config)

    def _all_task_ids():
        from libero.libero import benchmark
        return list(range(len(benchmark.get_benchmark_dict()[suite]().tasks)))

    def _build_one_task(tid):
        """Build ONE task's vec env (peak memory = n_envs, not n_tasks*n_envs).
        Replicates make_env's LIBERO call with a task_ids filter so a 50-init-state
        protocol stays feasible without an eager all-tasks build."""
        gk = dict(env_cfg.gym_kwargs); gk["task_ids"] = [tid]
        return create_libero_envs(
            task=suite, n_envs=n_envs, camera_name=env_cfg.camera_name,
            init_states=env_cfg.init_states, gym_kwargs=gk,
            env_cls=gym.vector.SyncVectorEnv, control_mode=env_cfg.control_mode,
            episode_length=env_cfg.episode_length)[suite][tid]

    # Two build modes yielding (task_id, env). Per-task builds+closes sequentially
    # (low peak memory, incremental results); eager builds every task up front.
    if per_task_build:
        task_ids = task_subset if task_subset is not None else _all_task_ids()
        def iter_task_envs():
            for tid in task_ids:
                env = _build_one_task(tid)
                try:
                    yield tid, env
                finally:
                    try: env.close()
                    except Exception: pass
    else:
        built = make_env(env_cfg, n_envs=n_envs)[suite]   # {task: VectorEnv}
        if task_subset is not None:
            keep = set(task_subset)
            for tid in list(built):
                if tid not in keep:
                    try: built[tid].close()
                    except Exception: pass
                    del built[tid]
            print(f"[eval] task subset -> {sorted(keep)}")
        task_ids = list(built)
        def iter_task_envs():
            for tid in task_ids:
                yield tid, built[tid]

    # LIBERO ties init_state[i] to sub-env i, so distinct episodes == distinct
    # sub-envs. Evaluating more than n_envs episodes/task would re-run the same
    # init states (falsely tight CI). Cap episodes/task at n_envs (one batch).
    eps_per_task = cfg.get("episodes_per_task", n_envs)
    if eps_per_task > n_envs:
        print(f"[eval] WARNING: episodes_per_task={eps_per_task} > eval_n_envs={n_envs}; "
              f"LIBERO would duplicate init states -- capping to {n_envs}. "
              f"Set eval_n_envs={eps_per_task} to evaluate that many distinct init states.")
        eps_per_task = n_envs
    print(f"[eval] {eps_per_task} distinct init-state episodes/task x {len(task_ids)} tasks"
          f"{' (per-task build)' if per_task_build else ''}", flush=True)

    # One eval = one perturbation condition (clean / image corruption / action noise).
    name = args.tag or Path(args.config).stem
    condition = args.corruption or (f"action_noise:{args.action_noise}"
                                    if args.action_noise > 0 else None)

    rec_cfg = cfg.get("record", {})
    recorder = None
    if rec_cfg.get("enabled", True):
        from sib.recording import RolloutRecorder
        recorder = RolloutRecorder(
            out, name, condition=condition or "clean",
            fps=rec_cfg.get("fps", 10), video=rec_cfg.get("video", True),
            max_videos_per_task=rec_cfg.get("videos_per_task", 3),
            camera_key=rec_cfg.get("camera_key"))

    all_succ, all_jerk, all_hfe = [], [], []
    per_task_results = []
    with GpuHourLogger(f"eval_{name}", out, cfg.get("n_gpus", 1)):
        for task_id, env in iter_task_envs():
            s, j, h = rollout_task(
                env, sib, env_preprocessor, env_postprocessor,
                preprocessor, postprocessor,
                n_episodes=eps_per_task,
                corruption=args.corruption, device=device, generator=generator,
                action_noise=args.action_noise,
                recorder=recorder, task_label=f"{suite}_{task_id}",
                seed_base=cfg["seeds"][0])
            all_succ += s; all_jerk += j; all_hfe += h
            per_task_results.append({
                "task_id": task_id,
                "n": len(s),
                "n_success": int(np.sum(s)),
                "success_rate": float(np.mean(s)) if s else 0.0,
                "rms_jerk_mean": float(np.nanmean(j)) if j else 0.0,
            })
            rs = int(np.sum(all_succ))
            print(f"[eval]   {suite}_{task_id}: {int(np.sum(s))}/{len(s)}  | "
                  f"running {rs}/{len(all_succ)} = {rs/max(len(all_succ),1):.3f}", flush=True)
    if recorder is not None:
        manifest = recorder.finalize()
        print(f"[eval] rollout artefacts -> {manifest}")

    n = len(all_succ)
    n_succ = int(np.sum(all_succ))
    p, lo, hi = wilson_ci(n_succ, n)
    bpb = module.bits_per_band()
    result = {
        "name": name,
        "config": cfg.get("module", "vanilla"),
        "corruption": condition,
        "n_episodes": n, "n_success": n_succ,
        "success_rate": success_rate(all_succ),
        "success_wilson95": [lo, hi],
        "rms_jerk_mean": float(np.nanmean(all_jerk)),
        "hf_energy_fraction_mean": float(np.nanmean(all_hfe)),
        "total_bits": None if bpb is None else float(bpb.sum()),
        "per_task": per_task_results,
    }
    suffix = f"__{condition.replace(':', '')}" if condition else ""
    save_json(result, out / f"eval_{result['name']}{suffix}.json")
    print(f"[eval] {result['name']}{suffix}: success={p:.3f} "
          f"[{lo:.3f},{hi:.3f}] n={n}  jerk={result['rms_jerk_mean']:.4g}")


if __name__ == "__main__":
    main()
