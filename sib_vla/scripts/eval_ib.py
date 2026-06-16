"""Evaluate SmolVLA + Fused IB-Adapter on LIBERO.

Usage:
    python scripts/eval_ib.py --config configs/ib_adapter.yaml \\
        --weights results/smolvla_ib.pt --tag ib_adapter

If --weights is omitted: evaluates vanilla SmolVLA (sanity check, == eval.py vanilla).
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch

from sib.ib_adapter import inject_fused_ib_adapter, load_ib_checkpoint
from sib.metrics import rms_jerk, hf_energy_fraction, success_rate, wilson_ci
from sib.utils import GpuHourLogger, load_config, resolve_device, save_json, set_seed


def _read_success(info, n_envs):
    fi = info.get("final_info")
    if isinstance(fi, dict) and "is_success" in fi:
        succ = np.asarray(fi["is_success"]).reshape(-1)
        mask = np.asarray(info.get("_final_info", np.ones(n_envs, bool))).reshape(-1)
        return succ, mask
    succ = np.asarray(info.get("is_success", np.zeros(n_envs))).reshape(-1)
    mask = np.asarray(info.get("_is_success", np.ones(n_envs, bool))).reshape(-1)
    return succ, mask


@torch.no_grad()
def rollout_task(env, policy, preprocessor, postprocessor,
                 env_preprocessor, env_postprocessor,
                 n_episodes, device, seed_base=0):
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    n_envs = env.num_envs
    max_steps = int(env.call("_max_episode_steps")[0])
    successes, jerks, hfes = [], [], []
    done_episodes = 0
    batch_ix = 0

    while done_episodes < n_episodes:
        seeds = [seed_base + batch_ix * n_envs + i for i in range(n_envs)]
        obs, info = env.reset(seed=seeds)
        policy.reset()
        traj = [[] for _ in range(n_envs)]
        ep_success = [False] * n_envs
        env_done = np.zeros(n_envs, dtype=bool)

        for _ in range(max_steps):
            observation = preprocess_observation(obs)
            observation = add_envs_task(env, observation)
            observation = env_preprocessor(observation)
            observation = preprocessor(observation)
            observation = {k: (v.to(device) if torch.is_tensor(v) else v)
                           for k, v in observation.items()}

            action = policy.select_action(observation)   # SmolVLA's native method
            action = postprocessor(action)
            action = env_postprocessor({ACTION: action})[ACTION]

            for i in range(n_envs):
                if not env_done[i]:
                    traj[i].append(action[i].cpu().numpy())

            np_action = action.to("cpu").numpy()
            obs, reward, terminated, truncated, info = env.step(np_action)

            succ, mask = _read_success(info, n_envs)
            for i in range(n_envs):
                if not env_done[i] and i < succ.size and bool(mask[i]) and bool(succ[i]):
                    ep_success[i] = True    # sticky: once True, stays True
            env_done |= (terminated | truncated)
            if env_done.all():
                break

        for i in range(n_envs):
            if len(traj[i]) > 1:
                tr = np.array(traj[i])   # (T, d)
                jerks.append(float(rms_jerk(tr)))
                hfes.append(float(hf_energy_fraction(tr)))
            successes.append(ep_success[i])
        done_episodes += n_envs
        batch_ix += 1

    return successes, jerks, hfes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None,
                    help="IB checkpoint (.pt); omit for vanilla SmolVLA")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-heads", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    device = resolve_device(cfg)
    out = Path(cfg["output_dir"])

    # ------------------------------------------------------------------
    # 1. Load SmolVLA + optionally inject IB-Adapter
    # ------------------------------------------------------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print("[eval_ib] loading SmolVLA ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg["checkpoint"]).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, cfg["checkpoint"])
    H = policy.config.chunk_size
    d = policy.config.action_feature.shape[0]

    if args.weights is not None:
        inject_fused_ib_adapter(policy, D_out=960, n_heads=args.n_heads)
        policy.to(device)   # move newly injected IBAdapter to GPU before loading weights
        load_ib_checkpoint(policy, args.weights, device=str(device))
        policy.eval()
        conn = policy.model.vlm_with_expert.vlm.model.connector
        coeff = conn.modality_projection.proj.ib_contribution
        print(f"[eval_ib] IB-Adapter loaded  fusion_coeff={coeff:.3f}", flush=True)
    else:
        print("[eval_ib] no weights → vanilla SmolVLA", flush=True)

    n_action_steps = cfg.get("n_action_steps") or policy.config.n_action_steps
    policy.config.n_action_steps = n_action_steps   # enforce receding-horizon cadence
    print(f"[eval_ib] n_action_steps={n_action_steps}  chunk H={H}  d={d}", flush=True)

    # ------------------------------------------------------------------
    # 2. Build LIBERO envs
    # ------------------------------------------------------------------
    import gymnasium as gym
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.configs import LiberoEnv
    from libero.libero import benchmark

    suite = cfg["suite"]
    n_envs = cfg.get("eval_n_envs", 1)
    eps_per_task = cfg.get("episodes_per_task", n_envs)
    per_task_build = cfg.get("per_task_build", False)

    env_cfg = LiberoEnv(task=suite, fps=cfg["fps"],
                        episode_length=cfg.get("episode_length", 300))
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config)

    all_task_ids = list(range(len(benchmark.get_benchmark_dict()[suite]().tasks)))
    task_ids = cfg.get("tasks") or all_task_ids

    def _build_one(tid):
        gk = dict(env_cfg.gym_kwargs); gk["task_ids"] = [tid]
        return create_libero_envs(
            task=suite, n_envs=n_envs, camera_name=env_cfg.camera_name,
            init_states=env_cfg.init_states, gym_kwargs=gk,
            env_cls=gym.vector.SyncVectorEnv, control_mode=env_cfg.control_mode,
            episode_length=env_cfg.episode_length)[suite][tid]

    print(f"[eval_ib] {eps_per_task} ep/task x {len(task_ids)} tasks  "
          f"n_envs={n_envs}  suite={suite}", flush=True)

    # ------------------------------------------------------------------
    # 3. Rollout
    # ------------------------------------------------------------------
    name = args.tag or Path(args.config).stem
    all_succ, all_jerk, all_hfe = [], [], []
    per_task_results = []

    with GpuHourLogger(f"eval_ib_{name}", out, cfg.get("n_gpus", 1)):
        for task_id in task_ids:
            env = _build_one(task_id)
            try:
                s, j, h = rollout_task(
                    env, policy, preprocessor, postprocessor,
                    env_preprocessor, env_postprocessor,
                    n_episodes=min(eps_per_task, n_envs),
                    device=device, seed_base=cfg["seeds"][0])
            finally:
                try: env.close()
                except Exception: pass

            all_succ += s; all_jerk += j; all_hfe += h
            per_task_results.append({
                "task_id": task_id,
                "n": len(s),
                "n_success": int(np.sum(s)),
                "success_rate": float(np.mean(s)) if s else 0.0,
                "rms_jerk_mean": float(np.nanmean(j)) if j else 0.0,
            })
            rs = int(np.sum(all_succ))
            print(f"[eval_ib]   {suite}_{task_id}: {int(np.sum(s))}/{len(s)}  | "
                  f"running {rs}/{len(all_succ)} = {rs/max(len(all_succ),1):.3f}",
                  flush=True)

    n = len(all_succ)
    n_succ = int(np.sum(all_succ))
    p, lo, hi = wilson_ci(n_succ, n)
    result = {
        "name": name,
        "config": "ib_adapter" if args.weights else "vanilla",
        "corruption": None,
        "n_episodes": n, "n_success": n_succ,
        "success_rate": success_rate(all_succ),
        "success_wilson95": [lo, hi],
        "rms_jerk_mean": float(np.nanmean(all_jerk)),
        "hf_energy_fraction_mean": float(np.nanmean(all_hfe)),
        "total_bits": None,
        "per_task": per_task_results,
    }
    save_json(result, out / f"eval_{result['name']}.json")
    print(f"[eval_ib] {name}: success={p:.3f} [{lo:.3f},{hi:.3f}] n={n}  "
          f"jerk={result['rms_jerk_mean']:.4g}")


if __name__ == "__main__":
    main()
