#!/usr/bin/env python
"""
plus_eval_aegis.py — ONE shard of LIBERO-Plus base-vs-AEGIS: a single (suite, arm, seed).
7 categories x --n-per-cat stratified tasks x 1 trial, resumable.

MUST run in the lerobot_lplus env with the LIBERO-Plus repo on PYTHONPATH:
  PYTHONPATH=<LIBERO-plus>:<act_src>:<sib_vla> LIBERO_CONFIG_PATH=<.libero_lplus> \
  MAGICK_HOME=<env> LD_LIBRARY_PATH=<env>/lib MUJOCO_GL=egl \
  <env>/bin/python plus_eval_aegis.py --suite libero_spatial --dataset lerobot/libero_spatial_image \
      --base-ckpt ../act_ckpts/Spatial/act/30000 --arm aegis \
      --rib-weights ../results/aegis_act_v2/Spatial/rib.pt --seed 42 --n-per-cat 12 --out ...

Perturbed tasks are reached via create_libero_envs(task_ids=[id], init_states=False) (the proven
path); the env obs format matches the colleague's clean env, so we de-vectorize and feed their
exact env_obs_to_policy_input (180-flip + state). Instruction is cleaned from the classification
name for all categories except Language (where the reworded prompt IS the perturbation).
"""
from __future__ import annotations
import os, sys, json, argparse, re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LIBERO_PLUS_REPO", "/home/user/Desktop/vla_projects/LIBERO-plus")
for p in (REPO, HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import libero.libero.envs.env_wrapper  # noqa: F401  (before torch — linker gotcha)
import numpy as np
np.float_ = getattr(np, "float_", np.float64); np.complex_ = getattr(np, "complex_", np.complex128)
import gymnasium as gym
import torch

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.libero import create_libero_envs
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from evaluate_act import env_obs_to_policy_input
from aegis_eval_common import load_aegis_policy

CATEGORIES = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
              "Objects Layout", "Robot Initial States", "Language Instructions"]
_PERT = re.compile(r"_(table|view|add|light|noise|language|initstate)_\d+.*$")
_SCENE = re.compile(r"^[A-Z][A-Za-z_]*?SCENE\d+_")


def clean_instruction(name):
    s = _SCENE.sub("", _PERT.sub("", name))
    return s.replace("_", " ").strip()


def stratified_sample(tasks, n, seed=0):
    import random
    rng = random.Random(seed)
    by_d = defaultdict(list)
    for t in tasks:
        by_d[t["difficulty_level"]].append(t)
    for v in by_d.values():
        v.sort(key=lambda t: t["id"]); rng.shuffle(v)
    out, levels, i = [], sorted(by_d, key=lambda d: (d is None, d)), 0
    while len(out) < min(n, len(tasks)):
        lv = levels[i % len(levels)]
        if by_d[lv]:
            out.append(by_d[lv].pop())
        i += 1
        if all(len(by_d[lv]) == 0 for lv in levels):
            break
    return out


def _devec(obs):
    return {"pixels": {"image": obs["pixels"]["image"][0], "image2": obs["pixels"]["image2"][0]},
            "robot_state": {"eef": {"pos": obs["robot_state"]["eef"]["pos"][0],
                                    "quat": obs["robot_state"]["eef"]["quat"][0]},
                            "gripper": {"qpos": obs["robot_state"]["gripper"]["qpos"][0]}}}


@torch.no_grad()
def rollout(policy, pre, post, env, instr, max_steps, record_path=None):
    policy.reset()
    obs, info = env.reset()
    env.envs[0].task_description = instr
    frames = [] if record_path else None
    success = False
    for _ in range(max_steps):
        if frames is not None:
            frames.append(np.asarray(obs["pixels"]["image"][0])[::-1].copy())  # un-flip for viewing
        pin = pre(env_obs_to_policy_input(_devec(obs), instr))
        pin = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        a = post(action).to("cpu").float().numpy().reshape(1, -1)
        obs, r, term, trunc, info = env.step(a)
        fi = info.get("final_info")
        isk = info.get("is_success")
        if (isinstance(fi, dict) and bool(np.asarray(fi.get("is_success", False)).any())) or \
           (isk is not None and bool(np.asarray(isk).any())):
            success = True; break
        if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
            break
    if record_path and frames:
        import imageio, os as _os
        _os.makedirs(_os.path.dirname(record_path), exist_ok=True)
        imageio.mimsave(record_path, frames, fps=20)
    return success


def build_env(suite, tid, max_steps):
    rep = LiberoEnv(task=suite, fps=10, episode_length=max_steps)
    gk = dict(rep.gym_kwargs); gk["task_ids"] = [tid]
    return create_libero_envs(task=suite, n_envs=1, camera_name=rep.camera_name,
                              init_states=False, gym_kwargs=gk, env_cls=gym.vector.SyncVectorEnv,
                              control_mode=rep.control_mode, episode_length=max_steps)[suite][tid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--arm", choices=["base", "aegis"], required=True)
    ap.add_argument("--rib-weights", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-per-cat", type=int, default=12)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--categories", default=None)
    ap.add_argument("--record-dir", default=None, help="if set, save rollout mp4s here")
    ap.add_argument("--videos-per-cat", type=int, default=1)
    ap.add_argument("--fusion-mult", type=float, default=1.0, help="scale RIB strength (0=identity,1=trained)")
    ap.add_argument("--baseline", default="none", choices=["none", "tta", "bn"],
                    help="competitor baseline wrapped on the BASE policy (no RIB)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.arm == "aegis" and not args.rib_weights:
        raise SystemExit("--arm aegis requires --rib-weights")
    cats = CATEGORIES if not args.categories else [c.strip() for c in args.categories.split(",")]

    tc = json.load(open(f"{REPO}/libero/libero/benchmark/task_classification.json"))
    suite_tasks = tc[args.suite]
    by_cat = defaultdict(list)
    for t in suite_tasks:
        t = {**t, "index": t["id"]}
        by_cat[t["category"]].append(t)
    sampled = {c: stratified_sample(by_cat[c], args.n_per_cat, seed=args.seed) for c in cats}

    meta = LeRobotDatasetMetadata(args.dataset)
    rib = args.rib_weights if args.arm == "aegis" else None
    policy, pre, post = load_aegis_policy(meta, args.base_ckpt, args.replan, rib_weights=rib,
                                          fusion_mult=args.fusion_mult)
    if args.baseline != "none":
        from baselines import TTAWrapper, BNAdaptWrapper
        policy = (TTAWrapper(policy, n_action_steps=args.replan) if args.baseline == "tta"
                  else BNAdaptWrapper(policy, n_action_steps=args.replan)).to("cuda").eval()
    print(f"[plus] {args.suite} arm={args.arm} baseline={args.baseline} seed={args.seed} "
          f"(rib={'yes' if rib else 'no'})", flush=True)

    out_dir = Path(args.out) / args.arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "result.json"
    res = json.loads(jpath.read_text()) if jpath.exists() else {"per_cat": {}, "robustness_average": None}

    for c in cats:
        cres = res["per_cat"].setdefault(c, {"per_task": {}, "average": None})
        is_lang = (c == "Language Instructions")
        rec_n = 0
        for t in sampled[c]:
            key = str(t["index"])
            if key in cres["per_task"]:
                continue
            env = build_env(args.suite, t["index"], args.max_steps)
            instr = env.envs[0].task_description if is_lang else clean_instruction(t["name"])
            rpath = None
            if args.record_dir and rec_n < args.videos_per_cat:
                safe_c = re.sub(r"[^A-Za-z0-9]+", "_", c)
                rpath = f"{args.record_dir}/{safe_c}/{args.arm}_idx{t['index']}.mp4"
                rec_n += 1
            ok = rollout(policy, pre, post, env, instr, args.max_steps, record_path=rpath)
            env.close()
            cres["per_task"][key] = {"name": t["name"], "difficulty": t["difficulty_level"], "success": bool(ok)}
            cres["average"] = float(np.mean([v["success"] for v in cres["per_task"].values()]))
            jpath.write_text(json.dumps(res, indent=2))
            print(f"[plus/{args.arm}/s{args.seed}/{c[:12]}] idx{t['index']}: {'OK' if ok else 'x'} "
                  f"(cat {cres['average']*100:.0f}%)", flush=True)
        cat_avgs = [res["per_cat"][c]["average"] for c in cats if res["per_cat"].get(c, {}).get("average") is not None]
        res["robustness_average"] = float(np.mean(cat_avgs)) if cat_avgs else None
        jpath.write_text(json.dumps(res, indent=2))

    print(f"[plus] DONE {args.suite} {args.arm} s{args.seed} robust_avg={(res['robustness_average'] or 0)*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
