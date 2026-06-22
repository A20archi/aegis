"""AEGIS vs frozen-SmolVLA on LIBERO-Plus (external robustness benchmark, arXiv 2510.13626).

Bridges our sib AEGIS modules (RIB perception bottleneck + RASF action filter) into the
proven LIBERO-Plus harness (lerobot_lplus env -> LIBERO-Plus OffScreenRenderEnv so the 7
perturbation dims auto-apply by parsing the bddl filename). Same fast replan rollout for
both arms, so the per-dimension delta isolates the AEGIS modules.

  --method baseline   frozen SmolVLA (the floor)
  --method aegis      SmolVLA + RIB (+ RASF)        [RIB is the perception leg = the one
                      LIBERO-Plus's mostly-visual dims actually exercise]

Run (in the lerobot_lplus clone, with the env recipe; PYTHONPATH must include BOTH the
LIBERO-Plus repo AND sib_vla):
  PYTHONPATH=<repo>:<sib_vla> LIBERO_CONFIG_PATH=<cfg> MAGICK_HOME=<env> \
  LD_LIBRARY_PATH=<env>/lib MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  <env>/bin/python scripts/libero_plus_aegis_eval.py --method aegis --suite libero_object --per-cat N
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# IMPORTANT: import LIBERO-Plus's wand-dependent env module BEFORE torch (linker gotcha).
import libero.libero.envs.env_wrapper  # noqa: F401
import json, time, argparse
from collections import defaultdict
import numpy as np
np.float_ = np.float64  # LIBERO-Plus fog/plasma_fractal uses np.float_ (gone in NumPy 2.0)
import torch

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.utils.constants import ACTION

# --- our AEGIS bits ---
from sib.wrapper import load_smolvla
from sib.robust_ib import inject_fused_rib, load_rib_checkpoint

REPO = os.environ.get("LIBERO_PLUS_REPO", "/home/user/Desktop/vla_projects/LIBERO-plus")
_T0 = time.monotonic()
def el(): return f"{time.monotonic()-_T0:6.1f}s"
def p(*a): print(*a, flush=True)

BASKET_PHRASE = " and place it in the basket"
def clean_instruction(raw):
    """Strip the perturbation-encoding suffix LIBERO-Plus appends to .language (else the
    policy gets a garbled OOD instruction -> false 0% floor). Keep raw for Language dim."""
    if BASKET_PHRASE in raw:
        return raw.split(BASKET_PHRASE)[0] + BASKET_PHRASE
    return raw


def load_rasf_module(weights, H, d, device):
    """Rebuild the RASF module from its checkpoint (mirrors eval.py:load_trained_module)."""
    from sib.bottleneck import build_module
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    m = build_module(ckpt["config"], ckpt["H"], ckpt["d"]).to(device)
    m.load_state_dict(ckpt["module_state"]); m.eval()
    return m


def rollout_success(env, policy, pre, post, env_pre, env_post, seed, max_steps,
                    rasf=None, replan_h=10):
    policy.reset()
    obs, info = env.reset(seed=seed)
    env.envs[0].task_description = clean_instruction(env.envs[0].task_description)
    success = False; step = 0; chunk = None; qi = replan_h
    done = np.array([False])
    while not done.all() and step < max_steps:
        if qi >= replan_h:
            observation = preprocess_observation(obs)
            observation = add_envs_task(env, observation)
            observation = env_pre(observation)
            observation = pre(observation)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    chunk = policy.predict_action_chunk(observation)   # (B,H,d); RIB applies inside
                nd = chunk.dtype
                if rasf is not None:                                   # action leg (float32 math)
                    chunk = rasf(chunk.float(), context=None)["A_hat"].to(nd)
            qi = 0
        action = post(chunk[:, qi, :])
        a = env_post({ACTION: action})[ACTION].to("cpu").numpy()
        obs, r, term, trunc, info = env.step(a)
        qi += 1; step += 1
        if "final_info" in info and isinstance(info["final_info"], dict):
            success = success or bool(info["final_info"].get("is_success", False))
        done = np.logical_or(np.asarray(term), np.asarray(trunc))
        if success:
            break
    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "aegis"], default="aegis")
    ap.add_argument("--ckpt", default="HuggingFaceVLA/smolvla_libero", help="base SmolVLA")
    ap.add_argument("--rib-weights", default=None, help="RIB ckpt (aegis perception leg)")
    ap.add_argument("--rasf-weights", default=None, help="RASF ckpt (aegis action leg; optional)")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--per-cat", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="write per-category + total SR to this JSON "
                    "(enables persistence + Modal resume-skip; eval-only, no training)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = json.load(open(f"{REPO}/libero/libero/benchmark/task_classification.json"))
    suite_tasks = tc[args.suite]
    by_cat = defaultdict(list)
    for t in suite_tasks:
        by_cat[t["category"]].append(t["id"])
    cats = sorted(by_cat)
    if args.cats:
        cats = [c for c in cats if any(s.lower() in c.lower() for s in args.cats)]
    p(f"[{el()}] method={args.method} suite={args.suite} cats={ {c: len(by_cat[c]) for c in cats} }")

    # --- policy (load via our path so RIB injection matches) ---
    policy, pre, post, H, d = load_smolvla(args.ckpt, device)
    rasf = None
    if args.method == "aegis":
        if not args.rib_weights:
            raise SystemExit("--method aegis requires --rib-weights")
        rc = torch.load(args.rib_weights, map_location="cpu", weights_only=False).get("config", {})
        inject_fused_rib(policy, D_out=rc.get("D_out", 960), d_z=rc.get("d_z", 512),
                         n_heads=rc.get("n_heads", 8))
        policy.to(device)
        load_rib_checkpoint(policy, args.rib_weights, device=device)
        policy.eval()
        p(f"[{el()}] RIB injected + loaded")
        if args.rasf_weights:
            rasf = load_rasf_module(args.rasf_weights, H, d, device)
            p(f"[{el()}] RASF loaded")

    policy_cfg = policy.config
    picks = {}
    for c in cats:
        ids = by_cat[c]; n = min(args.per_cat, len(ids)); stp = max(1, len(ids)//n)
        picks[c] = ids[::stp][:n]

    rep_cfg = LiberoEnv(task=args.suite, task_ids=[picks[cats[0]][0]], init_states=False)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=rep_cfg, policy_cfg=policy_cfg)
    p(f"[{el()}] processors built; starting rollouts")

    results = defaultdict(list)
    for c in cats:
        for tid in picks[c]:
            env_cfg = LiberoEnv(task=args.suite, task_ids=[tid], init_states=False)
            envs = make_env(env_cfg, n_envs=1)
            env = next(iter(next(iter(envs.values())).values()))
            succ_n = 0
            for ep in range(args.episodes):
                s = rollout_success(env, policy, pre, post, env_pre, env_post,
                                    args.seed + tid*10 + ep, args.max_steps, rasf=rasf)
                succ_n += int(s); results[c].append(s)
            env.close()
            p(f"  [{el()}] {c:20s} task{tid:4d}: {succ_n}/{args.episodes}")

    p(f"\n========== LIBERO-Plus: {args.method} ({args.suite}) ==========")
    overall = []
    for c in cats:
        r = results[c]; overall += r
        p(f"  {c:22s}: {100.0*sum(r)/max(1,len(r)):5.1f}%  (n={len(r)})")
    p(f"  {'TOTAL':22s}: {100.0*sum(overall)/max(1,len(overall)):5.1f}%  (n={len(overall)})")
    if args.out:
        rec = {"method": args.method, "suite": args.suite, "per_cat": args.per_cat,
               "per_category": {c: {"sr": (sum(results[c]) / max(1, len(results[c]))),
                                    "n": len(results[c])} for c in cats},
               "total_sr": (sum(overall) / max(1, len(overall))), "n_episodes": len(overall)}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as _f:
            json.dump(rec, _f, indent=2)
        p(f"[out] wrote {args.out}")
    p("PROBE_DONE")


if __name__ == "__main__":
    main()
