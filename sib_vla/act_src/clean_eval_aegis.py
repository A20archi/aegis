#!/usr/bin/env python
"""
clean_eval_aegis.py — ONE shard of the clean (non-perturbed) base-vs-AEGIS sweep:
a single (suite, arm, seed). 10 tasks x --episodes, resumable. Reuses the colleague's
exact rollout + obs conversion from evaluate_act.py.

  python clean_eval_aegis.py --suite libero_spatial --dataset lerobot/libero_spatial_image \
      --base-ckpt ../act_ckpts/Spatial/act/30000 --arm aegis --rib-weights ../results/aegis_act_v2/Spatial/rib.pt \
      --seed 42 --episodes 20 --out ../results/act_clean_v2/Spatial
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from evaluate_act import rollout, _make_base_libero_env, _sanitize
from aegis_eval_common import load_aegis_policy

import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--arm", choices=["base", "aegis"], required=True)
    ap.add_argument("--rib-weights", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", type=int, default=2, help="episodes/task to record (0=off)")
    ap.add_argument("--fusion-mult", type=float, default=1.0, help="scale RIB strength (0=identity,1=trained)")
    args = ap.parse_args()
    if args.arm == "aegis" and not args.rib_weights:
        raise SystemExit("--arm aegis requires --rib-weights")

    meta = LeRobotDatasetMetadata(args.dataset)
    rib = args.rib_weights if args.arm == "aegis" else None
    policy, pre, post = load_aegis_policy(meta, args.base_ckpt, args.replan, rib_weights=rib,
                                          fusion_mult=args.fusion_mult)
    print(f"[clean] {args.suite} arm={args.arm} seed={args.seed} loaded (rib={'yes' if rib else 'no'})", flush=True)

    out_dir = Path(args.out) / args.arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "result.json"
    res = json.loads(jpath.read_text()) if jpath.exists() else {"per_task": {}, "average": None}
    vdir = out_dir / "videos"; vdir.mkdir(exist_ok=True)

    for task_id in range(args.n_tasks):
        key = str(task_id)
        if key in res["per_task"]:
            continue
        env = _make_base_libero_env(task_id, suite_name=args.suite)
        task_desc = env.task_description
        succ = []
        for ep in range(args.episodes):
            cap = ep < args.videos
            ok, frames = rollout(policy, pre, post, env, task_desc, args.max_steps,
                                 seed=args.seed * 1000 + ep, capture=cap)
            succ.append(ok)
            if cap and frames:
                fn = f"task{task_id}_{_sanitize(task_desc)}_ep{ep:02d}_{'PASS' if ok else 'FAIL'}.mp4"
                try:
                    imageio.mimwrite(str(vdir / fn), frames, fps=20, codec="libx264",
                                     macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
                except Exception as e:
                    print(f"[vid] {fn}: {e}", flush=True)
        env.close()
        rate = float(np.mean(succ))
        res["per_task"][key] = {"task": task_desc, "success_rate": rate, "n": args.episodes}
        res["average"] = float(np.mean([v["success_rate"] for v in res["per_task"].values()]))
        jpath.write_text(json.dumps(res, indent=2))
        print(f"[clean/{args.arm}/s{args.seed}] {args.suite} task{task_id}: {rate*100:.1f}%  "
              f"(running avg {res['average']*100:.1f}%)", flush=True)

    print(f"[clean] DONE {args.suite} arm={args.arm} seed={args.seed} avg={res['average']*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
