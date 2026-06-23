#!/usr/bin/env python
"""Local clean-SR runner (eval_libero_v.py, stock LIBERO via the `lerobot` env).

Default: LONG suite, 3 seeds, both arms, n=100 (10 tasks x 10 envs) -> a complete,
consistent clean-long row. Each clean cell is one batched process (eval_n_envs envs share
the policy), so it is GPU-light (~4GB) and overlaps fine with the LIBERO-Plus run.
"""
import os, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla"
PY = "/home/user/miniconda3/envs/lerobot/bin/python"          # stock LIBERO lives here
SCRIPT = f"{ROOT}/scripts/eval_libero_v.py"
CKPT = f"{ROOT}/outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model"
RIB = f"{ROOT}/results/ib_on86/rib_on86.pt"
RASF = f"{ROOT}/results/rasf_on86/rasf_on86.pt"
OUTROOT = f"{ROOT}/results/local_clean"
SUITES_MAP = {"object": "libero_object", "goal": "libero_goal",
              "spatial": "libero_spatial", "long": "libero_10"}
ELEN = {"object": 280, "goal": 300, "spatial": 220, "long": 520}

SUITES = os.environ.get("CSUITES", "long").split(",")
SEEDS = [int(s) for s in os.environ.get("CSEEDS", "42,123,456").split(",")]
ARMS = ["baseline", "aegis"]
N_ENVS = int(os.environ.get("N_ENVS", "10"))      # envs per task (= one batch)
NTASKS = int(os.environ.get("NTASKS", "10"))      # tasks per cell (5 -> n=50 to match snapshot)
CONC = int(os.environ.get("CCONC", "6"))
TASKS = ",".join(str(i) for i in range(NTASKS))
EXPECT = N_ENVS * NTASKS

ENV = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
       "PYTHONPATH": ROOT, "HF_HUB_OFFLINE": "1"}
_LK = threading.Lock()


def od(suite):
    return f"{OUTROOT}/{suite}"

def out_json(suite, seed, arm):
    return f"{od(suite)}/seed{seed}/libero_v/{arm}/eval_clean.json"

def done(suite, seed, arm):
    import json
    p = out_json(suite, seed, arm)
    try:
        j = json.load(open(p))
        return int(j.get("n_episodes", 0)) >= EXPECT and not j.get("partial", False)
    except Exception:
        return False

def run_cell(job):
    suite, seed, arm = job
    if done(suite, seed, arm):
        return job, "skip", 0.0
    sk = SUITES_MAP[suite]
    cfgdir = f"{od(suite)}"
    os.makedirs(cfgdir, exist_ok=True)
    cfg = f"{cfgdir}/_cfg_{suite}_{seed}_{arm}.yaml"
    with open(cfg, "w") as f:
        f.write(f"inherit: {ROOT}/configs/base.yaml\n"
                f"checkpoint: {CKPT}\n"
                f"repo_id: HuggingFaceVLA/libero\n"
                f"output_dir: {od(suite)}\n"
                f"suite: {sk}\n"
                f"episode_length: {ELEN[suite]}\n"
                f"n_action_steps: 1\n"
                f"eval_n_envs: {N_ENVS}\n"
                f"episodes_per_task: {N_ENVS}\n"
                f"record:\n  enabled: false\n")
    cmd = [PY, "-u", SCRIPT, "--config", cfg, "--method", arm]
    if arm == "aegis":
        cmd += ["--rib-weights", RIB, "--rasf-weights", RASF]
    cmd += ["--n-action-steps", "1", "--episodes", str(N_ENVS),
            "--tasks", TASKS, "--only", "clean", "--seed", str(seed),
            "--forge-ensemble", "--ensemble-coeff", "0.01"]
    log = f"{cfgdir}/seed{seed}_{arm}.log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with _LK:
        time.sleep(4)
    t0 = time.monotonic()
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    ok = done(suite, seed, arm)
    return job, ("ok" if (rc == 0 and ok) else f"FAIL rc={rc} ok={ok}"), time.monotonic() - t0

def main():
    jobs = [(s, sd, a) for s in SUITES for sd in SEEDS for a in ARMS]
    todo = [j for j in jobs if not done(*j)]
    print(f"[clean] {len(jobs)} cells, {len(todo)} to run, n={EXPECT}/cell, conc={CONC}", flush=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for f in as_completed({ex.submit(run_cell, j): j for j in todo}):
            job, st, dt = f.result()
            print(f"  [{(time.monotonic()-t0)/60:5.1f}m] {job} -> {st} ({dt/60:.1f}m)", flush=True)
    print(f"[clean] done {(time.monotonic()-t0)/60:.1f}m", flush=True)

if __name__ == "__main__":
    main()
