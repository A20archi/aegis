#!/usr/bin/env python
"""Local A100 parallel runner for the LIBERO-Plus robustness table.

Saturates a single A100 + 48-core box by running many (suite, seed, arm) eval cells
CONCURRENTLY (sim is CPU-bound, GPU mostly idle -> safe to overlap N processes that
share the GPU). Each cell -> one libero_plus_aegis_eval.py subprocess writing its own
JSON, so the pool is resumable (a finished cell is skipped on re-run).

ENV knobs:  PER_CAT (default 12 -> n=84/cell), CONC (default 14), SEEDS, SUITES, ARMS
"""
import os, sys, json, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_LAUNCH_LOCK = threading.Lock()  # stagger CUDA inits so N processes don't OOM on simultaneous load

ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla"
PY = "/home/user/miniconda3/envs/lerobot_lplus/bin/python"
SCRIPT = f"{ROOT}/scripts/libero_plus_aegis_eval.py"
# The on86 RIB/RASF modules were trained against THIS base (expert_width_multiplier=0.75 ->
# 720-wide expert); the HF default smolvla_libero is 480-wide and mismatches. Both arms use it.
CKPT = os.environ.get("CKPT_DIR", f"{ROOT}/outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model")
RIB = os.environ.get("RIB_WEIGHTS", f"{ROOT}/results/ib_on86/rib_on86.pt")
RASF = os.environ.get("RASF_WEIGHTS", f"{ROOT}/results/rasf_on86/rasf_on86.pt")
OUTDIR = os.environ.get("OUTDIR", f"{ROOT}/results/local_lplus")
LPLUS_REPO = "/home/user/Desktop/vla_projects/LIBERO-plus"

MAXSTEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
N_CATS = 7  # 7 perturbation categories per suite

PER_CAT = int(os.environ.get("PER_CAT", "12"))
CONC = int(os.environ.get("CONC", "14"))
SUITES = os.environ.get("SUITES", "libero_object,libero_goal,libero_spatial,libero_10").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123,456").split(",")]
ARMS = os.environ.get("ARMS", "baseline,aegis").split(",")

ENV = {**os.environ,
       "LIBERO_PLUS_REPO": LPLUS_REPO,
       "LIBERO_CONFIG_PATH": "/home/user/Desktop/vla_projects/.libero_lplus",
       "PYTHONPATH": f"{ROOT}:{LPLUS_REPO}",
       "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HUB_OFFLINE": "1"}

def expected_n():
    return N_CATS * PER_CAT

def cell_out(suite, seed, arm):
    return f"{OUTDIR}/{suite}/seed{seed}/{arm}.json"

def done(suite, seed, arm):
    p = cell_out(suite, seed, arm)
    if not os.path.exists(p):
        return False
    try:
        j = json.load(open(p))
        return int(j.get("n_episodes", 0)) >= expected_n()
    except Exception:
        return False

def run_cell(job):
    suite, seed, arm = job
    out = cell_out(suite, seed, arm)
    if done(suite, seed, arm):
        return job, "skip", 0.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = out.replace(".json", ".log")
    cmd = [PY, "-u", SCRIPT, "--method", arm, "--ckpt", CKPT, "--suite", suite,
           "--per-cat", str(PER_CAT), "--max-steps", str(MAXSTEPS[suite]),
           "--seed", str(seed), "--out", out]
    if arm == "aegis":
        cmd += ["--rib-weights", RIB, "--rasf-weights", RASF]
    with _LAUNCH_LOCK:          # stagger heavy CUDA inits
        time.sleep(4)
    t0 = time.monotonic()
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.monotonic() - t0
    ok = done(suite, seed, arm)
    return job, ("ok" if (rc == 0 and ok) else f"FAIL rc={rc} ok={ok}"), dt

def main():
    jobs = [(s, sd, a) for s in SUITES for sd in SEEDS for a in ARMS]
    # long-first: the 520-step libero_10 cells are slowest; start them in the first wave so
    # they finish inside the window instead of tailing out after the lighter cells.
    jobs.sort(key=lambda j: (0 if j[0] == "libero_10" else 1))
    todo = [j for j in jobs if not done(*j)]
    print(f"[plan] {len(jobs)} cells, {len(todo)} to run, per_cat={PER_CAT} "
          f"(n={expected_n()}/cell), conc={CONC}", flush=True)
    for j in jobs:
        if done(*j):
            print(f"  [skip] {j}", flush=True)
    t0 = time.monotonic()
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(run_cell, j): j for j in todo}
        for f in as_completed(futs):
            job, status, dt = f.result()
            tag = "ok" if status == "ok" else status
            if status == "ok": n_ok += 1
            elif status != "skip": n_fail += 1
            el = time.monotonic() - t0
            print(f"  [{el/60:5.1f}m] {job} -> {tag} ({dt/60:.1f}m)", flush=True)
    print(f"[done] ok={n_ok} fail={n_fail} total={ (time.monotonic()-t0)/60:.1f}m", flush=True)

if __name__ == "__main__":
    main()
