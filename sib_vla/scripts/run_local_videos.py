#!/usr/bin/env python
"""Record LIBERO-Plus rollout mp4s (baseline vs AEGIS, same seed = pairwise side-by-sides).

Low concurrency so it never starves the main table run. 1 task/category x both arms x all
suites -> 7 paired clips per suite, written to results/local_lplus_video/<suite>/<cat>/.
"""
import os, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi/sib_vla"
PY = "/home/user/miniconda3/envs/lerobot_lplus/bin/python"
SCRIPT = f"{ROOT}/scripts/libero_plus_aegis_eval.py"
CKPT = f"{ROOT}/outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model"
RIB = f"{ROOT}/results/ib_on86/rib_on86.pt"
RASF = f"{ROOT}/results/rasf_on86/rasf_on86.pt"
VIDDIR = f"{ROOT}/results/local_lplus_video"
LPLUS_REPO = "/home/user/Desktop/vla_projects/LIBERO-plus"
MAXSTEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SUITES = os.environ.get("SUITES", "libero_object,libero_spatial,libero_goal,libero_10").split(",")
ARMS = ["baseline", "aegis"]
SEED = 42
CONC = int(os.environ.get("VCONC", "3"))

ENV = {**os.environ, "LIBERO_PLUS_REPO": LPLUS_REPO,
       "LIBERO_CONFIG_PATH": "/home/user/Desktop/vla_projects/.libero_lplus",
       "PYTHONPATH": f"{ROOT}:{LPLUS_REPO}",
       "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HUB_OFFLINE": "1"}
_LK = threading.Lock()


def run(job):
    suite, arm = job
    rd = f"{VIDDIR}/{suite}"
    log = f"{VIDDIR}/{suite}_{arm}.log"
    os.makedirs(rd, exist_ok=True)
    cmd = [PY, "-u", SCRIPT, "--method", arm, "--ckpt", CKPT, "--suite", suite, "--per-cat", "1",
           "--max-steps", str(MAXSTEPS[suite]), "--seed", str(SEED),
           "--record-dir", rd, "--videos-per-cat", "1",
           "--out", f"{VIDDIR}/{suite}_{arm}.json"]
    if arm == "aegis":
        cmd += ["--rib-weights", RIB, "--rasf-weights", RASF]
    with _LK:
        time.sleep(4)
    t0 = time.monotonic()
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    return job, rc, time.monotonic() - t0


def main():
    jobs = [(s, a) for s in SUITES for a in ARMS]
    print(f"[vid] {len(jobs)} record-jobs, conc={CONC}", flush=True)
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for f in as_completed({ex.submit(run, j): j for j in jobs}):
            j, rc, dt = f.result()
            print(f"  [vid] {j} rc={rc} ({dt/60:.1f}m)", flush=True)
    print("[vid] done", flush=True)


if __name__ == "__main__":
    main()
