#!/usr/bin/env python
"""Preflight validator for eval jobs -- run BEFORE committing GPU-hours to a sweep.

Exercises the exact code paths and device placement a real eval uses, on dummy
data (no sim, no env build), so a config error / missing weight / device-mismatch
crash is caught in seconds instead of after a sweep silently "completes" with zero
results (the action-noise failure that motivated this).

Per job it checks:
  1. config loads
  2. module builds (identity for vanilla) or loads from --weights
  3. module forward runs on the resolved device, returns finite A_hat of right shape
  4. corruption path: corr.apply(cuda_obs, name, sev, cuda_generator)   [as in eval]
  5. action-noise path: perturb_actions(cpu_action, std, cuda_generator) [the bug]

Usage:
  python scripts/preflight.py jobs.txt          # one "tag|config|weights|corruption|action_noise" per line
  echo "sib_n25|configs/sib.yaml|results/sib_b1e-4.pt||0.1" | python scripts/preflight.py -
Exit code is nonzero if ANY job fails -- gate your sweep on it:
  python scripts/preflight.py jobs.txt && bash run_sweep.sh
"""
import sys, traceback, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from sib.utils import load_config, resolve_device
from sib.bottleneck import build_module, SpectralActionModule
from sib import corruptions as corr


def _read_jobs(arg):
    f = sys.stdin if arg == "-" else open(arg)
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = (line.split("|") + ["", "", "", ""])[:5]
        yield {"tag": parts[0], "config": parts[1], "weights": parts[2] or None,
               "corruption": parts[3] or None, "action_noise": float(parts[4]) if parts[4] else 0.0}


def check_job(job, device):
    cfg = load_config(job["config"])
    H, d = 50, 7
    # module: load trained or build identity (mirrors eval.load_trained_module)
    if job["weights"]:
        ck = torch.load(job["weights"], map_location=device, weights_only=False)
        m = build_module(ck["config"], ck["H"], ck["d"]).to(device)
        m.load_state_dict(ck["module_state"])
        H, d = ck["H"], ck["d"]
    else:
        m = SpectralActionModule("gain_no_rate", H, d).to(device)
    m.eval()
    A = torch.randn(4, H, d, device=device)
    out = m(A, update_lambda=False)
    assert out["A_hat"].shape == A.shape, "A_hat shape mismatch"
    assert torch.isfinite(out["A_hat"]).all(), "A_hat non-finite"
    # eval's shared generator lives on `device`
    g = torch.Generator(device=device).manual_seed(0)
    # corruption path: obs on device, generator on device (as eval applies it)
    if job["corruption"]:
        name, sev = job["corruption"].split(":")
        obs = torch.rand(2, 3, 32, 32, device=device)
        oc = corr.apply(obs, name, int(sev), generator=g)
        assert oc.shape == obs.shape and torch.isfinite(oc).all(), "corruption broke"
    # action-noise path: action on CPU (post env_postprocessor) + CUDA generator (the bug)
    if job["action_noise"] > 0:
        act = torch.zeros(2, d)  # CPU on purpose
        an = corr.perturb_actions(act, job["action_noise"], generator=g)
        assert an.shape == act.shape and torch.isfinite(an).all(), "action-noise broke"


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    device = resolve_device({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    jobs = list(_read_jobs(sys.argv[1]))
    if not jobs:
        print("no jobs to check"); sys.exit(2)
    fails = 0
    for j in jobs:
        try:
            check_job(j, device)
            print(f"  PASS  {j['tag']:16s} {j['config']} "
                  f"{'corr='+j['corruption'] if j['corruption'] else ''}"
                  f"{' an='+str(j['action_noise']) if j['action_noise'] else ''}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {j['tag']:16s} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\npreflight: {len(jobs)-fails}/{len(jobs)} jobs OK"
          + ("" if not fails else f"  -- {fails} FAILED, do NOT launch the sweep"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
