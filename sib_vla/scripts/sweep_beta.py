"""Stage 3 frontier: train + eval the SIB module across a list of betas.

Each beta is a separate, fully config-driven run (train then eval), so the
rate-distortion frontier (bits vs success vs jerk) is just the collection of
``results/eval_sib_beta*.json`` files.

    python scripts/sweep_beta.py --config configs/sweep_beta.yaml
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys
from pathlib import Path

from sib.utils import load_config

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    betas = cfg["betas"]
    base_cfg = cfg.get("method_config", "configs/sib.yaml")
    print(f"[sweep] betas={betas} from {base_cfg}")

    for beta in betas:
        tag = f"sib_beta{beta}"
        weights = Path(cfg["output_dir"]) / f"{tag}.pt"
        if not args.skip_train:
            run([sys.executable, str(HERE / "train.py"),
                 "--config", base_cfg, "--beta", str(beta), "--tag", tag])
        if not args.skip_eval:
            run([sys.executable, str(HERE / "eval.py"),
                 "--config", base_cfg, "--weights", str(weights), "--tag", tag])
    print(f"[sweep] done: {len(betas)} points")


if __name__ == "__main__":
    main()
