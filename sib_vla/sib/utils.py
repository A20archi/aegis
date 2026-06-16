"""Shared plumbing: config inheritance, seeding, device, GPU-hour logging.

Every run is ``python scripts/<x>.py --config configs/<y>.yaml``; no experiment
parameters are hard-coded in the scripts.  Method configs inherit ``base.yaml``
via an ``inherit:`` key and overlay their own keys.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, over: dict) -> dict:
    """Recursively overlay ``over`` onto ``base`` (``over`` wins)."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving a single-level (or chained) ``inherit:``.

    ``inherit`` may be a string or list of paths relative to the config's dir.
    The current file overlays its parents.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parents = cfg.pop("inherit", None)
    if parents is None:
        return cfg
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, load_config(path.parent / parent))
    return deep_merge(merged, cfg)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_resolved_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------- #
# Determinism / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        want = "cpu"
    return torch.device(want)


# --------------------------------------------------------------------------- #
# GPU-hour accounting (Section 10 budget map)
# --------------------------------------------------------------------------- #
class GpuHourLogger:
    """Time a run and append GPU-hours to ``results/gpu_hours.csv``.

    Usage::

        with GpuHourLogger("sib_mve", out_dir, n_gpus=1):
            ...  # run
    """

    def __init__(self, run_name: str, out_dir: str | Path, n_gpus: int = 1):
        self.run_name = run_name
        self.csv_path = Path(out_dir) / "gpu_hours.csv"
        self.n_gpus = n_gpus
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.t0
        gpu_hours = elapsed / 3600.0 * self.n_gpus
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run_name", "seconds", "gpu_hours", "n_gpus", "status"])
            w.writerow([self.run_name, f"{elapsed:.1f}", f"{gpu_hours:.4f}",
                        self.n_gpus, "error" if exc_type else "ok"])
        return False  # never suppress exceptions
