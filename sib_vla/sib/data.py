"""Data access for LIBERO via LeRobot, and the cached chunk-pair dataset.

Design note (why a cache).  The spec forbids backprop through the flow-matching
sampler and runs it under ``no_grad``.  The frozen policy is therefore a fixed
function of the observation: the predicted chunk ``A`` for a given sample never
changes during bottleneck training.  So we run the policy over the train set
*once* (``scripts/estimate_lambda.py``), cache the pairs ``(A, A_star)`` -- the
predicted chunk and the ground-truth chunk, both in the policy's **normalised
action space** -- and train the tiny bottleneck on those tensors.  This is both
correct (identical objective) and the budget-optimal use of the A100.

Action space.  ``predict_action_chunk`` returns actions in the policy's internal
normalised space (LeRobot 0.4.3 unnormalises later via a post-processor).  The
distortion target ``A_star`` must live in that same space, so we normalise the
raw ground-truth chunk with the policy's action statistics and mode.  Both
``mean_std`` and ``min_max`` (LeRobot's [-1, 1] convention) are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Normalisation between raw and policy-normalised action space.
# --------------------------------------------------------------------------- #
@dataclass
class ActionNorm:
    """Per-dimension action normalisation statistics and mode.

    ``mode`` is ``"mean_std"`` (z-score) or ``"min_max"`` (to [-1, 1]).
    Tensors are shape ``(d,)``.
    """
    mode: str
    a: Tensor  # mean (mean_std) or min (min_max)
    b: Tensor  # std  (mean_std) or max (min_max)
    eps: float = 1e-8

    def normalize(self, x: Tensor) -> Tensor:
        a, b = self.a.to(x), self.b.to(x)
        if self.mode == "mean_std":
            return (x - a) / (b + self.eps)
        if self.mode == "min_max":
            return 2.0 * (x - a) / (b - a + self.eps) - 1.0
        raise ValueError(f"unknown mode {self.mode!r}")

    def denormalize(self, y: Tensor) -> Tensor:
        a, b = self.a.to(y), self.b.to(y)
        if self.mode == "mean_std":
            return y * (b + self.eps) + a
        if self.mode == "min_max":
            return (y + 1.0) * 0.5 * (b - a + self.eps) + a
        raise ValueError(f"unknown mode {self.mode!r}")

    def state_dict(self) -> dict:
        return {"mode": self.mode, "a": self.a.cpu(), "b": self.b.cpu(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "ActionNorm":
        return cls(mode=sd["mode"], a=sd["a"], b=sd["b"], eps=sd.get("eps", 1e-8))


def action_norm_from_lerobot(stats: dict, mode: str) -> ActionNorm:
    """Build :class:`ActionNorm` from a LeRobot ``meta.stats['action']`` dict.

    ``stats`` holds tensors/arrays under keys ``mean``/``std`` or ``min``/``max``.
    """
    def t(x):
        return torch.as_tensor(x, dtype=torch.float32).flatten()
    if mode == "mean_std":
        return ActionNorm("mean_std", t(stats["mean"]), t(stats["std"]))
    if mode == "min_max":
        return ActionNorm("min_max", t(stats["min"]), t(stats["max"]))
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# LeRobot dataset access (used by the one-off precompute / lambda estimation).
# --------------------------------------------------------------------------- #
def make_action_delta_timestamps(fps: float, H: int) -> dict:
    """Delta-timestamps that fetch an ``H``-step action chunk starting at ``t``."""
    return {"action": [i / fps for i in range(H)]}


def build_lerobot_dataset(repo_id: str, H: int, fps: float,
                          root: Optional[str] = None, episodes=None):
    """Open a LeRobotDataset configured to return ``H``-step action chunks.

    Thin wrapper; kept here so the import stays in one place and the
    ``delta_timestamps`` convention is documented.  Verify ``repo_id``/``fps``
    match your LIBERO checkpoint's training data.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=make_action_delta_timestamps(fps, H),
    )


# --------------------------------------------------------------------------- #
# Cached (A, A_star) pair dataset -- what the bottleneck actually trains on.
# --------------------------------------------------------------------------- #
@dataclass
class ChunkCache:
    """A cache of predicted/target chunks plus the action normalisation.

    ``A`` and ``A_star_raw`` are ``(N, H, d)``.  ``A`` is already in normalised
    space (it is the policy output); ``A_star_raw`` is raw and is normalised on
    the fly by :class:`CachedPairDataset` using ``norm``.
    """
    A: Tensor
    A_star_raw: Tensor
    norm: ActionNorm
    context: Optional[Tensor] = None   # (N, C) optional context embedding

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"A": self.A.cpu(), "A_star_raw": self.A_star_raw.cpu(),
             "norm": self.norm.state_dict(),
             "context": None if self.context is None else self.context.cpu()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChunkCache":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(A=d["A"], A_star_raw=d["A_star_raw"],
                   norm=ActionNorm.from_state_dict(d["norm"]),
                   context=d.get("context"))


class CachedPairDataset(Dataset):
    """Yields ``(A, A_star_norm[, context])`` for bottleneck training."""

    def __init__(self, cache: ChunkCache) -> None:
        self.A = cache.A
        self.A_star = cache.norm.normalize(cache.A_star_raw)
        self.context = cache.context
        if self.A.shape != self.A_star.shape:
            raise ValueError("A and A_star shape mismatch")

    def __len__(self) -> int:
        return self.A.shape[0]

    def __getitem__(self, i: int):
        if self.context is None:
            return self.A[i], self.A_star[i]
        return self.A[i], self.A_star[i], self.context[i]
