"""Persist rollout artefacts: episode videos and trajectory traces.

Every LIBERO evaluation can keep (a) an **mp4 of what the policy saw** each step
(including any eval-time corruption, so corrupted runs look corrupted) and (b) a
**trajectory trace** ``.npz`` with the executed actions, success flag, and any
extra per-step signals.  Videos are capped per task to bound disk; traces are
cheap and kept for every recorded episode.

Layout under ``results/``::

    videos/<run>/<condition>/<task>/ep000.mp4
    videos/<run>/<condition>/<task>/ep000.npz
    videos/<run>/index.json                       # manifest of everything written

Writers degrade gracefully: mp4 via imageio-ffmpeg, else gif, else a stack of
PNG frames -- recording never crashes an eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def to_uint8_frame(img) -> np.ndarray:
    """Coerce a torch/numpy image to ``(H, W, 3)`` uint8.

    Accepts ``(H,W)``, ``(H,W,C)``, ``(C,H,W)``; float in ``[0,1]`` or uint8.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))          # CHW -> HWC
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        if np.nanmax(img) <= 1.0 + 1e-6:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def save_video(frames, path: str | Path, fps: int = 10) -> Optional[Path]:
    """Write ``frames`` to ``path`` as mp4 (else gif, else PNG stack)."""
    frames = [to_uint8_frame(f) for f in frames]
    if not frames:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio
        iio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception:
        try:
            import imageio.v2 as iio
            gif = path.with_suffix(".gif")
            iio.mimsave(gif, frames, duration=1.0 / max(fps, 1))
            return gif
        except Exception:
            from PIL import Image
            stem = path.with_suffix("")
            stem.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(stem / f"frame_{i:04d}.png")
            return stem


def save_trajectory(path: str | Path, actions, success: bool, **extra) -> Path:
    """Save executed actions ``(T, d)``, success, and any extra arrays as ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"actions": np.asarray(actions, dtype=np.float32),
               "success": np.asarray(bool(success))}
    for k, v in extra.items():
        payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)
    return path


def extract_frames(obs_batch: dict, camera_key: Optional[str] = None) -> list[np.ndarray]:
    """Per-env display frames from a preprocessed obs batch (pre policy-normalisation).

    Picks ``camera_key`` (or the first image key) of shape ``(B,C,H,W)`` /
    ``(B,T,C,H,W)``; returns a list of ``B`` uint8 ``(H,W,3)`` frames.
    """
    if camera_key is None:
        keys = [k for k, v in obs_batch.items() if "image" in k and torch.is_tensor(v) and v.dim() >= 4]
        if not keys:
            return []
        camera_key = sorted(keys)[0]
    x = obs_batch[camera_key]
    if x.dim() == 5:                                # (B, T, C, H, W) -> last step
        x = x[:, -1]
    return [to_uint8_frame(x[i]) for i in range(x.shape[0])]


class RolloutRecorder:
    """Collects frames/trajectories during a rollout and writes them out.

    ``max_videos_per_task`` caps mp4s (disk); every recorded episode still gets a
    trajectory ``.npz``.  Set ``video=False`` to keep only traces.
    """

    def __init__(self, out_dir: str | Path, run_name: str, *, condition: str = "clean",
                 fps: int = 10, video: bool = True, max_videos_per_task: int = 5,
                 camera_key: Optional[str] = None):
        self.root = Path(out_dir) / "videos" / run_name
        self.run_name = run_name
        self.condition = condition or "clean"
        self._cond_dir = self.condition.replace(":", "").replace("/", "_")  # path-safe
        self.fps = fps
        self.video = video
        self.max_videos_per_task = max_videos_per_task
        self.camera_key = camera_key
        self.manifest: list[dict] = []

    def episode_dir(self, task: str) -> Path:
        return self.root / self._cond_dir / str(task)

    def should_record_video(self, task: str, ep_idx: int) -> bool:
        return self.video and ep_idx < self.max_videos_per_task

    def write_episode(self, task: str, ep_idx: int, frames, actions,
                      success: bool, **extra) -> dict:
        d = self.episode_dir(task)
        stem = d / f"ep{ep_idx:03d}"
        entry = {"task": str(task), "episode": ep_idx, "condition": self.condition,
                 "success": bool(success), "steps": int(len(actions))}
        traj = save_trajectory(stem.with_suffix(".npz"), actions, success, **extra)
        entry["trajectory"] = str(traj)
        if self.should_record_video(task, ep_idx) and frames:
            vid = save_video(frames, stem.with_suffix(".mp4"), fps=self.fps)
            entry["video"] = None if vid is None else str(vid)
        self.manifest.append(entry)
        return entry

    def finalize(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        idx = self.root / "index.json"
        with open(idx, "w") as f:
            json.dump({"run": self.run_name, "condition": self.condition,
                       "n_episodes": len(self.manifest),
                       "n_videos": sum("video" in e for e in self.manifest),
                       "episodes": self.manifest}, f, indent=2)
        return idx
