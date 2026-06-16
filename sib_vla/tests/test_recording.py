"""Rollout recording: frame coercion, mp4/npz writing, recorder manifest + caps."""

import json

import numpy as np
import torch

from sib.recording import (RolloutRecorder, extract_frames, save_trajectory,
                           save_video, to_uint8_frame)


def test_to_uint8_frame_variants():
    # float CHW [0,1] -> uint8 HWC3
    f = to_uint8_frame(torch.rand(3, 8, 8))
    assert f.shape == (8, 8, 3) and f.dtype == np.uint8
    # grayscale HW -> HWC3
    assert to_uint8_frame(np.random.rand(8, 8)).shape == (8, 8, 3)
    # uint8 HWC passes through
    u8 = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    assert to_uint8_frame(u8).shape == (8, 8, 3)
    # RGBA -> RGB
    assert to_uint8_frame(np.zeros((8, 8, 4), dtype=np.uint8)).shape == (8, 8, 3)


def test_save_video_writes_file(tmp_path):
    frames = [torch.rand(3, 16, 16) for _ in range(8)]
    out = save_video(frames, tmp_path / "ep.mp4", fps=10)
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_save_video_empty_is_noop(tmp_path):
    assert save_video([], tmp_path / "ep.mp4") is None


def test_save_trajectory_roundtrip(tmp_path):
    actions = np.random.randn(40, 7).astype(np.float32)
    p = save_trajectory(tmp_path / "ep.npz", actions, success=True, rms_jerk=0.1)
    d = np.load(p)
    assert d["actions"].shape == (40, 7)
    assert bool(d["success"]) is True
    assert np.isclose(float(d["rms_jerk"]), 0.1)


def test_extract_frames_from_batch():
    batch = {"observation.images.cam": torch.rand(2, 3, 12, 12),
             "observation.state": torch.rand(2, 7)}
    frames = extract_frames(batch)                       # auto-pick image key
    assert len(frames) == 2 and frames[0].shape == (12, 12, 3)
    # 5-D (B,T,C,H,W) -> last step
    batch5 = {"observation.images.cam": torch.rand(2, 4, 3, 12, 12)}
    assert len(extract_frames(batch5)) == 2


def test_recorder_caps_videos_and_writes_manifest(tmp_path):
    rec = RolloutRecorder(tmp_path, "sib_test", condition="clean",
                          video=True, max_videos_per_task=2)
    frames = [torch.rand(3, 12, 12) for _ in range(5)]
    actions = np.random.randn(20, 7).astype(np.float32)
    for ep in range(4):                                  # 4 episodes, cap 2 videos
        rec.write_episode("task0", ep, frames, actions, success=(ep % 2 == 0))
    idx = rec.finalize()
    man = json.loads(idx.read_text())
    assert man["n_episodes"] == 4
    assert man["n_videos"] == 2                          # capped
    # all 4 episodes have trajectory traces
    assert all("trajectory" in e for e in man["episodes"])
