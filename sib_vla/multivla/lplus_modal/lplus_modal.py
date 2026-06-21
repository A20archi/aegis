"""LIBERO-Plus (arXiv 2510.13626) robustness eval on Modal — the paper-protocol robustness
proof. AEGIS (RIB perception leg, +RASF) vs frozen SmolVLA, per-category, per-suite gated.

Reuses the PROVEN SmolVLA image recipe (lerobot[smolvla]+sim stack) and ADDS:
  - ImageMagick + Wand (LIBERO-Plus env_wrapper.py uses wand for texture/lighting perturb)
  - the LIBERO-plus repo's OWN `libero` (perturbed bddls) on PYTHONPATH (NOT stock LIBERO)

Cost discipline (mirrors smolvla_modal):
  modal run lplus_modal.py::main --stage validate   # CHEAP CPU: imports + wand + one EGL env
  modal run lplus_modal.py::main --stage smoke       # 1 cheap L4 cell, tiny
  modal run lplus_modal.py::main --stage stage1       # per-suite x {baseline,aegis}, cap'd
L4 GPU, cpu=8, max_containers=4, resume-skip, incremental per-cat commit. Gate off where
AEGIS <= base (report base) — provably never worse.
"""
import json
import modal

CUDA = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
LPLUS = "https://github.com/sylvestf/LIBERO-plus.git"   # repo with perturbed bddls
image = (
    modal.Image.from_registry(CUDA, add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libegl1", "libgles2", "libosmesa6",
                 "libglew-dev", "ffmpeg", "patchelf", "libx11-6", "wget", "clang",
                 "libevdev-dev", "cmake", "build-essential", "libegl1-mesa-dev",
                 "libgles2-mesa-dev", "libosmesa6-dev",
                 "imagemagick", "libmagickwand-dev")   # Wand backend for LIBERO-Plus perturb
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("lerobot[smolvla]==0.4.3")
    .run_commands("pip uninstall -y cmake && cmake --version")
    .pip_install("robosuite==1.4.0", "robomimic==0.3.0", "mujoco==3.4.0", "bddl==3.6.0",
                 "gym==0.26.2", "gymnasium==1.2.3", "easydict==1.13", "termcolor==3.3.0",
                 "thop", "numpy==2.1.3", "matplotlib==3.10.0", "omegaconf==2.3.0",
                 "Wand==0.7.1", "scikit-image", "scipy")   # LIBERO-Plus corruptions deps
    # LIBERO-plus repo provides its OWN `libero` namespace pkg (perturbed bddls + env_wrapper).
    # Put it (not stock LIBERO) on PYTHONPATH. Seed its interactive config non-interactively.
    .run_commands(
        f"git clone --depth 1 {LPLUS} /opt/LIBERO-plus",
        "mkdir -p /opt/lplus_cfg && echo N | env LIBERO_CONFIG_PATH=/opt/lplus_cfg "
        "PYTHONPATH=/opt/LIBERO-plus python -c 'import libero.libero; "
        "from libero.libero import get_libero_path; print(\"LP_OK\", get_libero_path(\"bddl_files\"))'",
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HUB_OFFLINE": "0",
          "LIBERO_CONFIG_PATH": "/opt/lplus_cfg", "LIBERO_PLUS_REPO": "/opt/LIBERO-plus",
          "PYTHONPATH": "/assets/sib_vla:/opt/LIBERO-plus"})
)

app = modal.App("lplus-robust")
assets = modal.Volume.from_name("smolvla-assets", create_if_missing=True)   # reuse same vol

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ARMS = ["baseline", "aegis"]


@app.function(image=image, cpu=2.0, memory=8192, timeout=1200, volumes={"/assets": assets})
def validate():
    """CPU: confirm wand + LIBERO-plus libero + lerobot.envs import, task json, one EGL env."""
    import os
    out = {}
    import torch, lerobot, wand; out["torch"] = torch.__version__
    out["lerobot"] = lerobot.__version__; out["wand"] = wand.version.VERSION
    import libero.libero  # from /opt/LIBERO-plus
    out["libero_file"] = str(getattr(libero, "__file__", None))
    tc = json.load(open(f"{os.environ['LIBERO_PLUS_REPO']}/libero/libero/benchmark/task_classification.json"))
    out["suites"] = list(tc.keys())
    out["spatial_ntasks"] = len(tc["libero_spatial"])
    # wand-dependent env wrapper must import (the linker gotcha) — before heavy torch use
    import libero.libero.envs.env_wrapper  # noqa
    out["env_wrapper_ok"] = True
    from lerobot.envs.configs import LiberoEnv
    out["lerobot_LiberoEnv_ok"] = True
    out["assets_listing"] = os.listdir("/assets")
    return out


@app.local_entrypoint()
def main(stage: str = "validate"):
    if stage == "validate":
        print("VALIDATE:", validate.remote())
    else:
        print("stage = validate  (smoke/stage1 added after validate passes)")
