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
                 "libglew-dev", "ffmpeg", "patchelf", "libx11-6", "wget", "unzip", "clang",
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
                 "Wand==0.7.1", "scikit-image", "scipy",
                 "imageio-ffmpeg")   # LIBERO-Plus corruptions + mp4 writing (imageio ffmpeg plugin)
    # LIBERO-plus repo provides its OWN `libero` namespace pkg (perturbed bddls + env_wrapper).
    # Put it (not stock LIBERO) on PYTHONPATH. Seed its interactive config non-interactively.
    .run_commands(
        f"git clone --depth 1 {LPLUS} /opt/LIBERO-plus",
        "mkdir -p /opt/lplus_cfg && echo N | env LIBERO_CONFIG_PATH=/opt/lplus_cfg "
        "PYTHONPATH=/opt/LIBERO-plus python -c 'import libero.libero; "
        "from libero.libero import get_libero_path; print(\"LP_OK\", get_libero_path(\"bddl_files\"))'",
    )
    # LIBERO-Plus ships NO assets in git — the perturbation scenes/textures/objects (~6.4GB)
    # live in assets.zip on HF and must be unzipped to libero/libero/ (per the repo README),
    # else env build dies (FileNotFoundError: .../assets/scenes/libero_floor_base_style.xml).
    # CPU-only build step (no GPU cost); cached after first build.
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download; "
        "hf_hub_download('Sylvest/LIBERO-plus','assets.zip',repo_type='dataset',local_dir='/tmp/lpa')\"",
        "cd /tmp/lpa && unzip -q -o assets.zip && rm -f assets.zip",
        # the zip nests the real assets/ deep (inspire/.../LIBERO-plus-0/assets/) — locate it by
        # the marker scene file and lift it to libero/libero/assets (move, or merge if one exists).
        "SRC=$(dirname $(dirname $(find /tmp/lpa -name libero_floor_base_style.xml | head -1))); "
        "echo \"SRC=$SRC\"; "
        "if [ -d /opt/LIBERO-plus/libero/libero/assets ]; then cp -rn \"$SRC\"/. /opt/LIBERO-plus/libero/libero/assets/; "
        "else mv \"$SRC\" /opt/LIBERO-plus/libero/libero/assets; fi; rm -rf /tmp/lpa",
        "test -f /opt/LIBERO-plus/libero/libero/assets/scenes/libero_floor_base_style.xml "
        "&& echo LPLUS_ASSETS_OK || (echo LPLUS_ASSETS_MISSING; exit 1)",
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


A = "/assets/sib_vla"                       # uploaded source root on the volume
CKPT = f"{A}/outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model"
RIB = f"{A}/results/ib_on86/rib_on86.pt"
RASF = f"{A}/results/rasf_on86/rasf_on86.pt"


# per-suite max-steps (CRITICAL: libero_10/Long needs 520 or long-horizon SR silently dies).
LPLUS_MAXSTEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def _cells_lplus(per_cat=40, suites=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
                 seeds=(42, 123, 456)):
    """LIBERO-Plus eval cells: seed × suite × {baseline, aegis}. EVAL-ONLY (existing
    checkpoints; no training). per_cat tasks/category × 7 categories per suite, per seed.
    Default = the PAPER table: ALL 4 suites × 3 seeds × 2 arms = 24 cells, isolated under
    <suite>/seed<N>/ -> mean ± CI, resume-safe. Each cell carries its suite's max-steps."""
    return [{"suite": sk, "arm": arm, "per_cat": per_cat, "seed": seed,
             "max_steps": LPLUS_MAXSTEPS[sk]}
            for seed in seeds for sk in suites for arm in ARMS]


def _cells_lplus_video(suite="libero_object", per_cat=1, vids_per_cat=1, seed=42):
    """Curated LIBERO-Plus-native pairwise videos: SAME suite/seed for baseline & aegis ->
    matched side-by-sides. Records 1 task/category (7 categories) per arm. Tiny + cheap.
    mp4s -> /assets/results_modal/liberoplus_video/<suite>/videos/<category>/<arm>_task<id>.mp4"""
    rd = f"/assets/results_modal/liberoplus_video/{suite}/videos"
    return [{"suite": suite, "arm": arm, "per_cat": per_cat, "seed": seed,
             "max_steps": LPLUS_MAXSTEPS[suite], "od": "liberoplus_video",
             "record_dir": rd, "videos_per_cat": vids_per_cat} for arm in ARMS]


# L4 + cpu=8: LIBERO-Plus rollouts are MuJoCo/EGL sim-bound (not GPU). max_containers caps
# in-flight cells; resume-skip + commit daemon = "save every inch".
@app.function(image=image, gpu="L4", cpu=8.0, memory=32768, timeout=14400,
              max_containers=4, volumes={"/assets": assets})
def eval_cell(cell: dict):
    """One LIBERO-Plus cell (suite, arm). Streams progress; resume-skips a finished JSON."""
    import os, subprocess, sys, threading
    sk, arm, pc = cell["suite"], cell["arm"], cell.get("per_cat", 30)
    seed = cell.get("seed")
    seed_sub = f"/seed{seed}" if seed is not None else ""
    od = f"/assets/results_modal/liberoplus/{sk}{seed_sub}"
    os.makedirs(od, exist_ok=True)
    out = f"{od}/{arm}.json"
    if os.path.exists(out):                  # resume-skip: a written JSON == this cell is done
        try:
            d = json.load(open(out))
            if d.get("n_episodes", 0) > 0:
                return {"cell": cell, "rc": 0, "sr": round(d["total_sr"] * 100, 1),
                        "n": d["n_episodes"], "resumed": True}
        except Exception:
            pass
    cmd = ["/usr/local/bin/python", "-u", f"{A}/scripts/libero_plus_aegis_eval.py",
           "--method", arm, "--ckpt", CKPT, "--suite", sk, "--per-cat", str(pc), "--out", out,
           "--max-steps", str(cell.get("max_steps", 280))]   # per-suite; Long=520
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if cell.get("record_dir"):                       # LIBERO-Plus-native pairwise videos
        cmd += ["--record-dir", cell["record_dir"],
                "--videos-per-cat", str(cell.get("videos_per_cat", 1))]
        if cell.get("cats"):
            cmd += ["--cats", *cell["cats"]]
    if arm == "aegis":
        cmd += ["--rib-weights", RIB, "--rasf-weights", RASF]
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "LIBERO_PLUS_REPO": "/opt/LIBERO-plus", "LIBERO_CONFIG_PATH": "/opt/lplus_cfg",
           "PYTHONPATH": f"{A}:/opt/LIBERO-plus"}
    tag = f"{sk}/{arm}" + (f"/s{seed}" if seed is not None else "")
    print(f"[lplus START] {tag} per_cat={pc}", flush=True)
    stop = threading.Event()
    def _commit():
        while not stop.wait(120):
            try: assets.commit()
            except Exception as e: print(f"[commit] warn: {e}", flush=True)
    threading.Thread(target=_commit, daemon=True).start()
    p = subprocess.Popen(cmd, cwd=A, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        sys.stdout.write(f"[{tag}] {line}"); sys.stdout.flush()
        tail.append(line)
        if len(tail) > 80: tail.pop(0)
    p.wait(); stop.set()
    res = {"cell": cell, "rc": p.returncode}
    if os.path.exists(out):
        d = json.load(open(out)); res["sr"] = round(d["total_sr"] * 100, 1); res["n"] = d["n_episodes"]
    else:
        res["err"] = "".join(tail)[-1500:]
    print(f"[lplus DONE] {tag} rc={res['rc']} sr={res.get('sr', 'ERR')}", flush=True)
    assets.commit()
    return res


@app.local_entrypoint()
def main(stage: str = "validate", per_cat: int = 40):
    if stage == "validate":
        print("VALIDATE:", validate.remote())
    elif stage == "smoke":
        print("SMOKE:", eval_cell.remote({"suite": "libero_object", "arm": "aegis",
                                          "per_cat": 1, "seed": 42}))
    elif stage == "stage1":
        # PAPER table: seeds 42,123,456 × {spatial,object,goal,long} × {baseline,aegis} = 24 cells
        cells = _cells_lplus(per_cat)
        print(f"launching {len(cells)} LIBERO-Plus cells (4 suites × 3 seeds, per_cat={per_cat})...")
        for r in sorted(eval_cell.map(cells), key=lambda x: (x['cell']['seed'], x['cell']['suite'], x['cell']['arm'])):
            c = r["cell"]; print(f"  s{c['seed']} {c['suite']:14} {c['arm']:8} -> SR={r.get('sr','ERR')} (rc={r['rc']})")
    elif stage == "video":
        # curated LIBERO-Plus-native pairwise videos (object suite, 7 cats × {base,aegis})
        cells = _cells_lplus_video()
        print(f"launching {len(cells)} LIBERO-Plus VIDEO cells (pairwise)...")
        for r in eval_cell.map(cells):
            c = r["cell"]; print(f"  VID {c['suite']:14} {c['arm']:8} -> SR={r.get('sr','ERR')} (rc={r['rc']})")
    else:
        print("stage = validate | smoke | stage1 | video")
