"""SmolVLA robustness on Modal — MAX-PARALLEL. Moves Track-B (LIBERO-V Object+Goal
robustness, + ablations) off the single local A100 to many Modal GPUs at once.

Cost discipline / gates:
  modal run smolvla_modal.py::main --stage validate   # CHEAP CPU: deps + EGL sim import
  modal run smolvla_modal.py::main --stage smoke       # 1 cheap GPU cell, 2 episodes
  modal run smolvla_modal.py::main --stage stage1       # PARALLEL: 24 cells fan out
Every function declares memory= (no RAM OOM). AEGIS gate open = --method aegis
(RIB+RASF); gate closed = --method baseline (≡ base). Per-suite gating handled by
which arm we report.
"""
import json
import modal

CUDA = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
image = (
    modal.Image.from_registry(CUDA, add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libegl1", "libgles2",
                 "libosmesa6", "libglew-dev", "ffmpeg", "patchelf", "libx11-6", "wget",
                 "clang", "libevdev-dev",          # clang: build evdev (lerobot teleop) C ext
                 "cmake", "build-essential",       # egl_probe (robosuite dep) cmake build
                 "libegl1-mesa-dev", "libgles2-mesa-dev", "libosmesa6-dev")  # EGL headers
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    # lerobot owns its EXACT pinned tree (draccus==0.10.0, opencv-python-headless<4.13,
    # diffusers<0.36, accelerate, einops, safetensors, imageio[ffmpeg], torchcodec); the
    # [smolvla] extra pulls transformers>=4.57.1 — the SmolVLA policy deps. Do NOT re-pin
    # these: over-pinning past lerobot's exact versions is what made the resolver fail.
    .pip_install("lerobot[smolvla]==0.4.3")
    # lerobot drags a PIP cmake (a /usr/local/bin/cmake python shim) that fails inside
    # egl_probe's isolated build ('No module named cmake'). Remove it so egl_probe builds
    # against the real apt cmake (/usr/bin). lerobot never calls cmake at runtime.
    .run_commands("pip uninstall -y cmake && cmake --version")
    # sim stack lerobot does NOT ship — versions verified coexisting in the working env.
    .pip_install("robosuite==1.4.0", "robomimic==0.3.0", "mujoco==3.4.0", "bddl==3.6.0",
                 "gym==0.26.2", "gymnasium==1.2.3", "easydict==1.13", "termcolor==3.3.0",
                 "thop", "numpy==2.1.3", "matplotlib==3.10.0", "omegaconf==2.3.0")
    # `libero` is a NAMESPACE package (no top-level __init__; libero.__file__ is None) and
    # the working env uses LEGACY egg-link editable = repo root on sys.path. PEP 660 editable
    # and the git wheel both fail to map the namespace. Replica: clone + put repo on PYTHONPATH.
    .run_commands(
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        # LIBERO's __init__ prompts interactively on first import to create config.yaml
        # (EOFError in a non-interactive build). Auto-answer 'N' (use default paths) with a
        # FIXED LIBERO_CONFIG_PATH so the config is baked into the image deterministically.
        "mkdir -p /opt/libero_cfg && echo N | env LIBERO_CONFIG_PATH=/opt/libero_cfg "
        "PYTHONPATH=/opt/LIBERO python -c 'from libero.libero import get_libero_path; "
        "print(\"BDDL:\", get_libero_path(\"bddl_files\"))' && cat /opt/libero_cfg/config.yaml",
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HUB_OFFLINE": "0",
          "LIBERO_CONFIG_PATH": "/opt/libero_cfg",
          "PYTHONPATH": "/assets/sib_vla:/opt/LIBERO"})
)

app = modal.App("smolvla-robust")
assets = modal.Volume.from_name("smolvla-assets", create_if_missing=True)

# ---- the robustness sweep: Object+Goal × 6 axes × {baseline, aegis} = 24 cells ----
# suite -> lerobot LiberoEnv name. episode_length (max-steps) per suite is VERIFIED against
# the local configs: Spatial=220, Object=280, Goal=300, Long(libero_10)=520 (Long MUST be
# 520 — the project notes flag that truncating it at 300 silently kills long-horizon SR).
# spatial = the ABLATION suite (modules' training distribution; the +14.1 headline).
# long    = LIBERO-Long robustness stress test (same 6 axes as the V grid, long-horizon).
SUITES = {"object": "libero_object", "goal": "libero_goal",
          "spatial": "libero_spatial", "long": "libero_10"}
ELEN = {"object": 280, "goal": 300, "spatial": 220, "long": 520}
AXES = ["gaussian_noise_1", "motion_blur_1", "lighting_1", "texture_1",
        "viewpoint_medium", "viewpoint_large"]
ARMS = ["baseline", "aegis"]            # gate closed / gate open


def _cells_stage1(episodes=10, n_envs=10):
    # The cross-suite grid is Object+Goal ONLY (pinned explicitly so adding suites to SUITES
    # for ablation/long can't silently balloon stage1). n_envs=10 -> n=100/condition.
    # record=False: recording every cell cost ~17min/task; demo videos are a separate pass.
    cells = []
    for sk in ("object", "goal"):
        for axis in AXES:
            for arm in ARMS:
                cells.append({"suite": sk, "axis": axis, "arm": arm,
                              "episodes": episodes, "n_envs": n_envs, "record": False})
    return cells


def _cells_long(episodes=10, n_envs=10):
    """LIBERO-Long (libero_10) robustness — the SAME 6 axes × {baseline, aegis} as the V grid,
    on the long-horizon suite (episode_length 520). 12 cells. Separate od so it never clashes
    with the Object/Goal grid; identical protocol so deltas are directly comparable."""
    cells = []
    for axis in AXES:
        for arm in ARMS:
            cells.append({"suite": "long", "axis": axis, "arm": arm, "episodes": episodes,
                          "n_envs": n_envs, "record": False, "od": "liberov_long"})
    return cells


def _cells_video():
    # Curated demo footage only: a few visually compelling axes, BOTH arms (side-by-side
    # baseline-fail / AEGIS-succeed), tiny (n_envs=2, tasks 0-1) -> ~minutes, ~$2 total.
    vid_axes = ["motion_blur_1", "gaussian_noise_1", "viewpoint_medium"]
    cells = []
    for axis in vid_axes:
        for arm in ARMS:
            cells.append({"suite": "object", "axis": axis, "arm": arm,
                          "episodes": 2, "n_envs": 2, "tasks": "0,1", "record": True,
                          "od": "liberov_video"})   # separate dir -> no clash with SR grid
    return cells


def _cells_ablation(episodes=10, n_envs=10, suite="spatial"):
    """Full ablation table on the Spatial suite (modules' training distribution), 6 axes.
    ALL checkpoints already exist -> eval-only, NO retraining. Each arm gets its OWN `od`
    so arms that share --method aegis (RIB-only / RASF-only) never clobber each other or
    the real grid. method= the eval --method (== the output subdir eval writes to).

    Per-locus attribution (de-strength flags, no retrain):
      vanilla   no TE, no modules                    -> floor (isolates TE via vanilla->baseline)
      baseline  SmolVLA + TE                          -> honest reference
      ribonly   aegis + RASF off (--rasf-gate-max 0)  -> perception leg alone
      rasfonly  aegis + RIB off  (--rib-fusion-scale 0)-> action leg alone
      aegis     full RIB + RASF + TE                  -> the method
    Design ablations (defend the novelty claims; separate trained checkpoints):
      raw_vib   sib w/o DCT  (raw_vib_b1e-3.pt)        -> spectral basis matters
      no_rate   sib w/o rate (gain_no_rate.pt)         -> the rate/IB objective matters
      naive_ib  naive bottleneck (ib_on86.pt)          -> a naive IB is dormant (StableVLA contrast)
    """
    RIB, RASF = "{A}/results/ib_on86/rib_on86.pt", "{A}/results/rasf_on86/rasf_on86.pt"
    arms = {
        "vanilla":  {"method": "vanilla", "te": False, "method_args": ["--method", "vanilla"]},
        "baseline": {"method": "baseline", "method_args": ["--method", "baseline"]},
        "ribonly":  {"method": "aegis", "method_args": ["--method", "aegis", "--rib-weights", RIB,
                     "--rasf-weights", RASF, "--rasf-gate-max", "0"]},
        "rasfonly": {"method": "aegis", "method_args": ["--method", "aegis", "--rib-weights", RIB,
                     "--rasf-weights", RASF, "--rib-fusion-scale", "0"]},
        "aegis":    {"method": "aegis", "method_args": ["--method", "aegis", "--rib-weights", RIB,
                     "--rasf-weights", RASF]},
        "raw_vib":  {"method": "sib", "method_args": ["--method", "sib", "--weights",
                     "{A}/results/raw_vib_b1e-3.pt"]},
        "no_rate":  {"method": "sib", "method_args": ["--method", "sib", "--weights",
                     "{A}/results/gain_no_rate.pt"]},
        "naive_ib": {"method": "ib", "method_args": ["--method", "ib", "--weights",
                     "{A}/results/ib_on86/ib_on86.pt"]},
    }
    cells = []
    for arm, spec in arms.items():
        for axis in AXES:
            cells.append({"suite": suite, "axis": axis, "arm": arm, "episodes": episodes,
                          "n_envs": n_envs, "record": False, "od": f"ablation/{arm}", **spec})
    return cells


def _cells_multiseed(episodes=10, n_envs=10, seeds=(123, 456),
                     suites=("object", "goal", "long")):
    """TOP-PRIORITY deliverable: multi-seed robustness across the LIBERO suites.
    Runs the EXTRA seeds (123, 456) into <od>/<suite>/seed<N>/; combined with the primary
    seed-42 (Object/Goal grid already on disk; Long from the `long` stage) -> a 3-seed
    mean ± CI table. Default = 2 seeds × {Object, Goal, Long} × 6 axes × {baseline, aegis}
    = 72 cells. Each seed isolated & resume-safe.
    Spatial is the in-distribution/ablation suite (n=200 protocol) — add it explicitly if
    a multi-seed in-dist table is also wanted."""
    od_for = {"object": "liberov_objgoal", "goal": "liberov_objgoal", "long": "liberov_long",
              "spatial": "liberov_spatial"}
    cells = []
    for seed in seeds:
        for sk in suites:
            for axis in AXES:
                for arm in ARMS:
                    cells.append({"suite": sk, "axis": axis, "arm": arm, "episodes": episodes,
                                  "n_envs": n_envs, "record": False, "seed": seed,
                                  "od": od_for[sk]})
    return cells


def _cells_clean(episodes=10, n_envs=10, seeds=(42, 123, 456),
                 suites=("spatial", "object", "goal", "long")):
    """Clean (no-perturbation) per-suite SR, multi-seed — the 'we preserve clean success'
    table. ALL 4 suites × 3 seeds × {baseline, aegis} = 24 cells, eval-only. Uses the
    `clean` condition; isolated under <suite>/seed<N>/. Currently only seed 42 exists."""
    return [{"suite": sk, "axis": "clean", "arm": arm, "episodes": episodes, "n_envs": n_envs,
             "record": False, "seed": seed, "od": "clean_sr"}
            for seed in seeds for sk in suites for arm in ARMS]


@app.function(image=image, cpu=2.0, memory=8192, timeout=1200, volumes={"/assets": assets})
def validate():
    """CPU: confirm sim stack imports + EGL offscreen render works (no GPU)."""
    out = {}
    import torch; out["torch"] = torch.__version__
    import lerobot, robosuite, mujoco, transformers
    out.update(lerobot=lerobot.__version__, robosuite=robosuite.__version__,
               mujoco=mujoco.__version__, transformers=transformers.__version__)
    from libero.libero import benchmark, get_libero_path
    out["bddl_path"] = get_libero_path("bddl_files")
    # one offscreen EGL env step to prove rendering works headless
    from libero.libero.envs import OffScreenRenderEnv
    import os
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.reset(); obs, *_ = env.step([0.0]*7); env.close()
    out["egl_render_ok"] = True
    out["assets_listing"] = __import__("os").listdir("/assets")
    return out


# cpu=8: SyncVectorEnv steps the LIBERO MuJoCo envs sequentially on CPU and EGL-renders
# each — the physics+render is the bottleneck, NOT the GPU. So use the CHEAP L4 (~same
# speed as A10G here, ~30% cheaper). max_containers=6 bounds in-flight (unsaved) cells to
# 6 -> spend accrues gradually & observably; a cap/stop loses <=6 cells, not the whole run.
# timeout=10800 (3h) headroom (cells ~2h).
@app.function(image=image, gpu="L4", cpu=8.0, memory=32768, timeout=10800,
              max_containers=6, volumes={"/assets": assets})
def eval_cell(cell: dict):
    """One robustness cell: runs eval_libero_v.py for (suite, axis, arm). Streams progress
    live to the Modal log and returns SR. n_envs/tasks configurable (cheap smoke vs full)."""
    import os, subprocess, glob, sys, threading
    sk, axis, arm, ep = cell["suite"], cell["axis"], cell["arm"], cell["episodes"]
    n_envs = cell.get("n_envs", 20)                 # episodes/task == one batch == n_envs
    tasks = cell.get("tasks", "0,1,2,3,4,5,6,7,8,9")
    record = cell.get("record", True)               # save 1 sim video/task by default
    expected = n_envs * len([t for t in tasks.split(",") if t != ""])
    A = "/assets/sib_vla"
    # `method` = the eval --method, which is also the output SUBDIR eval writes to
    # (results land at <od>/libero_v/<method>/eval_<axis>.json). For the grid, method==arm
    # (baseline/aegis). For ABLATIONS, several arms share method=aegis (RIB-only / RASF-only
    # are aegis + a de-strength flag), so they MUST each carry a distinct `od` to avoid
    # clobbering each other and the real aegis grid — and resume/read must key on `method`.
    method = cell.get("method", arm)
    te = cell.get("te", True)                       # vanilla turns TE off (true floor)
    seed = cell.get("seed")                          # multi-seed: eval writes under seed<N>/
    seed_sub = f"/seed{seed}" if seed is not None else ""
    od = f"/assets/results_modal/{cell.get('od', 'liberov_objgoal')}/{sk}"
    os.makedirs(od, exist_ok=True)
    # RESUME: skip ONLY if a COMPLETE json exists (n>=expected and not partial). Partials
    # (from a killed run) are re-run, never silently treated as done.
    done = glob.glob(f"{od}{seed_sub}/libero_v/{method}/eval_{axis}.json")
    if done:
        d = json.load(open(done[0]))
        if d.get("n_episodes", 0) >= expected and not d.get("partial", False):
            return {"cell": cell, "rc": 0, "sr": round(d["success_rate"]*100, 1),
                    "n": d["n_episodes"], "resumed": True}
    cfg = f"/tmp/_lv_{sk}_{axis}_{arm}.yaml"
    with open(cfg, "w") as f:
        # absolute inherit resolves to the volume's configs/base.yaml regardless of cfg dir.
        # eval_n_envs MUST equal episodes_per_task (one batch = n_envs distinct init states).
        f.write(f"inherit: {A}/configs/base.yaml\ncheckpoint: {A}/outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model\n"
                f"output_dir: {od}\nsuite: {SUITES[sk]}\nepisode_length: {ELEN[sk]}\nn_action_steps: 1\n"
                f"eval_n_envs: {n_envs}\nepisodes_per_task: {n_envs}\nrecord:\n  enabled: false\n")
    TE = ["--forge-ensemble", "--ensemble-coeff", "0.01"]
    cmd = ["/usr/local/bin/python", "-u", f"{A}/scripts/eval_libero_v.py", "--config", cfg]
    # method block: ABLATION cells pass an explicit `method_args` (with {A} placeholders for
    # the asset root); grid cells fall back to the proven baseline/aegis args.
    if "method_args" in cell:
        cmd += [a.replace("{A}", A) for a in cell["method_args"]]
    elif arm == "baseline":
        cmd += ["--method", "baseline"]
    else:  # aegis = gate OPEN (RIB + RASF)
        cmd += ["--method", "aegis",
                "--rib-weights", f"{A}/results/ib_on86/rib_on86.pt",
                "--rasf-weights", f"{A}/results/rasf_on86/rasf_on86.pt"]
    cmd += ["--n-action-steps", "1", "--episodes", str(ep), "--tasks", tasks, "--only", axis]
    if seed is not None:
        cmd += ["--seed", str(seed)]                 # multi-seed run -> output_dir/seed<N>/
    if te:
        cmd += TE                                    # both arms carry TE except vanilla (floor)
    if record:
        cmd += ["--record", "--videos-per-task", "1"]   # 1 sim video/task -> volume
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": f"{A}:/opt/LIBERO"}
    tag = f"{sk}/{axis}/{arm}"
    print(f"[cell START] {tag} n_envs={n_envs} tasks={tasks} ep={ep} record={record}", flush=True)
    # COMMIT DAEMON: persist volume (partial JSONs + videos) every 2 min so a kill can't
    # erase progress — "save every inch". Stops on completion; final commit at the end.
    stop_commit = threading.Event()
    def _committer():
        while not stop_commit.wait(120):
            try: assets.commit()
            except Exception as e: print(f"[commit] warn: {e}", flush=True)
    threading.Thread(target=_committer, daemon=True).start()
    # stream subprocess output live (no blind capture) + keep a tail for the err field
    p = subprocess.Popen(cmd, cwd=A, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        sys.stdout.write(f"[{tag}] {line}"); sys.stdout.flush()
        tail.append(line)
        if len(tail) > 80:
            tail.pop(0)
    p.wait()
    stop_commit.set()
    res = {"cell": cell, "rc": p.returncode}
    j = glob.glob(f"{od}{seed_sub}/libero_v/{method}/eval_{axis}.json")
    if j:
        d = json.load(open(j[0])); res["sr"] = round(d["success_rate"]*100, 1); res["n"] = d["n_episodes"]
    else:
        res["err"] = "".join(tail)[-1500:]
    print(f"[cell DONE] {tag} rc={res['rc']} sr={res.get('sr','ERR')}", flush=True)
    assets.commit()
    return res


@app.local_entrypoint()
def main(stage: str = "validate", episodes: int = 20):
    if stage == "validate":
        print("VALIDATE:", validate.remote())
    elif stage == "smoke":
        # TRUE minimal smoke: 1 task, 4 envs (== 4 episodes), one perturbation. ~minutes.
        # Streams progress so per-task timing is visible (sizes the full sweep).
        print("SMOKE:", eval_cell.remote({"suite": "object", "axis": "gaussian_noise_1",
                                          "arm": "aegis", "episodes": 4, "n_envs": 4,
                                          "tasks": "0"}))
    elif stage == "stage1":
        cells = _cells_stage1(episodes)             # 24 cells, n_envs=10 (n=100), NO record
        print(f"launching {len(cells)} cells in parallel...")
        results = list(eval_cell.map(cells))
        for r in sorted(results, key=lambda x: (x['cell']['suite'], x['cell']['axis'], x['cell']['arm'])):
            c = r["cell"]; print(f"  {c['suite']:7} {c['axis']:18} {c['arm']:8} -> SR={r.get('sr','ERR')} (rc={r['rc']})")
    elif stage == "videos":
        cells = _cells_video()                      # curated demo footage only (~$2)
        print(f"launching {len(cells)} VIDEO cells...")
        for r in eval_cell.map(cells):
            c = r["cell"]; print(f"  VID {c['axis']:18} {c['arm']:8} -> SR={r.get('sr','ERR')} (rc={r['rc']})")
    elif stage == "long":
        cells = _cells_long(episodes)               # LIBERO-Long robustness: 6 axes × 2 arms = 12
        print(f"launching {len(cells)} LIBERO-Long cells (libero_10, 520-step)...")
        results = list(eval_cell.map(cells))
        for r in sorted(results, key=lambda x: (x['cell']['axis'], x['cell']['arm'])):
            c = r["cell"]; print(f"  long {c['axis']:18} {c['arm']:8} -> SR={r.get('sr','ERR')} (rc={r['rc']})")
    elif stage == "ablation":
        # Spatial ablation table: 8 arms × 6 axes = 48 cells, eval-only (all ckpts exist).
        cells = _cells_ablation(episodes)
        print(f"launching {len(cells)} ABLATION cells (Spatial, eval-only)...")
        results = list(eval_cell.map(cells))
        order = ["vanilla", "baseline", "ribonly", "rasfonly", "aegis", "raw_vib", "no_rate", "naive_ib"]
        by = {}
        for r in results:
            c = r["cell"]; by.setdefault(c["arm"], {})[c["axis"]] = r.get("sr", "ERR")
        hdr = "arm".ljust(10) + "".join(a.replace("_1", "")[:11].rjust(12) for a in AXES)
        print("\n==== ABLATION (Spatial) SR% ====\n" + hdr)
        for arm in order:
            row = by.get(arm, {})
            print(arm.ljust(10) + "".join(str(row.get(ax, "--")).rjust(12) for ax in AXES))
    elif stage == "clean":
        # clean per-suite SR, multi-seed: 4 suites × 3 seeds × 2 arms = 24 cells
        cells = _cells_clean(episodes)
        print(f"launching {len(cells)} CLEAN-SR cells (4 suites × 3 seeds)...")
        results = list(eval_cell.map(cells))
        for r in sorted(results, key=lambda x: (x['cell']['suite'], x['cell']['seed'], x['cell']['arm'])):
            c = r["cell"]; print(f"  {c['suite']:7} s{c['seed']} {c['arm']:8} -> SR={r.get('sr','ERR')}")
    elif stage == "multiseed":
        # TOP PRIORITY: extra seeds 123,456 × {object,goal,long} × 6 axes × 2 arms = 72 cells
        cells = _cells_multiseed(episodes)
        print(f"launching {len(cells)} MULTI-SEED cells (LIBERO suites: object/goal/long)...")
        results = list(eval_cell.map(cells))
        for r in sorted(results, key=lambda x: (x['cell']['seed'], x['cell']['suite'], x['cell']['axis'], x['cell']['arm'])):
            c = r["cell"]; print(f"  s{c['seed']} {c['suite']:7} {c['axis']:18} {c['arm']:8} -> SR={r.get('sr','ERR')}")
    else:
        print("stage = validate | smoke | stage1 | long | clean | videos | ablation | multiseed")
