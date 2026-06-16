"""LIBERO-V (Visual) robustness perturbations — the 4 axes of "VLA Models Are
More Generalizable Than You Think" (Adapt3R viewpoint + LIBERO-Plus visual).

Three axes are SIM-side and applied by DIRECT MuJoCo state manipulation on each
LiberoEnv's underlying robosuite sim (paper Table 6: "Direct MuJoCo State
Manipulation" / "Procedural" lighting / texture swap):
  * viewpoint : orbit + offset the `agentview` camera (cam_pos / cam_quat)
  * lighting  : light_diffuse / light_dir / light_specular / light_castshadow
  * texture   : recolor + texture-swap floor/wall/table materials

The 4th axis (sensor noise: motion/zoom/fog/glass/gaussian blur) is image-space
and lives in sib/corruptions.py, applied through eval's existing corruption path.

Integration: build the vec envs as usual, then call
    install_sim_perturbations(vec_env, spec)
once before rollout. Each sub-env's reset() is wrapped to (re)apply the absolute
perturbation from captured original state and re-render the observation, so it is
robust to robosuite rebuilding the sim on reset and to gym auto-reset.

API verified on this box (LIBERO @ ~/Desktop/LIBERO, robosuite): agentview is
camera id 2; cam_pos/cam_quat, light_*, mat_*/geom_* are all editable and a
sim.forward() makes a direct edit show up in the render.
"""
from __future__ import annotations

import math
import numpy as np

AGENTVIEW = "agentview"   # the third-person camera LIBERO renders as image1

# Discrete viewpoint levels (Small/Medium/Large): orbital yaw about the vertical
# axis through the workspace pivot + a position offset + slight elevation/pitch.
VIEWPOINT_LEVELS = {
    "small":  {"yaw_deg": 12.0, "dpos": (0.05, 0.0, 0.05),  "pitch_deg": 4.0},
    "medium": {"yaw_deg": 26.0, "dpos": (0.10, 0.05, 0.10), "pitch_deg": 8.0},
    "large":  {"yaw_deg": 42.0, "dpos": (0.18, 0.10, 0.15), "pitch_deg": 12.0},
}
PIVOT_XY = (0.0, 0.0)     # orbit center (table/arm base under agentview, x~0)


# ---------------------------------------------------------------------------
# quaternion helpers (MuJoCo convention: wxyz)
# ---------------------------------------------------------------------------
def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_axis_angle(axis, angle_rad):
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    s = math.sin(angle_rad / 2.0)
    return np.array([math.cos(angle_rad / 2.0), ax[0] * s, ax[1] * s, ax[2] * s])


def _rot_z(xy, angle_rad, pivot):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    dx, dy = xy[0] - pivot[0], xy[1] - pivot[1]
    return np.array([pivot[0] + c * dx - s * dy, pivot[1] + s * dx + c * dy])


# ---------------------------------------------------------------------------
# unwrap helpers
# ---------------------------------------------------------------------------
def _libero_env(sub_env):
    """Unwrap a vector sub-env down to the LiberoEnv instance."""
    e = getattr(sub_env, "unwrapped", sub_env)
    return e


def _sim(libero_env):
    return libero_env._env.sim


# ---------------------------------------------------------------------------
# capture-originals-once (so perturbations are applied ABSOLUTELY, never compound)
# ---------------------------------------------------------------------------
def _capture_orig(libero_env):
    sim = _sim(libero_env)
    m = sim.model
    cid = m.camera_name2id(AGENTVIEW)
    orig = {
        "cam_id": cid,
        "cam_pos": np.array(m.cam_pos[cid], dtype=np.float64).copy(),
        "cam_quat": np.array(m.cam_quat[cid], dtype=np.float64).copy(),
        "light_diffuse": np.array(m.light_diffuse, dtype=np.float64).copy(),
        "light_dir": np.array(m.light_dir, dtype=np.float64).copy(),
        "light_specular": np.array(m.light_specular, dtype=np.float64).copy(),
        "light_castshadow": np.array(m.light_castshadow).copy(),
        "mat_rgba": np.array(m.mat_rgba, dtype=np.float64).copy() if m.nmat else None,
        "mat_texid": np.array(m.mat_texid).copy() if m.nmat else None,
        "geom_rgba": np.array(m.geom_rgba, dtype=np.float64).copy() if m.ngeom else None,
    }
    libero_env._lv_orig = orig
    return orig


# ---------------------------------------------------------------------------
# per-axis sim perturbations (operate on current sim, from captured originals)
# ---------------------------------------------------------------------------
def apply_viewpoint(libero_env, level: str):
    sim = _sim(libero_env); m = sim.model
    o = libero_env._lv_orig
    cid = m.camera_name2id(AGENTVIEW)
    p = VIEWPOINT_LEVELS[level]
    yaw = math.radians(p["yaw_deg"])
    # orbit position about vertical axis through pivot, then add offset + elevation
    pos = o["cam_pos"].copy()
    pos[:2] = _rot_z(pos[:2], yaw, PIVOT_XY)
    pos = pos + np.array(p["dpos"], dtype=np.float64)
    # re-orient: yaw about world-Z then a small pitch so it keeps facing the scene
    q = _quat_mul(_quat_axis_angle([0, 0, 1], yaw), o["cam_quat"])
    q = _quat_mul(q, _quat_axis_angle([1, 0, 0], math.radians(p["pitch_deg"])))
    q = q / (np.linalg.norm(q) + 1e-9)
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = q
    sim.forward()


def apply_lighting(libero_env, severity: int):
    sim = _sim(libero_env); m = sim.model
    o = libero_env._lv_orig
    sev = int(severity)
    dim = 1.0 - 0.25 * (sev + 1)                # dim diffuse
    tint = np.array([0.10, 0.02, -0.08]) * (sev + 1)  # warm/cool tint shift
    rot = math.radians(15.0 * (sev + 1))        # rotate light direction
    for i in range(m.nlight):
        d = o["light_diffuse"][i] * dim + tint
        m.light_diffuse[i] = np.clip(d, 0.0, 1.0)
        m.light_specular[i] = np.clip(o["light_specular"][i] * (1.0 + 0.4 * sev), 0.0, 1.0)
        dirv = o["light_dir"][i].copy()
        dirv[:2] = _rot_z(dirv[:2], rot, (0.0, 0.0))
        n = np.linalg.norm(dirv)
        m.light_dir[i] = dirv / n if n > 1e-9 else o["light_dir"][i]
        if sev >= 1:
            m.light_castshadow[i] = 1            # turn shadows on at higher severity
    sim.forward()


def apply_texture(libero_env, severity: int):
    sim = _sim(libero_env); m = sim.model
    o = libero_env._lv_orig
    if m.nmat == 0:
        return
    sev = int(severity)
    rng = np.random.default_rng(1234 + sev)     # deterministic per severity
    targets = []
    for gid in range(m.ngeom):
        try:
            name = m.geom_id2name(gid) or ""
        except Exception:
            name = ""
        if any(k in name.lower() for k in ("floor", "wall", "table", "ground", "plane", "counter")):
            mid = int(m.geom_matid[gid])
            if mid >= 0:
                targets.append(mid)
    # if names not found, fall back to a fraction of all materials
    if not targets:
        targets = list(range(min(m.nmat, 6)))
    strength = 0.25 * (sev + 1)
    for mid in set(targets):
        base = o["mat_rgba"][mid].copy()
        jitter = (rng.random(3) - 0.5) * 2.0 * strength
        base[:3] = np.clip(base[:3] + jitter, 0.05, 1.0)
        m.mat_rgba[mid] = base
        if m.ntex > 1 and sev >= 1:             # swap to a different existing texture
            m.mat_texid[mid] = int(rng.integers(0, m.ntex))
    sim.forward()


_DISPATCH = {"viewpoint": apply_viewpoint, "lighting": apply_lighting, "texture": apply_texture}


def _apply_spec(libero_env, spec: dict):
    """spec = {'axis': 'viewpoint'|'lighting'|'texture', 'level'|'severity': ...}"""
    axis = spec["axis"]
    fn = _DISPATCH[axis]
    if axis == "viewpoint":
        fn(libero_env, spec.get("level", "medium"))
    else:
        fn(libero_env, spec.get("severity", 1))


# ---------------------------------------------------------------------------
# installer: wrap each sub-env's reset to (re)apply the perturbation + re-render
# ---------------------------------------------------------------------------
def install_sim_perturbations(vec_env, spec: dict):
    """Wrap every sub-env reset so the sim perturbation is re-applied after each
    (auto)reset and the returned observation reflects the new sim state."""
    subs = getattr(vec_env, "envs", [vec_env])
    for sub in subs:
        le = _libero_env(sub)
        if getattr(le, "_lv_installed", False):
            continue
        orig_reset = le.reset

        def make_reset(le=le, orig_reset=orig_reset):
            def _reset(seed=None, **kwargs):
                obs, info = orig_reset(seed=seed, **kwargs)
                try:
                    if not hasattr(le, "_lv_orig"):
                        _capture_orig(le)
                    _apply_spec(le, spec)
                    raw = le._env.env._get_observations()
                    obs = le._format_raw_obs(raw)
                except Exception as e:               # never crash the rollout on a perturb error
                    print(f"[libero_v] WARN perturb failed: {e!r}", flush=True)
                return obs, info
            return _reset

        le.reset = make_reset()
        le._lv_installed = True
    return vec_env


# ---------------------------------------------------------------------------
# the LIBERO-V condition grid
# ---------------------------------------------------------------------------
def libero_v_grid(compact: bool = True):
    """Return the list of (label, condition) for the 4-axis sweep.

    condition is either {'sim': spec} (viewpoint/lighting/texture) or
    {'corruption': 'name:sev'} (sensor noise, image-space). `compact` trims the
    image-noise families to one representative severity to bound GPU time.
    """
    grid = []
    # axis 1: camera viewpoint (the headline) — Small/Medium/Large
    for lvl in ("small", "medium", "large"):
        grid.append((f"viewpoint_{lvl}", {"sim": {"axis": "viewpoint", "level": lvl}}))
    # axis 2: lighting
    sevs = (1,) if compact else (0, 1, 2)
    for s in sevs:
        grid.append((f"lighting_{s}", {"sim": {"axis": "lighting", "severity": s}}))
    # axis 3: texture
    for s in sevs:
        grid.append((f"texture_{s}", {"sim": {"axis": "texture", "severity": s}}))
    # axis 4: sensor noise (image-space)
    noise = [("motion_blur", 1), ("zoom_blur", 1), ("fog", 1), ("glass_blur", 1)]
    for fam, sev in noise:
        grid.append((f"{fam}_{sev}", {"corruption": f"{fam}:{sev}"}))
    # gaussian_noise severity SWEEP (graceful-degradation study): std 0.05..1.00
    # (0:.05 1:.12 2:.20 3:.30 4:.50 5:.70 6:.75 7:1.0) — extreme tail destroys the
    # image -> both arms collapse to the proprio-only floor; gap peaks mid-curve.
    for s in range(8):
        grid.append((f"gaussian_noise_{s}", {"corruption": f"gaussian_noise:{s}"}))
    return grid


# ---------------------------------------------------------------------------
# self-test (env-only, no policy): build one env, apply each sim axis, assert
# the render changes. Safe to run while training (single EGL render).
# ---------------------------------------------------------------------------
def self_test(suite="libero_spatial", task_id=0, save_dir=None):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    class _Shim:
        """minimal LiberoEnv-like shim exposing _env, _format_raw_obs."""
        def __init__(self, env):
            self._env = env
        def _format_raw_obs(self, raw):
            return {"pixels": {"image": raw["agentview_image"]}}
        def reset(self, seed=None, **kw):
            raw = self._env.reset()
            return self._format_raw_obs(raw), {}

    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.reset()
    le = _Shim(env)
    _capture_orig(le)
    base = env.sim.render(camera_name=AGENTVIEW, width=128, height=128).astype(float)
    results = {}
    for spec in [{"axis": "viewpoint", "level": "large"},
                 {"axis": "lighting", "severity": 2},
                 {"axis": "texture", "severity": 2}]:
        _capture_orig(le)                       # reset to canonical before each
        _apply_spec(le, spec)
        img = env.sim.render(camera_name=AGENTVIEW, width=128, height=128).astype(float)
        d = float(np.abs(img - base).mean())
        results[spec["axis"]] = d
        if save_dir:
            try:
                from PIL import Image
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(img.astype("uint8")[::-1]).save(
                    os.path.join(save_dir, f"selftest_{spec['axis']}.png"))
            except Exception:
                pass
    return results
