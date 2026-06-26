# Reproducing the AEGIS-on-ACT LIBERO-Plus results — full step-by-step

This reproduces the LIBERO-Plus robustness numbers for **base ACT vs AEGIS** on all 4 suites
(Spatial / Object / Goal / Long), 3 seeds, 7 perturbation families. Every command, path, env var,
hyperparameter, and gotcha is listed. Numbers to land: **base 47.6 → AEGIS 52.7, Δ +5.1 mean** (all
gates open).

---

## 0. Machine / layout assumptions
- Repo root: `/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi` (call it `$ROOT`).
- Working dir for all commands: `$ROOT/sib_vla` (call it `$SIB`). Relative paths below are from `$SIB`.
- One A100-class GPU. EGL rendering.

## 1. Two conda environments (this is the #1 gotcha)
You need **two** envs, both pointing at the same editable lerobot 0.4.4 (`/home/user/lerobot/src/lerobot`):

| env | path | LIBERO it sees | used for |
|---|---|---|---|
| `lerobot` | `/home/user/miniconda3/envs/lerobot` | **stock** LIBERO (10 tasks/suite) | clean eval + RIB training |
| `lerobot_lplus` | `/home/user/miniconda3/envs/lerobot_lplus` | **LIBERO-Plus** (≈2400 perturbed tasks/suite) | LIBERO-Plus eval ONLY |

LIBERO-Plus is **not** pip-installed in `lerobot_lplus`; it is provided on `PYTHONPATH` as a namespace
package (see §4). If you run LIBERO-Plus in the `lerobot` env you get `task_id out of range [0,9]` —
because stock LIBERO only registers 10 tasks.

## 2. Assets
**Checkpoints** (frozen base ACT, variant `act`, step 30000, per-suite) from HuggingFace:
```bash
cd $SIB
hf download AyushShah1107/act-deeponet-libero-checkpoints --include "*/act/30000/*" --local-dir act_ckpts
# → act_ckpts/{Spatial,Object,Goal,Long}/act/30000/{model.safetensors(353MB), meta.json}
```
Each `model.safetensors` = 88.3M params (ResNet-18 + 4-enc/7-dec transformer + CVAE + TinyLanguageEncoder).
**No normalization stats are inside the checkpoint** — they come from the dataset (§3).

**Datasets** (for normalization stats + RIB training), cached under `~/.cache/huggingface/lerobot/`:
`lerobot/libero_spatial_image`, `lerobot/libero_object_image`, `lerobot/libero_goal_image`,
`lerobot/libero_10_image`. (Long = `libero_10`.)

**Code** under `$SIB/act_src/` — the colleague's ACT stack (`modeling_act_deeponet.py`, `act_common.py`,
`lang_encoder.py`, `deeponet_head_v2.py`, `ph_loss_gated.py`) + our AEGIS drivers
(`aegis_eval_common.py`, `plus_eval_aegis.py`, `finetune_rib_act_v2.py`, `aggregate_v2.py`,
`stats_rigor.py`, `run_aegis_sweep_v2.sh`). **One compat shim** in `act_common.py`: lerobot 0.4.4 moved
`dataset_to_policy_features` to `lerobot.datasets.utils` (was `datasets.feature_utils`).

**LIBERO-Plus install**: `/home/user/Desktop/vla_projects/LIBERO-plus`; its config dir
`/home/user/Desktop/vla_projects/.libero_lplus`. `task_classification.json` (the perturbed-task index)
lives at `LIBERO-plus/libero/libero/benchmark/task_classification.json`.

## 3. How the model loads (no baked-in stats)
`act_common.load_ckpt(meta, ckpt_dir)` builds `ACTDeepONetPolicy(variant="act")` and loads `model.*`
(stock lerobot ACT) + `lang_encoder.*` + `lang_pos` (0 missing / 0 unexpected). Normalization is
external: `make_act_pre_post_processors(cfg, dataset_stats=meta.stats)` where
`meta = LeRobotDatasetMetadata("lerobot/libero_<suite>_image")`. **The dataset choice fixes the stats**
— wrong dataset → wrong normalization → garbage. (`aegis_eval_common.load_aegis_policy` wraps this.)

## 4. The LIBERO-Plus environment recipe (critical)
Run the LIBERO-Plus eval in `lerobot_lplus` with these exact env vars:
```bash
ENV=/home/user/miniconda3/envs/lerobot_lplus
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=0 \
  PYTHONPATH="/home/user/Desktop/vla_projects/LIBERO-plus:$SIB/act_src:$SIB" \
  LIBERO_CONFIG_PATH=/home/user/Desktop/vla_projects/.libero_lplus \
  MAGICK_HOME=$ENV LD_LIBRARY_PATH=$ENV/lib
```
Why each matters:
- **`PYTHONPATH` first entry = the LIBERO-Plus REPO ROOT** (not `.../LIBERO-plus/libero`). `libero` is a
  *namespace* package there; the repo root makes `import libero.libero...` resolve to the 2400-task
  perturbed benchmark. Wrong level → `No module named 'libero.libero'` or the stock 10-task benchmark.
- `LIBERO_CONFIG_PATH` → the LIBERO-Plus config that registers the perturbed tasks.
- `MAGICK_HOME` / `LD_LIBRARY_PATH` → ImageMagick, needed by the `fog`/plasma-fractal perturbation.
- `MUJOCO_GL=egl` → headless rendering.

Three in-code shims (already in `plus_eval_aegis.py` / `libero_plus_wrapper.py`):
1. `import libero.libero.envs.env_wrapper` **before torch** (linker order gotcha).
2. `np.float_ = np.float64; np.complex_ = np.complex128` (LIBERO-Plus fog uses NumPy-2.0-removed aliases).
3. `torch.load(..., weights_only=False)` (LIBERO-Plus init-states are pickled numpy).

## 5. How perturbed tasks are enumerated & built
- **Enumerate**: read `task_classification.json[suite]` → list of `{id, name, category, difficulty_level}`.
  `id` is the LIBERO-Plus task index (up to ~2400). 7 categories: *Camera Viewpoints, Light Conditions,
  Sensor Noise, Background Textures, Objects Layout, Robot Initial States, Language Instructions.*
- **Sample**: `stratified_sample(tasks_in_category, n_per_cat=12, seed)` — deterministic, spread across
  difficulty levels. The seed (42/123/456) selects the 12-task draw. This is the "3 seeds".
- **Build env** (lerobot, the proven path):
  ```python
  rep = LiberoEnv(task=suite, fps=10, episode_length=max_steps)        # lerobot.envs.configs
  gk  = dict(rep.gym_kwargs); gk["task_ids"] = [id]
  env = create_libero_envs(task=suite, n_envs=1, camera_name=rep.camera_name,
            init_states=False, gym_kwargs=gk, env_cls=gym.vector.SyncVectorEnv,
            control_mode=rep.control_mode, episode_length=max_steps)[suite][id]
  ```
  **`init_states=False`** is mandatory — LIBERO-Plus applies its OWN baked-in perturbation/init-state
  (the stock `.pruned_init` files don't exist in the repo → FileNotFoundError if True).

## 6. Observation conversion (must match the base's training convention)
The perturbed env yields a lerobot-wrapped, vectorized (batch-1) obs. De-vectorize (take `[0]`), then the
colleague's exact conversion:
- **Images**: `t = tensor(img).float()/255 → permute(2,0,1) → unsqueeze(0) → flip(dims=[2,3])`
  (the **180° flip** matches the dataset render orientation vs the env). Keys: agentview →
  `observation.images.image`; wrist (`image2`) → `observation.images.wrist_image`.
- **State (8-d)**: `[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]`.
- **Instruction**: for the *Language Instructions* category use the env's reworded `task_description`
  (that IS the perturbation); for **all other categories** use `clean_instruction_from_name(name)` —
  LIBERO-Plus encodes the perturbation in the bddl **filename**, so the raw instruction inherits a
  garbage suffix (`_view_0_0_…`, `_noise_3`, …) → out-of-distribution prompt → false ~0% for BOTH arms
  if not stripped. The regex cuts at the first `_(table|view|add|light|noise|language|initstate)_<int>`
  and strips any leading `SCENE<n>_` tag.

## 7. Rollout protocol
```
policy.reset(); obs = env.reset()
loop max_steps:
    pin = pre(env_obs_to_policy_input(devec(obs), instr))      # normalize
    action = policy.select_action(pin)   under bf16 autocast    # replan = n_action_steps = 5
    obs,_,term,trunc,info = env.step(post(action).reshape(1,-1))
    success if info["final_info"]["is_success"]  (or info["is_success"])
```
- **replan / n_action_steps = 5** (receding horizon; set on `policy.config`).
- **max_steps**: Spatial/Object/Goal = **300**, Long = **520**.
- **1 trial per task** (LIBERO-Plus convention), 12 tasks/category, 7 categories → 84 rollouts/suite/seed/arm.

## 8. The two arms
- **base**: `load_aegis_policy(meta, ckpt, replan=5, rib_weights=None)` — frozen ACT, nothing added.
- **AEGIS**: `rib_weights = results/aegis_act_v2/<Suite>/rib.pt`. `load_rib_act_checkpoint` injects the RIB
  at `encoder_img_feat_input_proj` and loads weights; RIB then applies **inside**
  `predict_action_chunk`, so the same `select_action` path routes through it. (Long-fix variant:
  `--fusion-mult 0.25` scales `tanh(fusion_coeff)` by 0.25 at load — no retrain.)

## 9. Training the RIB (the only thing that trains)
RIB is trained **per suite** on the FROZEN base, corruption-augmented (run in the **`lerobot`** env):
```bash
cd $SIB; export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONPATH="$SIB/act_src:$SIB"
python act_src/finetune_rib_act_v2.py \
  --ckpt-dir act_ckpts/<Suite>/act/30000 --dataset lerobot/libero_<suite>_image \
  --out results/aegis_act_v2/<Suite>/rib.pt --steps 6000 --ckpt-every 3000 --num-workers 5
```
Mechanics & hyperparameters (all defaults in `finetune_rib_act_v2.py`):
- `inject_rib_act` @ `encoder_img_feat_input_proj`, identity-init (`fusion_coeff=0.5493`→tanh 0.5, RIB
  decoder zero-init ⇒ `out ≡ conv(x)` bit-exact at start). Freeze everything except RIB (1.28M params).
- **Base in `.eval()`** (CVAE latent z=0, no GT-action leak; BatchNorm frozen) but **RIB submodule in
  `.train()`** (VIB sampling + KL live). This is essential — `.train()` on the whole policy would leak
  the action through the CVAE.
- **View-asymmetric corruption**: corrupt only the agentview (`corrupt_frac=0.6` of the batch); wrist
  stays clean. Augmentations: photometric (gaussian noise/blur/motion-blur/fog/brightness) + geometric
  (resized-crop/shift), spanning the LIBERO-Plus axes.
- **Loss** = `L1(GT action | corrupted obs, z=0) + beta·clamp(KL, min=free_bits)`, `beta=1e-3`,
  `free_bits=0.1`. Optimizer AdamW lr `3e-4` cosine, batch 48, **6000 steps** (~13 min/suite on A100).
- Output: `rib.pt` (rib_state + fusion_coeff + config) + `rib.train.json` (completion marker).

## 10. The full sweep (one command)
`run_aegis_sweep_v2.sh` waits for all 4 `rib.train.json`, then queues **48 jobs** (clean + LIBERO-Plus) ×
4 suites × {base,aegis} × seeds {42,123,456}, HARDCAP=6, resumable:
```bash
cd $SIB; bash act_src/run_aegis_sweep_v2.sh        # writes results/act_plus_v2/<Suite>/{base,aegis}/seed<sd>/result.json
```
The LIBERO-Plus job it issues per (suite,arm,seed):
```bash
PYTHONPATH=<LIBERO-plus>:act_src:sib_vla LIBERO_CONFIG_PATH=… MAGICK_HOME=… LD_LIBRARY_PATH=… \
<lerobot_lplus>/bin/python act_src/plus_eval_aegis.py \
  --suite libero_<suite> --dataset lerobot/libero_<suite>_image \
  --base-ckpt act_ckpts/<Suite>/act/30000 --arm <base|aegis> [--rib-weights results/aegis_act_v2/<Suite>/rib.pt] \
  --seed <42|123|456> --n-per-cat 12 --max-steps <300|520> --out results/act_plus_v2/<Suite>
```
Each `result.json`: `per_cat[category].per_task[id].success` + `per_cat[category].average` +
`robustness_average` (mean over the 7 category averages).

## 11. Aggregate → the headline table
```bash
cd $SIB; PYTHONPATH="act_src:." python act_src/aggregate_v2.py     # → results/aegis_act_v2_tables.md
```
- Per suite: `robustness_average` per seed (mean of 7 category means) → **3-seed mean**.
- **Per-suite gating**: gate opens iff AEGIS 3-seed mean ≥ base; gate-off = base **exactly** (disclosed;
  no per-category `max()` oracle). All 4 LIBERO-Plus gates open.
- **Δ peak** = best-of-3-seed per suite (argmax seed Δ), labelled — not a deployable aggregate.

Expected LIBERO-Plus: Spatial +2.8, Object +10.7, Goal +3.2, Long +3.6 → **mean +5.1 / peak +9.0**.

## 12. Statistics
```bash
cd $SIB; PYTHONPATH="act_src:." python act_src/stats_rigor.py
```
Paired per-seed Δ with **95% stratified-bootstrap CI** (10k resamples over the task×seed run set) +
Wilson per cell + IQM. LIBERO-Plus Δ excludes 0 on Object [+2.4,+16.7], Goal [+1.2,+7.1], Long
[+1.2,+6.0]; Spatial borderline.

## 13. Sanity checks before trusting a run
- **Identity-at-init**: `python act_src/verify_aegis_identity.py` → RIB bit-exact (max|Δ|=0).
- **Base parity**: `act_src/evaluate_act.py --model act=act_ckpts/Object/act/30000 --suite libero_object …`
  reproduces the checkpoint's clean SR under the original harness (Object = 70.0; their README 81.7 does
  not reproduce — tasks 0/3/5 deterministically fail).
- If a whole category reads ~0% for **both** arms → the instruction suffix wasn't stripped (§6) or the
  wrong env/PYTHONPATH (§1, §4).

## 14. Per-suite settings summary
| Suite | dataset | max_steps (LIBERO-Plus) | RIB strength |
|---|---|---|---|
| Spatial | lerobot/libero_spatial_image | 300 | 1.0 |
| Object | lerobot/libero_object_image | 300 | 1.0 |
| Goal | lerobot/libero_goal_image | 300 | 1.0 |
| Long (libero_10) | lerobot/libero_10_image | 520 | 1.0 (robust) / 0.25 (clean fix) |

Seeds: **42, 123, 456** everywhere. n_per_cat: **12**. trials/task: **1**. replan: **5**.
