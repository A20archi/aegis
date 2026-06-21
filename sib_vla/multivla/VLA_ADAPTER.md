# VLA-Adapter — Reproduction + AEGIS Porting Brief

> Target #2 for the cross-architecture robustness story (after SmolVLA). Goal: **reproduce
> VLA-Adapter's LIBERO SR under THEIR protocol, tie-or-beat it by a thin margin with AEGIS
> (identity-residual ⇒ ≥ base by construction), then show robustness gains** on the same
> harness. Repo: https://github.com/OpenHelix-Team/VLA-Adapter (cloned →
> `/home/user/Desktop/vla_projects/VLA-Adapter`). All file:line refs are from that clone.

## 0. Why VLA-Adapter is the right next model
- **Latest + SOTA-at-size:** Sep 2025, LIBERO non-Pro **97.3 avg** (Spatial 97.8 / Object 99.2 / Goal 97.2 / Long 95.0); Pro **98.5 avg**. Near the clean ceiling ⇒ maximal *robustness* headroom.
- **Cleanest action head:** continuous **L1-regression chunk H=8, action_dim=7** → RASF attaches one-shot, exactly as on SmolVLA. No discrete tokens, no diffusion loop.
- **Size-matched** to SmolVLA (Qwen2.5-0.5B + DINOv2/SigLIP ≈ 1B class) → controlled "same recipe, different backbone, same scale" claim.
- **PyTorch, no JAX.** Own codebase (not LeRobot) but standard PyTorch + DINOSigLIP/Qwen.

## 1. Exact AEGIS hook points (confirmed in their code)

### RIB — perception locus → `PrismaticProjector`
- File: `prismatic/extern/hf/modeling_prismatic.py:242-273` (class `PrismaticProjector`), attr path `model.projector`, invoked in `_process_vision_features()` (`:472`).
- Dims: **vision 2176 (fused DINOSigLIP) → LLM 2048 (Qwen2.5-0.5B)**.
- Port: wrap `model.projector` with `FusedRobustIBProjector` exactly as on SmolVLA's connector, but instantiate RIB at **D_out=2048** (SmolVLA used 960). Identity-residual + positive fusion_coeff unchanged.

### RASF — action locus → the (8,7) chunk, pre-unnormalization
- The final continuous chunk is produced in `_regression_or_discrete_prediction()` at
  `modeling_prismatic.py:868-871` → `normalized_actions.reshape(NUM_ACTIONS_CHUNK=8, ACTION_DIM=7)`, then unnormalized at `:970`.
- **RASF injection point: between :871 and :970** — filter the `(8,7)` normalized chunk in place. Identical contract to SmolVLA (DCT-II over the H=8 time axis, bounded residual, gate). `H=8` (vs SmolVLA's 50) — fewer bands, but the mechanism is unchanged.

### TE — receding horizon (OPTIONAL — protocol tension, see §3)
- Their rollout executes the **whole 8-step chunk open-loop** then replans (`run_libero_eval.py:311` `deque(maxlen=num_open_loop_steps=8)`, fill→popleft→replan-when-empty). TE needs *overlapping* chunks (replan every step), which changes their protocol and ×8 inference. **Recommendation: port RIB+RASF under their native protocol; treat TE as a separate, optional study.**

## 2. Their exact eval protocol (we follow this)
- **n = 500 / suite = 50 trials × 10 tasks** (`num_trials_per_task=50`, `run_libero_eval.py:110`). (Note: SmolVLA story used n=200; here we match VLA-Adapter's 500.)
- **Replan every 8 steps** (`num_open_loop_steps=8`), execute 1 action/step.
- Data suites: `libero_{spatial,object,goal,10}_no_noops`.
- Command (per README `:503-542`):
  ```bash
  CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
    --use_pro_version True --use_proprio True --num_images_in_input 2 \
    --use_film False --use_minivlm True \
    --num_trials_per_task 50 --task_suite_name libero_spatial
  ```
- Flags: `use_pro_version` → `MLPResNetBlock_Pro` (RoPE+FiLM, 98.5 avg) vs standard (97.3); `use_proprio` adds 8-dim proprio; `num_images_in_input 2` = main+wrist; `use_minivlm` = Qwen prompt format.

## 3. Image-perturbation injection (for robustness)
- `prepare_observation()` at `run_libero_eval.py:246-265`: `img = get_libero_image(obs)` (`:249`), `wrist_img` (`:250`), resize at `:253-255`.
- **Inject our LIBERO-V corruption grid right after `:249/:250`** (before resize) — reuse `sib/corruptions.py` + `sib/libero_v.py` perturbation families (gaussian noise, motion blur, lighting, texture, viewpoint) so the axes match the SmolVLA story.

## 4. Environment (isolated conda env `vla-adapter`)
```bash
conda create -n vla-adapter python=3.10.16 -y && conda activate vla-adapter
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
cd /home/user/Desktop/vla_projects/VLA-Adapter && pip install -e .
pip install "flash-attn==2.5.5" --no-build-isolation
# LIBERO (reuse existing install if compatible; else):
pip install -e LIBERO && pip install -r experiments/robot/libero/libero_requirements.txt
```
- Pins: transformers 4.40.1, timm 0.9.10, tokenizers 0.19.1, tensorflow 2.15.0, `libero==0.1.0`, dlimp fork `git+https://github.com/moojink/dlimp_openvla`.
- Checkpoints: HF org `VLA-Adapter/LIBERO-{Spatial,Object,Goal,Long}` (+ `-Pro`) → place in `outputs/`. ~1B model; inference footprint ~3–5 GB (fits easily; far lighter than SmolVLA evals).
- Gotcha (README): `AttributeError: ... eglQueryString` → EGL fix in their README; `MUJOCO_GL=egl` as we already use.
- Local artifacts found: `/home/user/Desktop/eval/prashant_audit/Models/vla_adapter_{git,smol}.pth` (~57 MB — likely policy-adapter only, not full model; verify before relying on).

## 5. Phased plan
- **P0 — setup (non-GPU):** env + checkpoints + perturbation hook. Confirm `vla_adapter_*.pth` provenance.
- **P1 — reproduce baseline:** run their command, all 4 suites, n=500. Target non-Pro **97.3** (Pro 98.5 as stretch). Lock our reproduced numbers.
- **P2 — AEGIS clean (tie/beat thin margin):** inject RIB@projector (D_out=2048) + RASF@(8,7) chunk, both identity-residual ⇒ clean SR **≥ base by construction**; train RIB/RASF lightly on VLA-Adapter features (corruption-augmented, as on SmolVLA). Report AEGIS ≥ 97.3.
- **P3 — robustness:** perturbation grid (§3) × {baseline, AEGIS}, n=500 (or n=200 to match SmolVLA axes), all suites or Spatial-first. Show AEGIS retention gains, mirroring the SmolVLA robustness table.

## 6. Grounded effort/compute
- **Setup (P0):** ~½–1 day eng (env + flash-attn build + 4 checkpoint downloads + perturbation hook). Non-GPU mostly.
- **Reproduce (P1):** ~1 B model, n=500/suite. Inference is light (~3–5 GB); expect faster per-episode than SmolVLA. 4 suites ≈ a few GPU-hours.
- **AEGIS train (P2):** RIB/RASF light finetune on cached features — hours, like SmolVLA's RIB/RASF runs.
- **Robustness (P3):** 6 axes × {base, AEGIS} × suites, n=500 — the main GPU spend; scope Spatial-first then expand.
- **Risk notes:** Pro vs non-Pro decision (start non-Pro = easier tie/beat + the headline 97.3); TE protocol tension (§1, keep optional); their codebase ≠ LeRobot so the eval-harness perturbation hook is new (but localized to `prepare_observation`).
