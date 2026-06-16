# PROJECT STATE HANDOFF — read this first on the new machine

_Last updated: 2026-06-16. Purpose: a self-contained snapshot so work continues even if the Claude session transcript doesn't auto-resume._

## What this project is
Robustness + generality for sub-/multi-scale VLAs on LIBERO. Internal module = **AEGIS** =
**RIB** (Robust Information Bottleneck @ visual connector) + **RASF** (Residual Adaptive
Spectral Filter @ action chunk) + **temporal ensembling (TE)**. External label is always
**"<model>+SIB"** — do NOT mention ForgeVLA / forge / L1–L5 levers / TE-as-forge externally.

## The deliverable (due 2026-06-30) — 3 models, each: paper-beating SR + LIBERO-V robustness
"Size-agnostic 0.5B→3B" generality story:
1. **SmolVLA** (~500M, LLM+flow-matching) — the lead model. Friday 6/19 deck.
2. **NanoVLA-S** (161M, ACT+CVAE) — cross-arch #1. Injection BUILT.
3. **GR00T N1.5** (3B, Eagle-2.5 VLM + diffusion DiT head) — cross-arch #2 / scale-up.

Honest framing: neither NanoVLA (paper withdrawn) nor GR00T N1.5 (no official LIBERO) has a
clean published number. Claim = "AEGIS beats each model's OWN controlled baseline" + clears
best citable third-party (GR00T N1.5 LIBERO-Spatial ≈ **92%**, third-party reproductions).

## Current state (2026-06-16)
- **SmolVLA base retrain** (batch 384, 30k steps, full LIBERO) running on the A100, pid 2128940.
  Step ~15k, loss ~0.062 (converged/plateaued). Checkpoints every 4k in
  `sib_vla/outputs/smolvla_spatial_v2/checkpoints/` (4k/8k/12k present; → 30k ~Wed 06:00).
  Old base = 86% SR; new base loss now slightly below old → forecast AEGIS clean **~88 modal,
  P(≥89)~35-40%, P(≥92)~5%**. Robustness is the real wedge (viewpoint +20pp measured).
- **Overfit guard:** keep ALL 8 checkpoints, eval rollout SR across **12k→30k**, pick the SR
  PEAK (NOT necessarily 30k). This is the only valid early-stop for behavior cloning.
- **NanoVLA AEGIS injection:** BUILT + CPU-smoke-verified → `nanovla/aegis_nanovla.py`
  (RIB@`encoder_img_feat_input_proj` 0.66M, RASF@action chunk; identity-at-init exact).
  Smoke: `CUDA_VISIBLE_DEVICES="" python scripts/smoke_aegis_nanovla.py`.
- **GR00T N1.5:** env + checkpoint READY. Env `miniconda3/envs/gr00t` (torch 2.5.1+cu124,
  flash-attn 2.7.1, transformers 4.51.3, gr00t 1.1.0). Repo `/home/user/vla/Isaac-GR00T`
  (on `main`=N1.7; `git checkout n1.5-release` for N1.5). Checkpoint `nvidia/GR00T-N1.5-3B`
  cached (5.1G). Injection (`aegis_groot.py`) NOT yet written: RIB@Eagle backbone visual
  output, RASF@H=16 DiT action chunk. Mirror `nanovla/aegis_nanovla.py`.

## Pending task queue (PAUSED for GPU migration — resume on A6000)
1. **(SmolVLA finish)** eval ckpts 12k–30k → SR peak → module retrain (RIB 12k + RASF 8k via
   `run_v2_pipeline.sh modules <peak>`) → AEGIS clean n=200 (`... aegis <peak>`) → Thu
   robustness (`run_libero_v_headline.sh`) + noise sweep (`run_noise_sweep.sh`) → deck.
2. **Ablation (KEPT):** official-ckpt + TE → isolate retrain vs ensemble contribution to 86%.
3. **NanoVLA:** baseline train (`scripts/finetune_nanovla.py`) → RIB+RASF train → clean+robustness eval.
4. **GR00T N1.5:** write `aegis_groot.py` → baseline finetune (Isaac-GR00T `gr00t_finetune.py`,
   `--no-tune_diffusion_model` if 48GB tight) → module train → clean+robustness eval. Beat ~92% Spatial.
5. **Jun 30:** 3-model generality table + LIBERO-V robustness, all labeled "+SIB".

DROPPED (user, 6/16): EMA blending, TinyVLA, SIB+LoRA, extra LIBERO-Long n=1, 500-ep CI.

## Key paths
- Repo: `/home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi`  (sub: `sib_vla/`)
- SmolVLA env: **anaconda3 base** `/home/user/anaconda3/bin/python` (lerobot 0.4.3, torch 2.6.0+cu124)
- GR00T env: `/home/user/miniconda3/envs/gr00t/bin/python`
- GR00T repo: `/home/user/vla/Isaac-GR00T`
- Claude context: `/home/user/.claude/projects/-home-user-Desktop-SAPTARSHI-ALT-steer-information-Saptarshi/`
  (this session: `dfc04c18-7ecb-4315-b882-442ce699f2b4.jsonl`; durable knowledge in `memory/`)
- Configs: `sib_vla/configs/*.yaml` (HARDCODE checkpoint/output paths → keep paths identical!)

## A6000 (48GB) feasibility
All remaining work FITS 48GB: NanoVLA tiny; GR00T finetune is NVIDIA-tested on A6000; module
train (batch 32) + n=200 evals small. ONLY the batch-384 base retrain needed 80GB (done before
switch). Expect ~1.8–2.2x slower wall-clock than A100. See NEW_MACHINE_CHECKLIST.md.
