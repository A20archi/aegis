# NanoVLA-S re-implementation plan (grounded in LeRobot ACT)

Decision (2026-06-13): re-implement NanoVLA-S as a **language-conditioned ACT**.
LeRobot's ACT already provides ~90% of NanoVLA-S's described architecture; the
real work is one architectural addition (BERT instruction token) + config + our
two plug-in points. GPU is serialized behind SmolVLA (~50h), so this is a
non-GPU code build now; training queues after.

## What LeRobot ACT already gives us (verified in `lerobot/policies/act/modeling_act.py`)
- ResNet backbone (`model.backbone`, ResNet18 via config) → image feature map. ✓ (NanoVLA vision)
- CVAE: `vae_encoder` (BERT-style over [cls, state, action_seq]) → `latent_dim*2` → z; β·KL in `forward()` (`config.use_vae`, `config.kl_weight`). ✓ (NanoVLA CVAE-KL)
- Transformer `encoder` + `decoder` w/ cross-attention (decoder action-query slots attend encoder memory = late fusion). ✓ (NanoVLA late fusion)
- Action chunk regression head → `A ∈ [B, chunk_size, action_dim]`; `predict_action_chunk()`; long-short via `chunk_size` (predict) vs `n_action_steps` (execute). ✓ (NanoVLA long-short chunking)
- `dim_model` default 512 = our **D**; `latent_dim`; `n_decoder_layers` ≈ ACT default → L_dec.

## The ONE real addition: frozen BERT-base instruction token
Vanilla ACT (Aloha) is NOT language-conditioned; NanoVLA-S adds a frozen BERT-base
language token into the encoder input sequence (LIBERO needs instruction following).
- Add `google-bert/bert-base-uncased` (frozen), pooled/CLS → `Linear(768 → dim_model)` → 1 language token.
- Prepend/concat that token to ACT's encoder input tokens (alongside state token + image tokens), with its own positional/type embedding.
- Dataset already carries the instruction (`task.language` in LIBERO); wire it through the processor → batch.

## Config (pin the paper's under-specified dims explicitly)
- `dim_model=512` (D), `n_encoder_layers=4` (paper), `n_decoder_layers=7` (ACT default), `latent_dim` (ACT default 32), `chunk_size=100` (H_train), `n_action_steps=10` (exec h), `action_dim=7` (LIBERO), `N_img≈49` (224² ResNet18), vision+BERT **frozen**, only ~52M enc/dec/heads train.
- Dataset: LIBERO via LeRobot (reuse `HuggingFaceVLA/libero`, image keys image/image2 — but NanoVLA-S used single 3rd-person; decide 1- vs 2-cam, paper real-robot used one 3rd-person).
- Eval: reuse our LIBERO eval harness (the same `eval_libero_v.py` / eval.py rollout path, n=50/suite to match paper).

## Plug-in attachment points (same plug-in, new host)
- **IB front:** wrap the image-token projection (ACT's `encoder_img_feat_input_proj`, the 1×1-conv analog `W_img`) — insert IB-Adapter on `X_img ∈ [B, N_img, D]` before the encoder. (Mirror `sib/ib_adapter.py::inject_fused_ib_adapter`, retargeted from SmolVLA connector to ACT image proj.)
- **SIB back:** apply SIB on the action chunk `A ∈ [B, H=100, d_act=7]` (deterministic regression head → one-shot, exactly like SmolVLA; no denoising loop). Reuse `sib/bottleneck.py` + `SIBPolicy` wrapper.
- **Caveat:** CVAE-KL on z + SIB on A both regularize the chunk → run a joint β-vs-SIB ablation (see NANOVLA.md §6.2).

## Build order (next turns)
1. `sib_vla/nanovla/` package: `modeling_nanovla.py` (ACT subclass + BERT token), `config_nanovla.py`.
2. Wire LIBERO dataset language → batch; baseline train script (`run_nanovla_baseline.sh`), reuse repro-style orchestration.
3. Validate baseline vs Table 1 (Spatial 81.6 / Object 93.6 / Goal 89.6 / Long 49.8 / Avg 78.7; accept Long/LIBERO-90 weak).
4. Port IB (image proj) + SIB (action chunk); ablate vs CVAE-KL.

## Risks (from NANOVLA.md §7)
No code/weights (full reimpl); under-specified hyperparams (D/β/LR/batch/L_dec); ~1M steps to converge (paper A.5); matching 78.7% avg is non-trivial. Compute is NOT the constraint (trained on one H20 / RTX 3060).
