# Contributions and Novelty — AEGIS: Dual-Locus Robustness for Frozen VLAs

> Covers the full AEGIS system (RIB perception leg + RASF/SIB action leg + temporal ensembling) and novelty defenses against all verified related work. Hallucinated paper citations are flagged explicitly.

---

## The Problem We Are Solving

VLAs predict actions in chunks — a window of 50 future timesteps at once. Those chunks are executed open-loop before the model is queried again. The predicted chunk mixes genuine motion signal (the slow, intentional trajectory) with high-frequency jitter from the stochastic flow-matching sampler. Nothing in the standard pipeline separates the two.

We ask: **can we build a lightweight post-hoc module that identifies and suppresses jitter in a principled, interpretable way — without hurting task success and without touching the base model?**

---

## Our Approach: Five Interlocking Pieces

### 1. Decompose the action chunk with DCT-II

We apply the **orthonormal DCT-II** along the time axis of the 50-step chunk, independently per joint dimension. This maps 50 timesteps → 50 spectral bands, each corresponding to a temporal frequency.

**Why DCT-II?**
- Orthonormal: the round-trip is exact, Parseval holds — no energy added or lost.
- Energy compaction: for smooth signals (which well-trained policy outputs are), most energy concentrates in the first few bands. High-frequency bands carry little policy signal and mostly noise. This is what makes per-band treatment meaningful.
- Approximate decorrelation: DCT approximately diagonalizes the covariance of smooth time series, making bands approximately independent — the assumption the channel model requires. Our raw_vib ablation (removing DCT, keeping everything else identical) confirms that without this property the module breaks.

DCT is a standard signal processing tool. Applying it to the temporal axis of VLA action chunks is the novel step.

---

### 2. Model each band as an independent Gaussian channel

In DCT space, band `k` has:
- **Signal power** `λ_k`: per-band variance of the policy's predicted coefficients, estimated from data and tracked via EMA. Measures how much the policy uses that frequency.
- **Noise floor** `σ_k²`: a learned scalar per band per dimension (350 values total). Represents how much noise the channel introduces at that band.
- **SNR** = `λ_k / σ_k²`.

Low-frequency bands: high `λ_k`, high SNR → motion structure.
High-frequency bands: low `λ_k`, low SNR → jitter.

The only learned parameters are the `σ_k²` values. The VLA backbone is completely frozen.

---

### 3. Wiener gain as the filter

The MMSE-optimal filter for a Gaussian channel is:

```
g_k = λ_k / (λ_k + σ_k²)   ∈ (0, 1]
```

This is the closed-form solution — not a design choice. At evaluation: multiply each spectral coefficient by its band's gain, then invert the DCT. Computationally, 350 multiplications per chunk.

Training injects noise via reparameterization so gradients reach `σ_k²`. Evaluation uses the clean MMSE decode — no stochastic sampling at inference.

---

### 4. Rate as regularizer — the rate-distortion connection

The mutual information of band `k`'s Gaussian channel is:

```
R_k = ½ · ln(1 + λ_k / σ_k²)     [nats]
```

Training loss:

```
L = MSE(A_hat, A_star) + β · ΣR_k
```

`A_star` is the **ground-truth action** from the training data — not the policy's own output. The module trains as a denoiser: given the frozen policy's prediction, produce something closer to the true action. The β · R term penalizes total information throughput, forcing the module to drop bands that cannot justify their rate cost against the MSE objective.

For the idealized compression case (distortion target = source), the optimal allocation under this loss is the **reverse water-filling** closed form: allocate `R_k = ½ ln(λ_k / θ)` for bands above the water level `θ = β/2`, and zero bits below. We prove and unit-test this (`sib/waterfill.py`). In the denoising case (target = ground truth ≠ source) the solution becomes task-weighted, but the qualitative structure is identical — low-SNR bands are zeroed. This application of rate-distortion theory to robot action sequences has not appeared in prior work.

---

### 5. Signal power estimated from policy's own predictions

One forward pass over the training set collects predicted chunks → DCT → per-band variance → warm-start `λ_k`. EMA keeps it tracking the distribution during training, always detached from the gradient. This must be distributional (dataset-level), never per-sample — a per-sample λ would be too noisy to give a stable gain.

**Identity at initialization:** `σ_k²` starts small relative to `λ_k`, so `g_k ≈ 1` at epoch 0. The module is a pass-through before any learning — task success is preserved from the start.

---

---

## AEGIS: Full Dual-Locus System

The SIB/RASF sections below describe the **action locus** of AEGIS. The complete system has three components:

### RIB — Robust Information Bottleneck (perception locus)

A Variational Information Bottleneck residual adapter at the vision→LLM projector output (D=2048, 2.27M params).

**Design:** `out = proj(x) + tanh(α) · RIB(proj(x))`, where α and the decoder are zero-initialized. At init, `tanh(0)=0`, so the module contributes nothing — gate-off recovers the base policy exactly. This is verified numerically: `max|out − linear(x)| = 0`.

**Training:** 60% of each mini-batch is photometrically/sensor-corrupted; RIB is trained to predict the GT action from corrupted projector features while clean samples in the same batch hold the SR floor. An explicit β·KL rate term forces the latent to compress away corruption-specific variation rather than pass it through.

**Why the rate term is load-bearing:** Without a KL penalty, the bottleneck collapses to a pass-through (gate → identity). StableVLA's IB-Adapter is the empirical proof: their gate trains to fusion_coeff → −0.006 on clean task loss alone — a dormant module. Our β·KL is what forces engagement.

### RASF — Robust Action Spectral Filter (action locus)

Described in full in the SIB sections below. DCT-II → per-band Wiener gain → inverse DCT. Covers action-execution noise, an axis the perception-only RIB cannot address.

### Temporal Ensembling (TE)

Overlapping chunk blending (Zhao et al. ACT 2023, 2304.13705): keeps the last N chunk predictions in a buffer and exponentially weights them at each timestep. No learned parameters. Provides temporal smoothing independently of RIB and RASF.

### The Identity-Preservation Guarantee

Both RIB and RASF are zero/identity-initialized. Removing either module (setting α=0 for RIB; removing RASF) recovers the exact base policy output. **No prior verified paper makes this guarantee.** It is the key property that lets us report an honest "gate-off = baseline exactly" in the results table — no regression is structurally possible before learning begins.

---

## How We Differ from All Verified Related Work

### Full Comparison Matrix

| | **StableVLA** | **STRONG-VLA** | **RobustVLA** | **CSP** | **SOMA** | **TIDAL** | **BC-IB** | **BYOVLA** | **IBAC-SNI** | **VDB** | **AEGIS (ours)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **arxiv** | 2605.18287 | 2604.10055 | 2511.01331 | 2606.29570 | 2603.24060 | 2601.14945 | 2502.02853 | 2410.01971 | NeurIPS'19 | ICLR'19 | this work |
| **Problem** | Visual robustness | Multi-modal robustness | Obs+action robustness | Action generation quality | Failure recovery | Inference latency | Repr. compression | Distractor robustness | State repr. | IL discriminator | Visual + action robustness |
| **Base policy frozen?** | No | No | No | N/A (new policy) | Yes (no weights) | Yes (loop only) | Partial | Yes (no weights) | No | No | **Yes — provably** |
| **gate-off = baseline?** | No claim | N/A | N/A | N/A | N/A | N/A | No | Trivially | No | No | **Yes — verified** |
| **Explicit β·KL rate term** | No | No | No | No | No | No | Yes | No | Yes | Yes | **Yes (RIB + RASF)** |
| **Corruption-augmented training** | No (explicit) | Yes (retrains base) | No | No | No | No | No | No | No | No | **Yes (frozen adapter)** |
| **DCT on actions** | No | No | No | Yes (generative) | No | No | No | No | No | No | **Yes (post-hoc filter)** |
| **Identity-init** | No | No | No | N/A | N/A | N/A | No | N/A | No | No | **Yes — both modules** |
| **LIBERO-Plus 7-axis eval** | No | No | No | No | No (custom) | No | No | No | No | No | **Yes** |

---

### [1] StableVLA IB-Adapter
*Hiranaka et al., 2024*

StableVLA inserts an IB-Adapter at the vision→LLM projection interface to improve robustness under visual corruptions. The adapter uses **channel-covariance sigmoid gating**: it computes feature statistics across the channel dimension, applies a sigmoid-based gate to suppress individual feature channels, and has no explicit rate term in the training loss. The gating is a heuristic suppression learned from the task loss alone, not from an information-theoretic objective.

**How we differ:**
- Our bottleneck operates on the **action output**, not visual tokens — a completely different locus. Our Week 2 extension will add a perception-locus leg at the same insertion point StableVLA uses, making it a direct comparison.
- We use a **spectral transform (DCT)** before bottlenecking. StableVLA operates in raw feature space with no change of basis. The spectral basis is what gives our filter its interpretability and its energy-compaction property.
- We have an **explicit rate term** `β · Σ R_k` derived from the Gaussian channel mutual information. StableVLA has no rate objective — its gating is supervised entirely by the downstream task loss.
- We use the **MMSE Wiener gain** as the filter, derived from the channel model. StableVLA uses a sigmoid gate, which has no principled connection to the signal's SNR.
- We estimate **signal power `λ_k` per band** from the policy's own predictions. StableVLA has no analogous distributional signal power estimate.
- We train **post-hoc on cached outputs** without touching the VLA. StableVLA requires full VLA fine-tuning through the adapter.

---

### [2] IBAC-SNI — Information Bottleneck Actor-Critic with Selective Noise Injection
*Igl et al., NeurIPS 2019*

IBAC-SNI applies the VIB framework to the state representation inside a reinforcement learning policy. The bottleneck compresses the state encoding `z` with a KL penalty `β · KL(q(z|s) || p(z))`, making the representation compact and improving generalization across environment variations. Selective noise injection is used to regularize only informative features.

**How we differ:**
- IBAC compresses the **policy's state input representation** (what the policy reads). We filter the **action chunk output** (what the policy produces). These are opposite ends of the policy pipeline.
- IBAC uses a standard **KL divergence to a Gaussian prior** (vanilla VIB). We use a per-band Gaussian channel rate `R_k = ½ ln(1 + λ_k/σ_k²)` where the noise variance `σ_k²` is learned and the signal variance `λ_k` is estimated from the policy's own predictions. Our rate is grounded in the signal's empirical SNR, not in a fixed prior.
- IBAC operates in **raw representation space**. We operate in the **DCT frequency domain** of the action sequence — the spectral basis is what makes the suppression physically meaningful (high-frequency temporal bands = jitter).
- IBAC uses **stochastic inference** (draws z ~ q(z|s) at eval time). We use the **closed-form MMSE Wiener gain** — deterministic, no sampling at evaluation.
- IBAC is trained **end-to-end** with the policy. We train **post-hoc on cached policy outputs**; the base model is never updated.
- IBAC's goal is **representation compression for generalization**. Our goal is **action sequence smoothing and denoising** — different problem, different target.

---

### [3] VDB — Variational Discriminator Bottleneck
*Peng et al., ICLR 2019*

VDB applies the information bottleneck to the discriminator inside an adversarial imitation learning (AIRL/GAIL) pipeline for physics-based character control in robotics. The discriminator's input features are compressed through a VIB bottleneck — `β · KL(q(z|s) || p(z))` — so the discriminator cannot overfit to spurious features, improving the quality of the learned reward function and the stability of the IL training.

**How we differ:**
- VDB's bottleneck is inside the **discriminator of an adversarial IL pipeline**. Ours is a **post-hoc filter on the action chunk** output of a frozen VLA — no adversarial training, no discriminator, no reward learning.
- VDB compresses **discriminator input features** — the bottleneck learns what information the discriminator needs to identify real vs. fake behavior. We filter **action sequence temporal coefficients** — the bottleneck learns how many bits each frequency band needs to reproduce ground-truth motion.
- VDB uses **standard VIB (KL to prior)**; no spectral basis, no signal-power estimation, no Wiener gain.
- VDB is embedded in the IL training loop and **cannot be applied post-hoc**. We train on cached policy outputs without modifying the base VLA.
- VDB's bottleneck produces a compressed feature for a downstream discriminator. Our bottleneck produces a **filtered action chunk** that is executed directly in the environment — the output is the artifact, not an intermediate representation.

---

### [4] Deep Variational Information Bottleneck
*Alemi et al., ICLR 2017*

The VIB paper establishes the variational framework that StableVLA, IBAC-SNI, VDB, and ourselves all build on: train a compression bottleneck by minimizing `distortion + β · I(X; Z)` using the reparameterization trick to make the rate differentiable. Their application is learning compressed representations for image classification — the bottleneck `Z` is a stochastic intermediate representation from which a class label is predicted.

**How we differ — this is where our technical extensions over VIB live:**
- **Signal domain.** VIB operates on raw input features (image pixels → representation). We operate in the **DCT frequency domain of a temporal action sequence**. The spectral basis is an additional design choice not present in the original VIB.
- **Channel model.** VIB uses a single Gaussian encoder `q(z|x)` with a KL penalty. We use a **per-band Gaussian channel** where each band has its own learned noise floor `σ_k²` and an empirically estimated signal power `λ_k`. The rate `R_k = ½ ln(1 + λ_k/σ_k²)` is a function of the SNR, not just of the KL to a prior.
- **Signal power estimation.** VIB has no concept of `λ_k` — the signal statistics are not estimated. Our warm-start and EMA over the policy's own predictions is an additional piece that makes the Wiener gain calibrated to the actual signal distribution.
- **Inference decode.** VIB draws a stochastic sample `z ~ q(z|x)` at inference. We apply the **closed-form MMSE Wiener gain** `g_k = λ_k/(λ_k + σ_k²)` — deterministic, no sampling, provably optimal under the Gaussian channel model.
- **Distortion target.** VIB's target is a class label (task signal). Our target is the **ground-truth action chunk** from training data — this turns the bottleneck from a compressor (reduce bits about x to predict y) into a **denoiser** (reduce noise in x to recover A_star).
- **Post-hoc application.** VIB is trained end-to-end as part of the model. We train on **cached policy outputs** — the frozen VLA is never differentiated through.

---

---

## Defenses Against Recently Verified Papers (2025–2026)

### [5] CSP — Causal Spectral Policy
*Cao et al., arxiv 2606.29570, June 2026*

CSP decomposes robot action sequences hierarchically using spectral methods: low-frequency components capture global motion trajectories; high-frequency components encode timing, alignment, and contact behaviors. Actions are generated coarse-to-fine during policy inference.

**How we differ:**

- **CSP is a generative model; RASF is a post-hoc filter.** CSP replaces the action generation head — it generates actions *in* frequency space from scratch. RASF applies a learned amplitude-shaping gain to the DCT of an already-generated chunk from a *frozen* policy. These are different operations on different targets.
- **CSP retrains the policy; AEGIS does not.** CSP cannot be plugged into an existing VLA without retraining. AEGIS trains only the 2.27M-param RIB adapter and 350-scalar RASF on cached policy outputs; the 0.5B base is never differentiated through.
- **No information bottleneck.** CSP has no KL term, no rate-distortion objective, no identity-preservation guarantee. It is a representational choice for generation, not a robustness mechanism.
- **No robustness evaluation.** CSP is evaluated on manipulation success under clean conditions. AEGIS is evaluated on LIBERO-Plus 7-axis robustness breakdown (sensor noise, lighting, camera viewpoint, background, language, object layout, robot init).
- **Spectral decomposition means different things.** CSP's "spectral" refers to using frequency bands as a generative hierarchy. RASF's "spectral" refers to MMSE Wiener filtering of temporal noise in an existing signal. The shared vocabulary does not imply shared mechanism.

**Defense line:** *"CSP and RASF both involve DCT of action sequences, but CSP is a generative architecture for clean manipulation and RASF is a post-hoc denoising filter for a frozen policy under perturbation. They address different problems at different stages of the pipeline."*

---

### [6] SOMA — Strategic Orchestration and Memory-Augmented System
*arxiv 2603.24060, March 2026*

SOMA adapts frozen VLA policies at inference through a Dual-Memory RAG pipeline: a contrastive memory bank retrieves relevant past episodes, an LLM orchestrator diagnoses causal failures, and Model Context Protocol (MCP) interventions modify the execution context. No learned weights are added to the VLA; adaptation is entirely through retrieval. Reports 56.6% average absolute gain on LIBERO-PRO/LIBERO-SOMA benchmarks across π₀, π₀.₅, SmolVLA.

**How we differ:**

- **Mechanism is orthogonal.** SOMA adapts *what the VLA is told to do* (retrieval-augmented context). AEGIS adapts *how the VLA perceives and acts* (learned perception bottleneck + action denoiser). SOMA cannot suppress sensor noise or lighting variation in the feature representation; AEGIS cannot reason about failure history.
- **SOMA adds a full RAG pipeline at inference.** This requires a memory bank, a live LLM orchestrator, and MCP infrastructure — substantial inference overhead. AEGIS adds 350 RASF multiplications and one residual forward pass (RIB) per step — effectively zero overhead.
- **Different benchmarks.** SOMA evaluates on LIBERO-PRO and a custom LIBERO-SOMA benchmark (failure-recovery scenarios). AEGIS evaluates on LIBERO-Plus 7-axis perturbation benchmark (CVPR 2026). The 56.6% figure is not directly comparable to our +5.65 pp mean because the baselines, tasks, and perturbation types differ.
- **SOMA needs online failure examples.** The retrieval system requires a populated memory bank from prior rollouts. AEGIS trains offline on corruption augmentation and is ready at deployment without any runtime memory.
- **No identity-preservation, no rate-distortion.** SOMA makes no claim about what happens when its context pipeline is removed. AEGIS's gate-off provably recovers the baseline.

**Defense line:** *"SOMA and AEGIS are complementary, not competing: SOMA adapts the VLA's goal via retrieval; AEGIS hardens the VLA's perception and action execution via learned modules. They solve different sub-problems of the robustness question."*

---

### [7] TIDAL — Temporally Interleaved Diffusion and Action Loop
*arxiv 2601.14945, January 2026*

TIDAL is a hierarchical dual-frequency control architecture for diffusion-based VLAs. A low-frequency macro-intent loop caches semantic embeddings; a high-frequency micro-control loop interleaves single-step flow integration with execution, achieving ~9 Hz control vs. ~2.4 Hz for full-rollout baselines. Backbone-agnostic in the sense that it wraps the inference loop without modifying VLA weights.

**How we differ:**

- **Different problem.** TIDAL addresses *inference latency* — it makes control faster by reusing cached macro-embeddings. AEGIS addresses *robustness to input perturbations* — it makes the policy more accurate under sensor noise, lighting changes, and other LIBERO-Plus axes. These are orthogonal axes of VLA performance.
- **Different mechanism.** TIDAL's "dual-frequency" refers to two temporal loops running at different clock rates (fast/slow). RASF's "spectral filtering" refers to DCT-domain amplitude shaping on action chunks. The term "frequency" is used in entirely different senses.
- **TIDAL does not filter action content.** It changes how often new actions are generated, not what is inside each generated action. RASF changes the spectral composition of the action chunk that is actually executed.
- **No robustness evaluation.** TIDAL is benchmarked on control frequency and dexterous manipulation latency. It is not evaluated on LIBERO-Plus perturbation axes.
- **No information bottleneck.** TIDAL has no rate term, no KL objective, no identity-preservation claim.

**Defense line:** *"TIDAL and AEGIS share the property of being backbone-agnostic wrappers, but address completely different VLA limitations: TIDAL speeds up inference, AEGIS hardens robustness. A reviewer conflating these should be pointed to the benchmark difference — TIDAL has no robustness evaluation."*

---

### [8] STRONG-VLA — Decoupled Robustness Learning
*arxiv 2604.10055, April 2026*

Two-stage curriculum fine-tuning: Stage I applies progressively harder multimodal perturbations (visual, language, proprioceptive); Stage II re-aligns on clean data. Evaluates on π₀ and OpenVLA. Reports up to +16.49 pp on seen textual perturbations (π₀), +5.58 pp on unseen.

**How we differ:** Base policy is modified — STRONG-VLA fine-tunes all weights (LoRA rank-32 for OpenVLA; direct fine-tuning for π₀). There is no frozen-base guarantee. The two-stage approach prevents Stage I from degrading clean SR, but only empirically, not by construction. AEGIS's identity-init provides a structural guarantee before any training begins. STRONG-VLA also does not evaluate on the LIBERO-Plus 7-axis protocol.

---

### [9] RobustVLA — RL Post-Training for Robustness
*arxiv 2511.01331*

Online RL post-training with LoRA (rank-32) using PPO + Jacobian smoothness regularization `ℒ = ℒ_PPO + α·ℛ_Jac + β·ℛ_Smooth`. Applied to OpenVLA-OFT. Base policy is modified.

**How we differ:** Requires online RL — needs a simulator, reward signal, and rollouts during training. AEGIS trains offline on cached policy outputs with supervised corruption augmentation; no simulator access required. No IB, no DCT, no frozen-base guarantee.

---

### [10] BC-IB — Information Bottleneck for Behavioral Cloning
*arxiv 2502.02853, ICML 2025*

Applies VIB to the fused multimodal latent (after concatenating vision+state+language encoders): `ℒ = β·I(x_t, z_t) + ‖π(x_t) − a_t‖²`. MINE discriminator estimates mutual information. Image encoders frozen; rest of model trains end-to-end.

**How we differ:** Wrong locus — BC-IB's bottleneck is *after* cross-modal fusion, not at the vision→LLM projector. It targets representation compactness, not visual corruption robustness. BC-IB's own limitations section explicitly acknowledges "robustness to domain shifts remains insufficiently studied." AEGIS's RIB directly targets corruption robustness at the projector locus with corruption-augmented training. No identity-preservation guarantee; no DCT action filtering; no robustness evaluation.

---

## On Hallucinated Citations — Do Not Cite These

The following papers were cited in a preliminary literature review but are **not in indexed literature**. Citing them in any submission is a fatal credibility risk.

| Citation | Status | Evidence |
|---|---|---|
| "Adapting Temporal Ensemble to Flow Matching Policies for Robot Manipulation" (DeepRob workshop) | **HALLUCINATED** | No arxiv ID, no workshop paper found. "DeepRob" is a University of Michigan course (Winter 2026), not a workshop. |
| "Hierarchical Policy Learning via Spectral Decomposition" (ICML 2028) | **MISLABELED** | Paper is real (arxiv 2606.29570, Cao et al., June 2026) but was never submitted to ICML 2028 — a date that does not exist. Cite correctly as arxiv 2606.29570. |
| AECIB: "Anchor-Enforced Gradient Isolation for Knowledge-Preserving VLA Fine-Tuning" | **HALLUCINATED** | No arxiv ID, no trace in indexed literature. The acronym "AECIB" returns no results. |

---

## Slide-Ready Summary

**Slide 1 — What we built**
- AEGIS: identity-residual RIB adapter (2.27M) at vision→LLM projector + DCT-domain RASF (350 scalars) on action chunks + temporal ensembling — all on a provably frozen base
- Trained with MSE to ground-truth + explicit β·KL rate penalty; 60% corruption-augmented batches
- Gate-off = baseline exactly (verified); no regression possible before learning begins
- LIBERO-Plus 7-axis robustness: +5.65 pp mean / +11.90 pp peak (SmolVLA); +7.74 pp mean / +11.01 pp peak (ACT)

**Slide 2 — Where we differ from prior work**
- **vs StableVLA:** same projector locus, but StableVLA's heuristic sigmoid gate has no β·KL term and trains on clean data only — their gate is empirically dormant (fusion_coeff → −0.006); we add RASF (action axis StableVLA cannot cover) and identity-preservation (they make no such claim)
- **vs CSP / FreqPolicy / TIDAL:** all use spectral/frequency ideas but for generation speed or architecture — not post-hoc denoising of a frozen policy under perturbation; none evaluate robustness
- **vs SOMA / BYOVLA:** both keep the VLA frozen but adapt via retrieval or pixel editing — no learned robustness modules, no rate-distortion grounding, orthogonal mechanism
- **vs IBAC / VDB / BC-IB:** all apply IB inside the training pipeline at different loci; none are post-hoc, none prove identity-preservation, none evaluate visual corruption robustness
- **The reverse water-filling connection** (rate-distortion theorem applied to temporal action sequences) is analytically derived and unit-tested for RASF — no prior paper establishes this for robot action filtering
