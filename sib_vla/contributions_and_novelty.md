# Contributions and Novelty — Spectral Information Bottleneck for VLAs

> Two slides worth of material. Written plainly.

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

## How We Differ from the Four Most Relevant Papers

### Comparison matrix

| | **StableVLA IB-Adapter** | **IBAC-SNI** | **VDB** | **Ours (SIB)** |
|---|---|---|---|---|
| Paper | Hiranaka et al., 2024 | Igl et al., NeurIPS 2019 | Peng et al., ICLR 2019 | This work |
| **Locus** | Visual tokens → LLM interface (perception) | Policy input: state representation (RL) | IL discriminator features | Action chunk output (+ perception, Week 2) |
| **Input to bottleneck** | Visual feature tokens (B, N, D) | State embedding | Discriminator features | DCT coefficients of action chunk |
| **Spectral basis** | None — operates in raw feature space | None | None | **DCT-II along time axis** |
| **Channel model** | Channel-covariance sigmoid gating | KL to a Gaussian prior (standard VIB) | KL to a Gaussian prior (standard VIB) | **Per-band Gaussian channel, signal power estimated from policy** |
| **Rate term in loss** | No explicit rate; gating is heuristic | β · KL divergence | β · KL divergence | **β · Σ R_k = β · Σ ½ ln(1 + λ_k/σ_k²)** |
| **Signal power (λ)** | Not estimated | Not estimated | Not estimated | **EMA over policy's predicted DCT coefficients** |
| **Inference decode** | Sigmoid-gated feature passthrough | Stochastic sample (z ~ q(z\|x)) | Stochastic sample | **MMSE / Wiener gain (closed-form, no sampling)** |
| **Distortion target** | Self-reconstruction / task loss | Task reward | Adversarial (AIRL) | **Ground-truth action A_star (denoiser, not compressor)** |
| **Post-hoc on frozen model** | No — requires full VLA fine-tuning | No — trained end-to-end | No — part of the IL training loop | **Yes — trains on cached policy outputs; base VLA never touched** |
| **Interpretable per-band allocation** | No | No | No | **Yes — bits per temporal frequency band** |

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

## Slide-Ready Summary

**Slide 1 — What we built**
- Post-hoc spectral denoiser for VLA action chunks: DCT-II → per-band Wiener gain → inverse DCT
- Trained with MSE to ground-truth actions + explicit rate penalty β · Σ R_k
- 350 learned scalars; zero inference overhead; base VLA completely frozen
- Identity at initialization — task success preserved before any learning

**Slide 2 — Where we differ from prior IB work**
- **vs StableVLA IB-Adapter:** same high-level IB idea, but StableVLA gates raw feature channels with no rate term and no spectral basis; we operate in DCT frequency space with an explicit per-band rate derived from signal SNR, and we apply it post-hoc without retraining the VLA
- **vs IBAC / VDB:** both compress an intermediate representation inside the training pipeline; we filter the action output post-hoc on a frozen model, targeting a fundamentally different quantity (executed trajectory quality vs representation compactness)
- **vs VIB (Alemi 2017):** we extend VIB with three additions: a spectral DCT basis before bottlenecking, a signal-power-informed per-band channel model (λ_k estimated from the policy), and MMSE Wiener decoding instead of stochastic sampling — turning a compressor into a calibrated denoiser
- **The rate-distortion theorem connection** (reverse water-filling) is provably present in our loss for the Gaussian case and unit-tested — none of the above papers establish this analytic link for their specific loss
