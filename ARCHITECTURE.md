# AEGIS — Architecture

**AEGIS** (*Adaptive Entropy-Gated Information Sieve*) is a robustness wrapper for a **frozen**
Vision–Language–Action (VLA) policy. It inserts two rate-limited information bottlenecks — one at the
**perception** interface (RIB), one at the **action** interface (RASF) — plus an inference-time
temporal consensus (TE). Every module is an **exact identity at initialization**, so attaching AEGIS
provably cannot degrade the underlying policy, and a per-suite gate recovers the base policy
*bit-exactly* whenever a module does not help.

This document specifies the architecture: the design principles, the data-flow, each module's exact
formulation, the non-regression guarantee, the rate–distortion theory that grounds the action module,
and the loci at which AEGIS attaches across five backbones.

---

## 1. Design principles

AEGIS is built around four commitments, in priority order:

1. **Frozen backbone.** No gradient flows into the VLA's vision encoder, language model, or
   action expert. AEGIS trains only the inserted modules (≈2.3M parameters), never the backbone.
2. **Dual-locus bottlenecks.** Corruption enters a VLA at two points — the visual observation and the
   sampled action chunk. AEGIS places a rate-limited bottleneck at *each*: RIB removes *systematic*
   visual shift a temporal average cannot; RASF regularizes the *temporal action spectrum*.
3. **Identity at initialization.** Both modules are exact pass-throughs at step 0. Robustness is
   therefore *strictly additive*: a module that does not help a suite is disabled at **zero clean-task
   cost**, recovering the frozen policy bit-exactly.
4. **Disclosed, gate-safe deployment.** Because gate-off equals the base policy exactly, per-suite
   gating is a structural fallback — not a hidden per-category oracle — and is reported openly.

---

## 2. Pipeline and notation

Let $\pi$ be the frozen VLA policy. A single control step maps an RGB observation to an executed action:

```
  RGB obs
    │
    ▼
 ┌──────────────┐   patch tokens   ┌──────────────────────────┐   corrected tokens
 │ Vision Enc.  │ ───────────────▶ │  W_proj  +  [ RIB ]       │ ─────────────────┐
 │  (frozen)    │      x            │  (vision→LLM connector)   │   z_out           │
 └──────────────┘                  └──────────────────────────┘                   │
                                                                                    ▼
                                                        ┌───────────────────────────────────┐
                                                        │  LLM / Action Expert  (frozen)      │
                                                        │  (flow-matching ODE / CVAE / …)     │
                                                        └───────────────────────────────────┘
                                                                                    │  sampled chunk A ∈ ℝ^{H×d}
                                                                                    ▼
                                                        ┌───────────────────────────────────┐
                                                        │        [ RASF ]  (action leg)        │
                                                        └───────────────────────────────────┘
                                                                                    │  Â
                                                                                    ▼
                                                        ┌───────────────────────────────────┐
                                                        │  [ TE ] temporal consensus (both arms)│
                                                        └───────────────────────────────────┘
                                                                                    │
                                                                                    ▼
                                                                               env step
```

**Notation.**

| Symbol | Meaning |
|---|---|
| $x$ | vision tokens after the encoder, before the connector projection $W_{\mathrm{proj}}$ |
| $z_{\mathrm{out}}$ | corrected connector output fed to the LLM / action expert |
| $A \in \mathbb{R}^{H\times d}$ | sampled action chunk ($H$ steps, $d$ action dims) |
| $\hat{A}$ | RASF-regularized chunk |
| $C$ | orthonormal DCT-II matrix over the temporal ($H$) axis |
| $\lambda_k,\ \sigma_k^2$ | per-band source variance and learned noise floor |
| $\alpha,\ \gamma$ | perception / action fusion gates (scalars, learned) |

RIB and RASF are trained **separately**, with different objectives and interfaces, and **composed at
inference**. TE is present in **both** the baseline and AEGIS arms, so every reported $\Delta$ isolates
RIB/RASF.

---

## 3. RIB — Robust Information Bottleneck (perception)

RIB sits on the vision→LLM connector and applies a **zero-initialized residual correction** to the
projected tokens:

$$
z_{\mathrm{out}} \;=\; W_{\mathrm{proj}}\,x \;+\; \tanh(\alpha)\cdot \mathrm{RIB}\!\left(W_{\mathrm{proj}}\,x\right),
\qquad \alpha,\ \mathrm{RIB}\ \text{zero-init.}
$$

**Identity at init.** At step 0, $\tanh(\alpha)=\tanh(0)=0$ *and* the RIB branch is zero-initialized, so
$z_{\mathrm{out}} = W_{\mathrm{proj}}x$ **bit-exactly** (verified $\max|z_{\mathrm{out}}-W_{\mathrm{proj}}x| = 0$).
The gate $\alpha$ is nonetheless *live from the first gradient step* — the module engages while starting
from the identity, avoiding a dead branch. The deployed RIB additionally carries a per-sample gate that
can route to the identity on benign inputs.

**Objective.** RIB is trained with a task loss plus a rate penalty on the latent:

$$
\mathcal{L}_{\mathrm{RIB}} \;=\; \mathcal{L}_{\mathrm{task}}(z_{\mathrm{out}}) \;+\; \beta_{\mathrm{RIB}}\cdot R_{\mathrm{latent}} .
$$

$R_{\mathrm{latent}}$ is a **bounded deterministic** rate proxy on the latent activations — explicitly
**not** a variational KL bottleneck (despite the "VIB" shorthand in figures). A full sampled/variational
rate caused representation collapse; the deterministic proxy sheds the corruption-sensitive subspace
without collapsing the representation. The task target is held **identical** for corrupted and clean
inputs, enforcing invariance.

**Training augmentation.** Each mini-batch is 60% generically perturbed / 40% clean, with **strict
train/test disjointness**: the training augmentation families (photometric jitter, affine viewpoint warp)
are disjoint from every evaluation perturbation family.

**Why the rate term matters.** RIB occupies the *same* connector locus as StableVLA's task-loss-only
IB-Adapter. Without the rate term (and the identity-at-init below), that design exhibits gate collapse
and unbounded compression that regresses every suite once trained to convergence. The rate penalty plus
the rate floor is the fix: post-training, RIB's fusion coefficient settles at $\approx 0.55$
($\tanh(\alpha)\approx 0.50$), where an otherwise-identical rate-free variant stays dormant.

---

## 4. RASF — Residual Adaptive Spectral Filter (action)

RASF regularizes the **temporal spectrum** of the sampled action chunk. Expressing the chunk in the
orthonormal DCT-II temporal basis $C$, RASF applies a **bounded residual** correction:

$$
\hat{A} \;=\; A \;+\; g_{\max}\,\tanh(\gamma)\,\Bigl(\underbrace{C^{\!\top}\,\mathrm{diag}(w)\,C\,A}_{\text{spectrally filtered}} \;-\; A\Bigr),
$$

with per-band gains $w\in(0,1]$ and a per-dimension, input-dependent gate $\gamma$. The gains follow from
a **per-band Gaussian channel**: for band $k$ with source variance $\lambda_k$ (an EMA over the policy's
own predictions) and learned noise floor $\sigma_k^2$, the MMSE (Wiener) gain and channel rate are

$$
g_k \;=\; \frac{\lambda_k}{\lambda_k + \sigma_k^2} \in (0,1],
\qquad
R_k \;=\; \tfrac{1}{2}\ln\!\Bigl(1 + \tfrac{\lambda_k}{\sigma_k^2}\Bigr),
$$

and $w_k := g_k$. RASF is trained to reconstruct the ground-truth action $A^\star$ under a per-band rate
penalty:

$$
\mathcal{L}_{\mathrm{RASF}} \;=\; \mathrm{MSE}(\hat{A},\,A^\star) \;+\; \beta \sum_k R_k .
$$

Because it reconstructs a *target* $A^\star$ rather than compressing the *source* $A$, the learned filter
inherits the reverse-water-filling **shape** (§7) but not its level.

**Five structural safety properties.** RASF cannot collapse the chunk:

1. **Pass-through at init** — $g_{\max}=0 \Rightarrow \hat{A}=A$ (identity).
2. **Bounded residual** — the correction is hard-capped; it cannot replace the chunk.
3. **Gain floor** — $w_k \ge \epsilon_{\min}=0.05$; no band is ever nulled.
4. **Input-adaptive** — the gate does nothing on benign input.
5. **Per-dimension gate** — gripper / pose dimensions are filtered independently.

In practice RASF keeps RMS jerk an order of magnitude below the unfiltered policy while remaining
SR-neutral on clean inputs. It adds only a few thousand scalars (e.g. SmolVLA: $H{=}50$, $d{=}7$).

---

## 5. TE — Temporal consensus

TE is a **position-aligned exponential consensus** over overlapping predicted chunks: at each executed
step, predictions for that step from successive (overlapping) chunks are combined with newer predictions
weighted higher. It is **inference-time only** and updates no weights.

Crucially, TE is present in **both arms** — the honest baseline is `host + TE`, and AEGIS is
`RIB + RASF + TE` — so every reported $\Delta$ isolates the AEGIS modules. The three mechanisms are
**complementary by construction**: TE averages out *stochastic* noise; RIB removes *systematic* visual
shift no average can; RASF regularizes the *action spectrum*.

---

## 6. The non-regression guarantee

Let $\Delta|_{t}$ denote the deviation of the composed system from the frozen policy $\pi$ at training
step $t$.

**Proposition (identity at init).** At $t=0$, $\Delta|_{0}=0$ at *both* interfaces: the perception
interface (since $\tanh(\alpha)=0$ and RIB is zero-init) and the action interface (since $g_{\max}=0$).
Hence $z_{\mathrm{out}}=W_{\mathrm{proj}}x$ and $\hat{A}=A$ bit-exactly, and the composed system
reproduces $\pi$ **exactly**. *(Verified numerically: $\max|z_{\mathrm{out}}-W_{\mathrm{proj}}x|=0$ and
$\max|\hat{A}-A|=0$ at init.)*

**Consequence — a bit-exact no-harm fallback always exists.** Closing a suite's gate recovers $\pi$
exactly, so no deployed configuration need regress. This is a guarantee about the **gate-off state and
the initialization**; the *trained, open-gate* modules are not the identity and *can* regress a suite —
we observe this on the long-horizon clean suite and disclose it rather than gate it away.

**Per-suite gating.** The gate for each suite is decided on the 3-seed suite mean and **openly
disclosed**. Because gate-off is $\pi$ bit-exactly, this is a structural fallback, not a per-category
$\max(\cdot)$ oracle. The init mechanism itself is standard (ReZero / SkipInit / LoRA-style residual
zeroing); the contribution is to leverage it as a *deployment-safety guarantee* for a frozen VLA.

---

## 7. Theoretical grounding (reverse water-filling)

The action-side rate–distortion objective admits a closed-form optimal allocation.

> **Theorem (reverse water-filling at $\theta=\beta/2$).** For independent Gaussian action bands with
> variances $\lambda_k$ sent through per-band AWGN channels and decoded by the Wiener gain (§4), the
> unique minimizer of $\sum_k [D_k + \beta R_k]$ assigns every *active* band constant distortion
> $D_k^\star = \beta/2$, and drops band $k$ iff $2\lambda_k \le \beta$ — exactly reverse water-filling
> $D_k = \min(\theta, \lambda_k)$ at level $\theta = \beta/2$.

*Proof idea.* Eliminating $\sigma_k^2$ gives a per-band cost strictly convex in $D_k$, with interior
optimum $D_k^\star=\beta/2$; the drop condition follows from KKT. The optimum is machine-checked to
$\max_k|D_k-\beta/2| = 5.1\times10^{-16}$ (6/6 unit tests). Full statement and proof, plus the
MMSE-optimality of the Wiener decoder, are in the supplementary.

**What transfers.** The theorem governs the *idealized compression* model, not the deployed *denoising*
RASF. Empirically, the learned filter matches the reverse-water-filling **shape** (per-band rate profile,
Pearson $r$ up to $0.991$, rising monotonically as $\beta$ grows) but **not** the water level, which
sits several orders above $\beta/2$ — consistent with a denoising rather than a pure-compression
objective. We report this as validation of the *allocation profile*, not a claim that the level transfers.

---

## 8. Portability across backbones

AEGIS attaches at a backbone's **vision→LLM connector** (RIB) and on its **sampled action chunk**
(RASF), regardless of the backbone's family, normalization scheme, or action head. It has been ported to
five structurally different VLAs spanning 88M–3B parameters:

| Backbone | Family | RIB locus | RASF chunk | Modules | Notes |
|---|---|---|---|---|---|
| **ACT-88M** | ResNet-18 + CVAE | `encoder_img_feat_input_proj` | $100{\times}7$ | RIB only | action head **frozen** → cleanest module isolation |
| **SmolVLA-0.5B** | SigLIP + flow-matching | connector modality projection | $50{\times}7$ | RIB + RASF (+ light head) | primary host |
| **VLA-Adapter-Pro-0.5B** | compact adapter VLA | connector projection | chunk | RIB + RASF | head-to-head vs. StableVLA at the same locus |
| **$\pi_{0.5}$-3B** | PaliGemma (SigLIP→Gemma) + action expert | post-connector vision tokens | chunk | RIB + RASF + geom. canonicalizer + floored OOD gate | zero clean tax |
| **GR00T-N1.5-3B** | Eagle + diffusion transformer | `eagle_model.mlp1` connector | $16{\times}32$ | RIB + RASF | 3B generalist |

The trainable footprint is small and backbone-agnostic: RIB adds ≈1.3–2.5M parameters (per backbone),
RASF a few thousand scalars, and the VLM backbone is updated by **zero** weights. Because the modules are
identity at init, portability is *safe by construction* — attaching AEGIS to a new backbone cannot
degrade it before training, and a closed gate recovers it exactly after.

> **Note.** Standard test-time adaptation baselines that presume a CNN+BatchNorm backbone (e.g. BN-adapt)
> are **structurally inapplicable** to this family: modern VLA encoders (SigLIP, Eagle, and even the
> lerobot ACT backbone) carry *zero* BatchNorm layers. AEGIS attaches at the connector regardless of
> normalization scheme — a portability property such methods lack.

---

## 9. Composition at inference

At evaluation the three modules compose in a single forward pass with no backbone gradient:

```
z_out = W_proj·x + tanh(α)·RIB(W_proj·x)      # perception correction (identity if gate closed)
A     = ActionExpert(z_out)                    # frozen sampler (flow-matching ODE / CVAE / diffusion)
Â     = A + g_max·tanh(γ)·(Cᵀ diag(w) C A − A) # action spectral regularization (identity if g_max=0)
a_t   = TE(Â, buffer)                          # position-aligned exponential consensus (both arms)
```

Selecting the arm is a configuration switch (`--method {vanilla, baseline, aegis}`): `vanilla` = frozen
policy, no modules/TE; `baseline` = host + TE (the honest reference); `aegis` = RIB + RASF + TE. Per-suite
gate-off reproduces `baseline` bit-exactly.

---

## 10. Property summary

| Property | RIB (perception) | RASF (action) |
|---|---|---|
| Interface | vision→LLM connector | sampled action chunk |
| Correction form | zero-init residual on tokens | bounded DCT-II spectral residual |
| Identity at init | $\tanh(\alpha){=}0$, RIB zero-init | $g_{\max}{=}0$ |
| Objective | task loss + deterministic rate proxy | MSE to $A^\star$ + per-band rate |
| Rate mechanism | bounded latent rate (not variational) | Wiener gain $g_k=\lambda_k/(\lambda_k+\sigma_k^2)$ |
| Collapse safeguards | per-sample identity gate, rate floor | bounded residual, gain floor $\epsilon_{\min}{=}0.05$, per-dim gate |
| Params | ≈1.3–2.5M | few thousand scalars |
| Backbone weights updated | 0 | 0 |

**In one line:** AEGIS is a pair of identity-initialized, rate-limited bottlenecks — one on the vision
connector, one on the action spectrum — that add robustness to a frozen VLA with a provable, bit-exact
no-harm fallback, portable across five backbones and grounded in a reverse-water-filling optimality
result on the action side.
