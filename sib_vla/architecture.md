# architecture.md — AEGIS: dual-locus robustness for a frozen VLA

> **One principle, two loci, plus a receding-horizon blend.** A rate-limited
> information bottleneck inserted at two points inside a **frozen** SmolVLA — the
> **perception locus** (vision→LLM connector, visual-corruption robustness) and the
> **action locus** (sampled action chunk, motion denoising) — both **identity-at-init**
> so clean success is structurally protected, and **temporal ensembling** at the
> receding horizon shared by both the baseline and AEGIS arms.

> Internal name = **AEGIS** (= RIB + RASF + TE). External label = **"SmolVLA+SIB"**.
> See `project.md` §"Naming" and `results/RESULTS_TRACKER.md`.

---

## 0. System overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          SmolVLA  (backbone FROZEN)                              │
│                                                                                  │
│  RGB obs ─▶ Vision Encoder (SmolVLM2, frozen) ─▶ patch tokens                    │
│                                       │                                          │
│                                       ▼                                          │
│            ┌──────────── connector: modality_projection.proj  (Linear) ───────┐ │
│            │            out = Z_mlp + tanh(coeff) · RIB(Z_mlp)                 │ │
│  ◀── PERCEPTION LOCUS ──▶  RIB: Robust Information Bottleneck (~2.3M)           │ │
│            │   enc→attn→latent(z, rate)→dec ; zero-init dec, coeff>0           │ │
│            └───────────────────────────────────────────────────┬─────────────┘ │
│                                       │ corrected tokens (B,N,960)               │
│                                       ▼                                          │
│  Action Expert  (flow-matching ODE; frozen weights, action head co-trained w/RIB)│
│                                       │  sampled chunk  A : (B, H=50, d=7)        │
│                                       ▼                                          │
│  ◀── ACTION LOCUS ──▶  RASF: Residual Adaptive Spectral Filter (identity-init)   │
│            A_hat = A + gate · ( IDCT( g(A) ⊙ DCT(A) ) − A )                      │
│                                       │  denoised chunk                          │
│                                       ▼                                          │
│  ◀── RECEDING HORIZON ──▶  TE: temporal ensemble of overlapping chunks           │
│            V[k] = Σ_s w_s · chunk_s[k+s],   w_s = exp(−coeff·s)                  │
│                                       │                                          │
│                                    env step                                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

Three insertions, none touching backbone weights. RIB trains on the single-forward
flow/task loss with corruption augmentation; RASF trains on the post-sampler chunk;
TE is inference-time only. **Both modules are exact identities at init**, so clean
success is preserved before any learning.

---

## 1. Perception locus — RIB (Robust Information Bottleneck)

`sib/robust_ib.py` · trained by `scripts/finetune_rib.py`

### 1.1 Why it exists (vs StableVLA)
StableVLA inserts an IB-Adapter at the same vision→LLM projection but trains it on
the **clean task loss alone** with a heuristic channel-covariance gate and **no rate
term**. With no corruption to defend against and no information objective, its
fusion coefficient stays at ~0 — **dormant** (measured: −0.006 after 10k steps).
RIB fixes all three: identity-init residual fusion, an explicit rate term, and
corruption-augmented training.

### 1.2 Insertion point & identity-at-init
Replaces the connector linear `policy.model…connector.modality_projection.proj`
with a `FusedRobustIBProjector`:

```
out = linear(x) + tanh(fusion_coeff) · RIB( linear(x) )
```

- RIB's decoder output layer is **zero-initialised** ⇒ `RIB(·) = 0` at step 0 ⇒
  `out == linear(x)` exactly (vanilla SmolVLA, clean SR safe).
- `fusion_coeff` is initialised **positive** (0.5493 → tanh = 0.5), **not** zero.
  Zero would also zero the decoder's gradient (`grad ∝ tanh(coeff)·dZ_ib`) — a
  dead start that keeps RIB dormant forever (StableVLA's failure). Positive coeff +
  zero-init decoder = gradient flows from step 1 *and* identity holds via `Z_ib=0`.

### 1.3 Module (D=960, d_z=448 ≈ 2.3M params, < 3M budget)
```
Z_mlp (B,N,960)
  → enc Linear(960→d_z) → LayerNorm → GELU
  → 1× token self-attention (MultiheadAttention, n_heads=7)   # spatial context:
                                                              # makes localized
                                                              # corruption detectable
  → to_z Linear(d_z→d_z)          # DETERMINISTIC latent (no sampling — see 1.4)
  → dec [Linear→GELU→Linear(→960)]  (zero-init last layer)
  → residual fuse (1.2)
```

### 1.4 The rate term (deterministic IB, not sampled VIB)
The information objective StableVLA lacks: `R = ‖z‖²` (mean over the deterministic
latent), penalised as `beta · R`, with a **free-bits floor** so there is no
compression pressure below the floor.

> Sampled VIB (reparameterised `z ~ q(z|x)`, KL rate) was tried and induced
> **posterior collapse**: the task loss learned to ignore the noisy latent. The
> deterministic L2-on-latent proxy drives the encoder to drop the nuisance /
> corruption-sensitive subspace (invariance) without that pathology.

### 1.5 Training — corruption-augmented consistency
`L = task_loss( GT action | images with a corrupted fraction ) + beta · R`

- A random fraction (`--corrupt-frac`, default 0.6) of each batch's images is
  corrupted **per-image**; clean images in the same batch preserve clean SR.
- Augmentation mixes one **photometric** family (gaussian noise / blur / motion-blur
  / fog / glass-blur / brightness-contrast, severity spanning the eval point) with a
  **geometric** warp (resized-crop + shift, scale 0.75–1.15) approximating viewpoint.
- **Trains:** RIB (~2.3M) + `fusion_coeff` + the LLM action head (`lm_expert`).
  **Frozen:** everything else.
- The bottleneck thus learns to keep task-relevant info and drop the
  corruption-induced subspace ⇒ visual robustness, while never being shown the
  actual LIBERO-V perturbations (clean-train discipline on the *test* perturbations).

---

## 2. Action locus — RASF (Residual Adaptive Spectral Filter)

`sib/adaptive_filter.py` · trained by `scripts/train_rasf.py`

### 2.1 Mechanism
1D **orthonormal DCT-II** along the time axis of the sampled chunk `A : (B,H=50,d=7)`,
a per-band adaptive gain, inverse DCT, applied as a **bounded residual**:

```
C   = DCT(A)                                      # (B,H,d) temporal spectrum
e   = log mean_d C²                               # (B,H) per-band log-energy
g   = floor + (1−floor) · sigmoid( band_logit + MLP(e) )   # ∈ [floor, 1], floor=0.30
Af  = IDCT( g ⊙ C )
A_hat = A + (gate_max · tanh(gate)) · (Af − A)    # gate per dim, gate_max=0.6
```

### 2.2 Structural guarantees (why it cannot collapse)
1. **Identity at init** — `gate = 0` ⇒ `A_hat = A` exactly.
2. **Bounded residual** — `|A_hat − A| ≤ gate_max · |Af − A|`; the filter can never
   fully replace the chunk.
3. **Gain floor** — every band keeps ≥ `floor` (0.30) of its energy; no band can be
   nulled (kills the v2-aggressive `g→0` collapse).
4. **Input-adaptive** — clean band energies are nominal ⇒ `g ≈ 1` (all-pass);
   anomalous (perturbation) energy ⇒ MLP pulls that band's gain down ⇒ denoise.
5. **Per-dim gate** — gripper (bang-bang) vs smooth pose dims handled independently.

### 2.3 Training — conservative denoiser (own-clean-prediction target)
The critical fix: the target is the policy's **own clean prediction** `A_pred`,
**not** the ground-truth demo. Identity is then the exact optimum on clean input, so
RASF has zero incentive to touch clean actions — it only learns to strip injected
perturbation.

```
x_clean = A_pred
x_noisy = A_pred + eps                         # per-sample gaussian + sparse spikes
L = W_CLEAN  · MSE(RASF(x_clean), x_clean)     # identity on clean (DOMINANT, W=5)
  + W_DENOISE· MSE(RASF(x_noisy), x_clean)     # strip injected noise
  + W_GAIN   · (1 − g_clean)²                  # keep clean gains ≈ all-pass
  + W_GATE   · tanh(gate)²                     # mild residual regulariser
```

No GT, no jerk penalty, no rate term (`uses_rate=False`). Runs on cached chunks.

> **History (why this design):** v1 (rate-distortion SIB, lossy) collapsed clean
> SR to ~15%. v2-aggressive (denoise-toward-GT-demo) crushed the clean spectrum
> (mean gain ≈ 0.08, ~18% of action energy removed) → clean SR ~28%. v2-conservative
> (this file) keeps RMS jerk ~10× lower than vanilla while preserving clean SR.

---

## 3. Receding horizon — TE (Temporal Ensembling)

`sib/wrapper.py :: ForgeActionHeadPolicy`

Position-aligned exponential blend of overlapping action chunks:
`V[k] = Σ_s w_s · chunk_s[k+s]`, `w_s = exp(−ensemble_coeff·s)` (coeff = 0.01, ACT
default) — newer predictions weighted higher. Inference-time only; trains nothing.

**Present in both arms.** Baseline = SmolVLA + TE; AEGIS = RIB + RASF + TE. AEGIS's
robustness gains are therefore reported *on top of* TE. Empirically TE handles the
*stochastic* noise axis (per-frame averaging), and RIB handles the *systematic*
visual-shift axis TE cannot average away — the two are complementary, not redundant.

---

## 4. Unified system

```
What trains:                          What is frozen:
  RIB module (~2.3M)                    SmolVLM2 vision encoder
  fusion_coeff (1 scalar)               connector linear weights
  action head (lm_expert)  [RIB run]    flow-matching expert weights
  RASF (~few-k params)     [RASF run]   (RASF run freezes the head too)

Losses (separate training runs, locus-appropriate objective):
  RIB :  L = task_loss(GT | corrupted images) + beta · ‖z‖²        # single forward
  RASF:  L = clean-identity + denoise + gain + gate regularisers    # post-sampler chunk
```

The legs are trained **separately** (different objectives, different loci) and
**composed at eval** by `eval_libero_v.py` (`--method aegis` loads both checkpoints,
injects RIB, wraps with RASF + TE). The unifying claim is conceptual — *one
rate-limited bottleneck, two loci, both identity-at-init* — not a shared training
path. Be explicit about this; do not claim operational identity.

---

## 5. Baselines (each isolates one confound)

| method (`eval_libero_v.py --method`) | what it is | isolates |
|---|---|---|
| `vanilla` | frozen SmolVLA, no module, no TE | floor |
| `baseline` | SmolVLA **+ TE** | the honest reference AEGIS is measured against |
| `ib` | StableVLA IB-Adapter (clean-task-loss trained) | "does the published competitor engage?" → **no** (dormant) |
| `sib` | action leg only (v1 SIB / RASF) | action-locus contribution |
| `aegis` | **RIB + RASF + TE** | the full method |

RIB-vs-IB-Adapter is the make-or-break for the perception claim: same insertion
point, the difference is identity-init + rate term + corruption-augmented training.

---

## 6. LIBERO-V — the robustness benchmark

`sib/libero_v.py` (4 axes; sim axes by direct MuJoCo state manipulation + re-render,
re-applied on every (auto)reset; noise axis via `sib/corruptions.py`).

| axis | how | levels |
|---|---|---|
| **viewpoint** (headline) | orbit + offset + pitch the `agentview` camera (cam_pos/cam_quat) | small / medium / large |
| **lighting** | diffuse dim + tint + direction rotation + specular + shadow toggle | sev 0–2 |
| **texture** | recolor + texture-swap floor/wall/table materials | sev 0–2 |
| **sensor noise** (image-space) | motion / zoom / glass blur, fog; gaussian-noise **sweep** | blur sev 1; noise 8 levels (0.05→1.0) |

The gaussian-noise sweep produces the graceful-degradation curve (gap ~0 at low std
where TE copes; widens mid-curve as TE breaks down and RIB denoises; both arms
collapse to the proprio floor at the extreme tail where the image is destroyed).

---

## 7. Metrics & figures

- **Clean SR** (LIBERO-Spatial, n=200) with Wilson 95% CIs — the "no clean
  regression" gate.
- **Robustness SR by LIBERO-V axis**, baseline+TE vs AEGIS, two-proportion z-tests.
- **RMS jerk** + HF-energy fraction (RASF's smoothness claim; ~10× jerk reduction).
- **RIB fusion coefficient** over training (engaged vs dormant — the RIB-vs-IB-Adapter
  contrast figure).
- **Noise graceful-degradation curve** (both arms, gaussian-noise sweep).
- **Cross-architecture** (next phase): the same axes on NanoVLA / a TinyVLA
  substitute, to show the plug-in is host-agnostic.

---

## 8. Hard constraints (all components)

1. **Backbone frozen** — no gradient to pretrained vision/LLM weights. (RIB run
   co-trains the lightweight action head; RASF trains only itself.)
2. **Identity at init** — RIB (zero-init decoder + residual fuse) and RASF (gate=0)
   are exact pass-throughs at step 0. Clean SR cannot structurally degrade.
3. **No backprop through the ODE sampler** — RASF on the post-sampler chunk; RIB on
   the single-forward flow/task loss.
4. **No exposure to the test perturbations during training** — RIB trains on its own
   augmentation families, evaluated on the held-out LIBERO-V axes; RASF trains on
   synthetic injected noise, never the eval shift.
5. **Honest comparison** — both arms carry TE; gains are reported on top of it.

---

## 9. Repo layout

```
sib/
  robust_ib.py        RIB + FusedRobustIBProjector + inject_fused_rib / load_rib_checkpoint
  adaptive_filter.py  RASF (AdaptiveSpectralFilter, build_rasf)
  ib_adapter.py       StableVLA IB-Adapter baseline
  bottleneck.py       SpectralActionModule (v1 SIB), GaussianChannel, build_module
  transforms.py       orthonormal DCT-II / IDCT (Parseval-tested)
  libero_v.py         LIBERO-V sim perturbations + condition grid
  corruptions.py      image-space sensor-noise corruptions (device-safe)
  wrapper.py          SIBPolicy, ForgeActionHeadPolicy (TE), receding-horizon eval
  data.py · losses.py · metrics.py · recording.py · waterfill.py · utils.py
scripts/
  finetune_rib.py     RIB training (corruption-augmented consistency)
  train_rasf.py       RASF training (conservative denoiser)
  finetune_ib.py      StableVLA IB-Adapter training (baseline)
  eval_libero_v.py    unified eval: vanilla | sib | ib | baseline | aegis (+ --forge-ensemble TE)
  eval.py             clean / corruption / action-noise eval + recording
configs/              on86 configs (sib_on86, rasf_on86, ib_on86, libero_v, ...)
results/
  RESULTS_TRACKER.md            verified numbers + external-labeling rule
  ib_on86/libero_v/{baseline,aegis}/   LIBERO-V eval outputs
  rasf_on86/                    RASF clean + robustness evals
multivla/             NanoVLA / TinyVLA / substitute cross-arch briefs
```

Companion: **`project.md`** (goals, status, roadmap), `scenario.md` (decision
playbook), `contributions_and_novelty.md` (prior-work positioning).
