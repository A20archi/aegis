# architecture.md — AEGIS: dual-locus robustness for a frozen VLA

> **One principle, two interfaces, plus a receding-horizon consensus.** A rate-limited
> information bottleneck realised at two points inside a **frozen** SmolVLA — the
> **perception interface** (vision→LLM connector, visual-corruption robustness) and
> the **action interface** (sampled action chunk, motion regularity) — both
> **pass-through at initialisation** so clean success is protected by construction,
> and a **receding-horizon consensus** step shared by both the baseline and AEGIS arms.

> Internal name = **AEGIS** (= RIB + RASF + TE). External label = **"SmolVLA+SIB"**.
> See `project.md` §"Naming" and `results/RESULTS_TRACKER.md`.

> **Scope of this doc.** Capability-level description: *what each component does, where
> it attaches, and why the design is structurally sound*. The exact reproduction
> recipe — initialisation schedule, objective weightings, band/gate parametrisation,
> training-target construction — is held in the source (`sib/`) and the training
> scripts, which are the authoritative spec. This prose deliberately stays one level
> above that.

---

## 0. System overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          SmolVLA  (backbone FROZEN)                              │
│                                                                                  │
│  RGB obs ─▶ Vision Encoder (frozen) ─▶ patch tokens                              │
│                                       │                                          │
│                                       ▼                                          │
│            ┌──────────── connector: modality_projection.proj  (Linear) ───────┐ │
│            │   robustness-corrected projection (pass-through at init)          │ │
│  ◀── PERCEPTION INTERFACE ──▶  RIB: perception-interface bottleneck (~2.3M)     │ │
│            └───────────────────────────────────────────────────┬─────────────┘ │
│                                       │ corrected tokens (B,N,960)               │
│                                       ▼                                          │
│  Action Expert  (flow-matching ODE; frozen weights, action head co-trained w/RIB)│
│                                       │  sampled chunk  A : (B, H=50, d=7)        │
│                                       ▼                                          │
│  ◀── ACTION INTERFACE ──▶  RASF: action-interface regulariser (pass-through init)│
│            bounded residual correction in a temporal-spectral basis              │
│                                       │  regularised chunk                       │
│                                       ▼                                          │
│  ◀── RECEDING HORIZON ──▶  TE: consensus over overlapping chunks                 │
│                                       │                                          │
│                                    env step                                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

Three insertions, none touching backbone weights. RIB trains on the single-forward
flow/task loss with a robustness-shaped augmentation; RASF trains on the post-sampler
chunk; TE is inference-time only. **Both modules are exact pass-throughs at init**, so
clean success is preserved before any learning — robustness is strictly additive.

---

## 1. Perception interface — RIB

`sib/robust_ib.py` · trained by `scripts/finetune_rib.py`

### 1.1 Why it exists
A bottleneck placed at the vision→LLM projection only helps if it *engages*. The naive
version — clean task loss alone, a heuristic gate, no information objective — has
nothing to defend against and no pressure to compress, so its fusion contribution
converges to ~0 and it stays **dormant** (measured on our baseline: −0.006 after 10k
steps). RIB is the design that makes the interface pay off, through three coupled
choices below.

### 1.2 Insertion point & pass-through at init
Replaces the connector linear `policy.model…connector.modality_projection.proj` with a
fused projector whose output is the original linear **plus a gated robustness
correction**. The correction is constructed so that:

- at step 0 it contributes exactly nothing ⇒ the projector is **identical** to the
  stock connector ⇒ clean SR is safe before any training; and
- the coupling is nonetheless **live from the first gradient step** — the design
  avoids the dead-start trap where a pass-through also zeroes the module's gradient
  and the bottleneck never wakes up. Engaging-yet-identity at init is the crux, and
  the parametrisation that achieves both is in the source.

### 1.3 Module (D=960, latent ≈ 2.3M params, < 3M budget)
A compact encode → spatial-context-mixing → latent → decode stack with a residual
fuse back into the connector stream. The latent is **deterministic**. Spatial context
mixing across tokens is what lets the module localise where a corruption sits, rather
than treating every token identically. Shapes and layer widths are in the source.

### 1.4 The information-rate objective
The compression pressure the naive version lacks: a bounded penalty on the latent's
information rate, with a floor below which there is **no** compression pressure (so the
module is free to be lossless on benign input). A sampled/variational rate was tried
and induced **representation collapse** — the task path learned to ignore the noisy
latent. The deterministic rate proxy drives the encoder to shed the
corruption-sensitive subspace (invariance) *without* that pathology. The exact proxy
and schedule are in the source.

### 1.5 Training — robustness-shaped consistency
The module learns invariance from a **self-generated** robustness signal: a fraction of
each batch's inputs is perturbed by *generic* augmentation families (a photometric
family plus a geometric warp that stands in for viewpoint), while the rest stay benign
to preserve clean SR; the task target is held fixed across both. The module thus learns
to keep task-relevant information and drop the perturbation-induced subspace ⇒ visual
robustness — and it is **never shown the actual evaluation perturbations** (strict
train/test separation on the shift). Trains: RIB + its fusion gate + the lightweight
action head. Frozen: everything else. The corruption fraction, family mix, and severity
spans are in the source.

---

## 2. Action interface — RASF

`sib/adaptive_filter.py` · trained by `scripts/train_rasf.py`

### 2.1 Mechanism
The sampled chunk `A : (B,H=50,d=7)` is mapped into a **temporal-spectral basis**
(orthonormal transform along time), an **input-adaptive per-band gain** is applied, the
chunk is mapped back, and the change is committed as a **bounded residual** correction
to the original chunk. Benign chunks have nominal band energies ⇒ the gain is ≈ all-pass
⇒ the chunk passes through; anomalous band energy ⇒ the gain pulls that band down ⇒ the
perturbation is regularised away. The transform, the gain parametrisation, and the
bound values are in the source.

### 2.2 Structural guarantees (why it cannot collapse)
1. **Pass-through at init** — the correction is exactly zero at step 0 ⇒ `A_hat = A`.
2. **Bounded residual** — the correction magnitude is hard-capped, so the filter can
   never fully replace the policy's own chunk.
3. **Gain floor** — every band retains a guaranteed minimum of its energy; no band can
   be nulled (this rules out the aggressive-variant collapse where gains drive to zero).
4. **Input-adaptive** — the response is conditioned on the chunk's own spectrum, so
   "do nothing" is the learned behaviour on benign input.
5. **Per-dim gate** — discrete (gripper) and smooth (pose) action dims are handled
   independently.

### 2.3 Training — conservative, self-referential denoiser
The design choice that makes it conservative: the supervision target is the policy's
**own benign prediction**, not an external demonstration. Pass-through is then the *exact
optimum* on benign input, so the module has zero incentive to touch clean actions — it
only learns to strip injected perturbation. No external target, no jerk penalty, no rate
term; it runs on cached chunks. The objective weighting that balances "identity on
benign" against "strip noise" is in the source.

> **Why this form.** Earlier variants traded clean success for smoothness — a lossy
> rate-distortion form, and an aggressive denoise-toward-external-target form that
> over-suppressed the benign spectrum. The current self-referential form keeps RMS jerk
> ~10× lower than the unfiltered policy while leaving clean success intact.

---

## 3. Receding horizon — TE

`sib/wrapper.py :: ForgeActionHeadPolicy`

Position-aligned exponential consensus over overlapping action chunks — newer
predictions weighted higher. Inference-time only; trains nothing. **Present in both
arms.** Baseline = SmolVLA + TE; AEGIS = RIB + RASF + TE, so AEGIS's robustness gains
are reported *on top of* it. Empirically the consensus step handles the *stochastic*
noise axis (per-frame averaging) and RIB handles the *systematic* visual-shift axis no
temporal average can remove — the two are complementary by construction, not redundant.

---

## 4. Unified system

```
What trains:                          What is frozen:
  RIB module (~2.3M)                    vision encoder
  fusion gate (1 scalar)                connector linear weights
  action head            [RIB run]      flow-matching expert weights
  RASF (~few-k params)   [RASF run]     (RASF run freezes the head too)

Losses (separate training runs, interface-appropriate objective):
  RIB :  task loss on robustness-shaped inputs + bounded rate penalty   # single forward
  RASF:  benign-identity + denoise + gain + gate regularisers           # post-sampler chunk
```

The two legs are trained **separately** (different objectives, different interfaces)
and **composed at eval** by `eval_libero_v.py` (`--method aegis` loads both checkpoints,
injects RIB, wraps with RASF + TE). The unifying claim is the **principle** — *one
rate-limited bottleneck, realised at two interfaces, both pass-through at init* — which
is what gives the method its defining property: robustness you can add at either
interface without ever risking clean success.

---

## 5. Baselines (each isolates one confound)

| method (`eval_libero_v.py --method`) | what it is | isolates |
|---|---|---|
| `vanilla` | frozen SmolVLA, no module, no TE | floor |
| `baseline` | SmolVLA **+ TE** | the honest reference AEGIS is measured against |
| `ib` | naive connector bottleneck (clean-task-loss trained) | "does a naive bottleneck engage?" → **no** (dormant) |
| `sib` | action leg only | action-interface contribution |
| `aegis` | **RIB + RASF + TE** | the full method |

The RIB-vs-naive-bottleneck contrast is the make-or-break for the perception claim:
same insertion point, and the difference is entirely RIB's three design choices
(engaging-identity init, rate objective, robustness-shaped training).

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
where the consensus step copes; widens mid-curve as it breaks down and RIB denoises;
both arms collapse to the proprio floor at the extreme tail where the image is destroyed).

---

## 7. Metrics & figures

- **Clean SR** (LIBERO-Spatial, n=200) with Wilson 95% CIs — the "no clean
  regression" gate.
- **Robustness SR by LIBERO-V axis**, baseline+TE vs AEGIS, two-proportion z-tests.
- **RMS jerk** + HF-energy fraction (the action leg's motion-regularity claim; ~10×
  jerk reduction).
- **RIB fusion coefficient** over training (engaged vs dormant — the headline
  engaging-vs-naive contrast figure).
- **Noise graceful-degradation curve** (both arms, gaussian-noise sweep).
- **Cross-architecture** (next phase): the same axes on other VLA families, to show the
  plug-in is host-agnostic.

---

## 8. Hard constraints (all components)

1. **Backbone frozen** — no gradient to pretrained vision/LLM weights. (RIB run
   co-trains the lightweight action head; RASF trains only itself.)
2. **Pass-through at init** — RIB and RASF are exact identities at step 0. Clean SR
   cannot structurally degrade; robustness is strictly additive.
3. **No backprop through the ODE sampler** — RASF on the post-sampler chunk; RIB on
   the single-forward flow/task loss.
4. **No exposure to the test perturbations during training** — RIB trains on its own
   generic augmentation families, evaluated on the held-out LIBERO-V axes; RASF trains
   on synthetic injected noise, never the eval shift.
5. **Honest comparison** — both arms carry TE; gains are reported on top of it.

---

## 9. Repo layout

```
sib/
  robust_ib.py        RIB + FusedRobustIBProjector + inject_fused_rib / load_rib_checkpoint
  adaptive_filter.py  RASF (AdaptiveSpectralFilter, build_rasf)
  ib_adapter.py       naive connector-bottleneck baseline
  bottleneck.py       SpectralActionModule (v1), GaussianChannel, build_module
  transforms.py       orthonormal DCT-II / IDCT (Parseval-tested)
  libero_v.py         LIBERO-V sim perturbations + condition grid
  corruptions.py      image-space sensor-noise corruptions (device-safe)
  wrapper.py          SIBPolicy, ForgeActionHeadPolicy (consensus), receding-horizon eval
  data.py · losses.py · metrics.py · recording.py · waterfill.py · utils.py
scripts/
  finetune_rib.py     RIB training (robustness-shaped consistency)
  train_rasf.py       RASF training (conservative self-referential denoiser)
  finetune_ib.py      naive connector-bottleneck baseline training
  eval_libero_v.py    unified eval: vanilla | sib | ib | baseline | aegis (+ --forge-ensemble TE)
  eval.py             clean / corruption / action-noise eval + recording
configs/              on86 configs (sib_on86, rasf_on86, ib_on86, libero_v, ...)
results/
  RESULTS_TRACKER.md            verified numbers + external-labeling rule
  ib_on86/libero_v/{baseline,aegis}/   LIBERO-V eval outputs
  rasf_on86/                    RASF clean + robustness evals
multivla/             cross-arch briefs
```

Companion: **`project.md`** (goals, status, roadmap), `scenario.md` (decision
playbook), `contributions_and_novelty.md` (prior-work positioning). The authoritative,
reproducible spec is the **source itself**, by design.
