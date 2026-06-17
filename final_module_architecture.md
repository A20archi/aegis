# final_module_architecture.md — AEGIS

**Dual-locus robustness for a frozen VLA.** Final, number-backed module spec.

> **One principle, two interfaces, one consensus.** A rate-limited information
> bottleneck realised at two points inside a **frozen** SmolVLA — the **perception
> interface** (vision→LLM connector) and the **action interface** (sampled action
> chunk) — both **pass-through at initialisation** so clean success is protected by
> construction, plus a **receding-horizon consensus** (TE) shared by both arms.

> Internal name **AEGIS** = **RIB** + **RASF** + **TE**. External label **"SmolVLA+SIB"**.
> This file supersedes `sib_vla/architecture.md` as the final consolidated spec; the
> authoritative reproduction recipe remains the source (`sib_vla/sib/`, `scripts/`).

---

## 0. System overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SmolVLA  (backbone FROZEN)                              │
│                                                                                │
│  RGB obs ─▶ Vision Encoder (frozen) ─▶ patch tokens                            │
│                                     │                                          │
│            ┌──── connector: modality_projection.proj (Linear) ──────────────┐ │
│            │   robustness-corrected projection (pass-through at init)        │ │
│ ◀ PERCEPTION INTERFACE ▶  RIB  (~2.3M, VIB @ vision→LLM connector)            │ │
│            └─────────────────────────────────────────────┬──────────────────┘ │
│                                     │ corrected tokens (B,N,960)               │
│                                     ▼                                          │
│  Action Expert (flow-matching ODE; frozen; head co-trained w/ RIB)            │
│                                     │  sampled chunk  A : (B,H=50,d=7)          │
│                                     ▼                                          │
│ ◀ ACTION INTERFACE ▶  RASF  (bounded spectral residual, pass-through init)    │
│                                     │  regularised chunk                       │
│                                     ▼                                          │
│ ◀ RECEDING HORIZON ▶  TE  (consensus over overlapping chunks; both arms)      │
│                                     │                                          │
│                                  env step                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

Three insertions, none touching backbone weights. **Both modules are exact
pass-throughs at init** (RIB `fusion_scale=0`, RASF `gate_max=0` → identical forward
pass), so clean success is preserved before any learning and **robustness is strictly
additive**. This is also what makes per-suite gating provably safe: a module turned off
is the base policy *exactly*, not an approximation.

---

## 1. Perception interface — RIB  (Robust Information Bottleneck)

`sib/robust_ib.py` · trained by `scripts/finetune_rib.py`

- **Insertion:** replaces the connector linear `…connector.modality_projection.proj`
  with a fused projector = original linear **+ gated robustness correction**. At step 0
  the correction contributes exactly nothing (identical to stock connector); the gate is
  nonetheless **live from the first gradient step** (engaging-yet-identity — avoids the
  dead-start trap). Verified non-dormant: `rib_on86.pt fusion_coeff=0.5508 → tanh=0.501`.
- **Module:** compact encode → spatial-context-mixing → **deterministic** latent
  (~2.27M params, < 3M budget) → decode → residual fuse. Spatial mixing localises *where*
  a corruption sits rather than treating tokens identically.
- **Objective:** bounded information-rate penalty with a floor (no compression pressure
  below it → free to be lossless on benign input). A sampled/variational rate caused
  **representation collapse**; the deterministic proxy sheds the corruption-sensitive
  subspace without that pathology.
- **Training:** robustness-shaped consistency — a fraction of each batch is perturbed by
  *generic* augmentation families (photometric + geometric warp), the rest benign, task
  target held fixed. **Never shown the eval perturbations** (strict train/test split).
  Trains RIB + fusion gate + lightweight action head; everything else frozen.

RIB's job is the **visual-robustness axis** (systematic shifts a temporal average
cannot remove).

---

## 2. Action interface — RASF  (Residual Adaptive Spectral Filter)

`sib/adaptive_filter.py` · trained by `scripts/train_rasf.py`

- **Mechanism:** sampled chunk `A:(B,H=50,d=7)` → orthonormal DCT-II along time →
  input-adaptive per-band gain → inverse transform → committed as a **bounded residual**:
  `A_hat = A + gate_max·tanh(gate)·(filtered − A)`. Benign chunks ⇒ gain ≈ all-pass ⇒
  pass-through; anomalous band energy ⇒ that band is pulled down ⇒ perturbation stripped.
- **Cannot collapse — five structural guarantees:** (1) pass-through at init
  (`gate_max=0` → `A_hat=A`); (2) bounded residual (hard cap, can't replace the policy's
  chunk); (3) gain floor (every band keeps a minimum energy; no band nulled);
  (4) input-adaptive (do-nothing is learned on benign input); (5) per-dim gate (gripper
  vs pose handled independently). On86 config: `gate_max=0.95, gain_floor=0.05`,
  effective strength ~0.85/dim.
- **Training:** conservative self-referential denoiser — supervision target is the
  policy's **own benign prediction**, so pass-through is the *exact optimum* on benign
  input (zero incentive to touch clean actions). Runs on cached chunks; no external
  target, no jerk penalty, no rate term. Keeps RMS jerk ~10× below the unfiltered policy.

RASF's job is the **action-spectrum axis** (motion regularity + injected action-noise
retention), complementary to RIB's perceptual axis.

---

## 3. Receding horizon — TE  (Temporal Ensembling)

`sib/wrapper.py`. Position-aligned exponential consensus over overlapping chunks (newer
predictions weighted higher). Inference-time only; trains nothing. **Present in both
arms** → baseline = SmolVLA+TE, AEGIS = RIB+RASF+TE, so every reported Δ isolates the
AEGIS modules. TE handles the *stochastic* noise axis (per-frame averaging); RIB handles
the *systematic* visual-shift axis no temporal average can remove — complementary by
construction.

---

## 4. Final verified results (LIBERO, n=200 unless noted)

**Clean — Spatial headline (deployment protocol):** base **86.0** / AEGIS **87.5**
(+1.5). AEGIS clean SR 0.875 verified, Wilson-95 CI [82.2–91.4].

**Clean — 4-suite, strict `n_action_steps=1`, per-suite gating** (module off ≡ base
exactly): AEGIS ≥ base on every suite, avg **85.25 vs 83.5 (+1.75)**.

**Robustness — LIBERO-V, Spatial, n=200/axis (★ headline):** AEGIS **wins all 6 axes**,
mean **+14.1**.

| axis | base+TE | AEGIS | Δ |
|---|---|---|---|
| motion blur | 4.0 | 50.0 | **+46.0** |
| gaussian noise σ=0.12 | 47.0 | 61.0 | +14.0 |
| lighting | 75.0 | 84.5 | +9.5 |
| viewpoint (moderate) | 11.0 | 20.5 | +9.5 |
| texture | 82.0 | 86.5 | +4.5 |
| viewpoint (extreme) | 0.0 | 1.0 | +1.0 |
| **mean** | **36.5** | **50.6** | **+14.1** |

**Noise graceful-degradation sweep (Spatial, n=200/level):** Δ rises to a **peak +24.5 at
σ=0.30, where base+TE is dead (0/200) and AEGIS still completes 24.5%**; base flatlines at
0% for every σ≥0.30 while AEGIS keeps operating.

**Behavior-cloning / spatial-generalization probe (object-offset, n=100):** SR degrades
gracefully (3cm: 72/78, 5cm: 51/57) — perception-conditioned, not memorized replay;
AEGIS holds a constant **+6.0** at both shifts.

Protocol: `n_action_steps=1`, 10 flow-matching denoise steps, per-suite max-steps,
single seed (42), LIBERO fixed init-states. Full tables + caveats in
[results_smolVLA.md](results_smolVLA.md).

---

## 5. Unified system & baselines

```
Trains:                               Frozen:
  RIB module (~2.27M)                   vision encoder
  fusion gate (1 scalar)                connector linear weights
  action head           [RIB run]       flow-matching expert
  RASF (~few-k params)  [RASF run]      (RASF run freezes the head too)
```
Two legs trained **separately** (different objectives/interfaces), composed at eval by
`eval_libero_v.py --method aegis` (injects RIB, wraps RASF + TE).

| `--method` | what | isolates |
|---|---|---|
| `vanilla` | frozen SmolVLA, no module/TE | floor |
| `baseline` | SmolVLA **+ TE** | the honest reference |
| `ib` | naive connector bottleneck | "does a naive bottleneck engage?" → no (dormant) |
| `sib` | action leg only | action-interface contribution |
| `aegis` | **RIB + RASF + TE** | the full method |

---

## 6. Hard constraints

1. **Backbone frozen** — no gradient to pretrained vision/LLM weights.
2. **Pass-through at init** — RIB and RASF are exact identities at step 0; clean SR
   cannot structurally degrade; robustness is strictly additive; gating is provably safe.
3. **No backprop through the ODE sampler** — RASF on the post-sampler chunk; RIB on the
   single-forward flow/task loss.
4. **No exposure to test perturbations in training** — generic augmentation only;
   LIBERO-V axes are held out.
5. **Honest comparison** — both arms carry TE; all gains reported on top of it.

---

## 7. Scope & next steps

- Robustness is **Spatial-only** (the modules' training distribution); cross-suite
  corrupted eval (Object/Goal) is the obvious next generalization test, not yet run.
- Reproduction gap: both arms ~5–10pp under paper clean SR under the strict protocol.
- Cross-architecture (NanoVLA, GR00T N1.5) is the next phase, not in this report.

---

## 8. Repo layout

```
sib/
  robust_ib.py        RIB + FusedRobustIBProjector + inject_fused_rib / load_rib_checkpoint
  adaptive_filter.py  RASF (AdaptiveSpectralFilter, build_rasf)
  ib_adapter.py       naive connector-bottleneck baseline
  transforms.py       orthonormal DCT-II / IDCT (Parseval-tested)
  libero_v.py         LIBERO-V sim perturbations + condition grid (incl. object_offset)
  corruptions.py      image-space sensor-noise corruptions
  wrapper.py          consensus policy wrapper (TE), receding-horizon eval
scripts/
  finetune_rib.py · train_rasf.py · finetune_ib.py · eval_libero_v.py
configs/              on86 configs (sib_on86, rasf_on86, ib_on86, libero_v, ...)
results/              ib_on86/, rasf_on86/, robust_spatial/, allsuites/ (eval outputs)
```

Companions: [results_smolVLA.md](results_smolVLA.md) (final numbers),
`sib_vla/project.md` (goals/status), `sib_vla/architecture.md` (capability-level prose).
