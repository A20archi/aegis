# project.md — AEGIS: a plug-in robustness module for frozen VLAs

> **One sentence.** AEGIS is a small, identity-initialised plug-in that bolts two
> information-bottleneck modules onto a **frozen** VLA — one at the *vision
> connector* (visual-corruption robustness) and one at the *action chunk* (motion
> denoising / action-noise robustness) — plus temporal ensembling at the receding
> horizon, and recovers robustness the base policy loses under distribution shift
> **without retraining the backbone and without degrading clean success**.

> **Naming (read first).** `AEGIS` is the **internal** name used in this repo and in
> these working docs. **Externally** (deck / paper / slides) the method is labeled
> **"SmolVLA+SIB"** and the baseline is **"SmolVLA"**. Do **not** surface in external
> materials: the internal component names (RIB/RASF/TE), ForgeVLA, the "forge"
> recipe, the L1/L3/L4/L5 levers, or temporal-ensembling-as-a-named-lever. Both the
> AEGIS arm and the baseline arm share temporal ensembling; it is not a surfaced
> lever. See `results/RESULTS_TRACKER.md` for the external-labeling rule.

---

## 1. What this project is now (and the pivot)

The project began as a **single-locus action smoother**: a spectral information
bottleneck (SIB) on the VLA's action chunk, sold on smoothness + interpretable
per-band bit allocation. That work is complete and is preserved (see
`contributions_and_novelty.md`, `README.md`), but it is **not** the headline
anymore.

**The pivot (robustness story).** The current SOTA small-VLA, VLA-Adapter, reports
~99.6% clean success at 0.5B on LIBERO but is **tested with no perturbation**. That
is the wedge: clean success is saturated; *robustness under distribution shift is
not*. So the project repositioned from "make actions smooth" to **"make a frozen
VLA robust"**, along two axes a single locus cannot cover at once:

- **Visual robustness** — viewpoint / lighting / texture / sensor-noise shift.
- **Action robustness** — high-frequency jitter and injected action-space noise.

AEGIS is the answer: **one principle (a rate-limited information bottleneck),
inserted at two loci**, both identity-at-init so clean success is structurally
protected.

---

## 2. The three components (internal names)

**AEGIS = RIB + RASF + TE.** Full technical detail (math, shapes, insertion points)
is in `architecture.md`; this is the project-level summary.

| Component | Locus | Role | Trains | Status |
|---|---|---|---|---|
| **RIB** — Robust Information Bottleneck | vision→LLM connector (`modality_projection.proj`, D=960) | visual-corruption robustness | RIB (~2.3M) + fusion gate + action head; backbone frozen | **built, engages, measured** |
| **RASF** — Residual Adaptive Spectral Filter | sampled action chunk (post-sampler, `(B,50,7)`) | action denoising / smoothness | RASF only (~few-k params); policy frozen | **built, measured** |
| **TE** — Temporal Ensembling | receding horizon (overlapping chunks) | reactivity + stochastic-noise averaging | nothing (inference-time blend) | **built; shared by both arms** |

- **RIB** is StableVLA's idea fixed: same insertion point as the StableVLA
  IB-Adapter, but (a) **identity-initialised + residually fused** so it cannot
  degrade clean success, (b) carries an **explicit rate term** (the IB objective
  StableVLA's heuristic gate lacks), and (c) is trained by **corruption-augmented
  consistency** so the bottleneck actually learns to drop the corruption subspace.
  The StableVLA IB-Adapter, trained on clean task loss alone, stays **dormant**
  (fusion coeff ≈ −0.006 after 10k steps); RIB engages.
- **RASF** is SIB v2 as a **conservative denoiser**: identity-at-init, residual
  hard-capped (`gate_max`), per-band gain floored (can't null a band), input-adaptive,
  per-dim gate. Trained to reproduce the policy's **own clean prediction** (not the
  demo), so identity is the exact optimum on clean input and it only strips injected
  perturbation. This fixed the v1 (rate-distortion, SR→~15%) and v2-aggressive
  (denoise-to-GT, clean SR→~28%) collapses.
- **TE** position-aligns and exponentially blends overlapping chunks. It is present
  in **both** the baseline and AEGIS arms, so AEGIS's gains are reported *on top of*
  TE — the honest comparison.

---

## 3. Status & verified results

Backbone: **SmolVLA** (frozen). Benchmark: **LIBERO-Spatial** (clean) +
**LIBERO-V** (visual robustness). All numbers below are measured on our from-base
retrain at the **86%-clean checkpoint** (`on86`); `(t0)` = single-task preview
(n=20, noisy); the n=200 LIBERO-V sweep is the Wed–Fri run.

### 3.1 Clean success — LIBERO-Spatial, n=200
| config | clean SR | Wilson95 | RMS jerk |
|---|---|---|---|
| SmolVLA + TE (baseline) | **86.0%** | [80.5, 90.1] | — |
| RASF + TE (action leg only) | 84.5% | [78.8, 88.9] | 0.057 |
| RASF, no TE (action leg, pure) | 78.5% | [72.3, 83.6] | 0.659 |
| **AEGIS (RIB + RASF + TE)** | **87.5%** | [82.2, 91.4] | **0.059** |

**Read:** AEGIS is **net-additive on clean** (+1.5pp over baseline+TE, CIs overlap
→ effectively a tie-or-better) — exactly what identity-at-init promises: the
plug-in cannot structurally hurt clean SR. RASF cuts RMS jerk by ~10× (0.659→0.057).

### 3.2 Visual robustness — LIBERO-V (baseline+TE vs AEGIS)
| axis / condition | type | baseline | AEGIS | gap |
|---|---|---|---|---|
| viewpoint_medium | systematic (geometry) | 0% (t0) | 20% (t0) | **+20pp (t0)** |
| gaussian_noise_1 | stochastic | 80% (t0) | 75% (t0) | ~tie |
| viewpoint large/small, lighting, texture, motion-blur | — | _pending n=200_ | _pending_ | — |

**Framing (data-driven, not assumed):** RIB **complements** TE. TE already
neutralizes *stochastic* per-frame noise (→ tie on gaussian_noise); RIB recovers
the *systematic* visual shifts TE cannot average away (→ +20pp on viewpoint, where
the baseline collapses to 0%). That dissociation is the perception-leg headline.

---

## 4. Benchmarks

- **LIBERO-Spatial** — clean success, n=200, Wilson 95% CIs (the "no clean
  regression" gate).
- **LIBERO-V** — visual robustness, 4 axes (`sib/libero_v.py`):
  - *viewpoint* (small/medium/large) — direct MuJoCo camera orbit + offset (headline)
  - *lighting* — diffuse/direction/specular/shadow shift (sim)
  - *texture* — floor/wall/table recolor + texture swap (sim)
  - *sensor noise* (image-space) — motion/zoom/glass blur, fog, and a
    gaussian-noise severity sweep (8 levels, for the graceful-degradation curve)
  - Sim axes are applied by direct state manipulation and re-rendered on every
    (auto)reset; the noise axis reuses `sib/corruptions.py`.

---

## 5. Roadmap

| When | Goal | Status |
|---|---|---|
| **This week (due Fri 2026-06-19 15:00)** | v2 base retrain (batch 384, 30k, full LIBERO) → target **92% clean**; full n=200 LIBERO-V sweep both arms; noise graceful-degradation curve | base retraining; LIBERO-V t0 done |
| **Next week (wk of 2026-06-22)** | **Cross-architecture generalization** — port RIB+RASF to other VLA families to show AEGIS is host-agnostic | briefs written |

**Cross-architecture targets** (briefs in `multivla/`):
| host | family | plug-in fit | status |
|---|---|---|---|
| SmolVLA-500M | decoder-LLM + flow-matching | clean (this project) | done |
| NanoVLA-S (161M) | ACT enc-dec + CVAE | full re-impl as language-conditioned ACT (no public code/ckpt; OpenReview withdrawn) | code build now, GPU queued |
| TinyVLA / substitute | LLM + diffusion head | diffusion head → must hook denoised x0; MetaWorld | substitute search (TinyVLA ships no checkpoint) |

The cross-arch attachment points are the same two loci everywhere: **IB "front"** =
the vision→backbone connector; **SIB/RASF "back"** = the action-chunk head.

---

## 6. Design principles (hard constraints, all components)

1. **Backbone frozen.** No gradient to pretrained vision/LLM weights. (RIB
   additionally co-trains the lightweight action head; RASF trains nothing but itself.)
2. **Identity at initialisation.** Both modules are exact pass-throughs at step 0
   (RIB: zero-init decoder + residual fuse; RASF: gate=0). Clean success cannot
   structurally degrade.
3. **No backprop through the ODE sampler.** RASF trains on the post-sampler chunk;
   RIB trains on the single-forward flow/task loss.
4. **No tuning on the test split**, no seed/task reselection after seeing success,
   no widening a baseline's disadvantage. Steering = better engineering, not data
   massaging (see `scenario.md`).
5. **Honest negatives are results.** A tie or a clean negative is reported as such.

---

## 7. Repo map (where things live)

```
sib/
  robust_ib.py        RIB (perception leg) + FusedRobustIBProjector + inject/load
  adaptive_filter.py  RASF (action leg, conservative denoiser)
  ib_adapter.py       StableVLA IB-Adapter (baseline; goes dormant)
  bottleneck.py       SpectralActionModule (v1 SIB), GaussianChannel, build_module
  transforms.py       orthonormal DCT-II / IDCT (Parseval-tested)
  libero_v.py         LIBERO-V sim perturbations (viewpoint/lighting/texture) + grid
  corruptions.py      image-space sensor-noise corruptions
  wrapper.py          SIBPolicy + ForgeActionHeadPolicy (temporal ensembling, TE)
scripts/
  finetune_rib.py     train RIB by corruption-augmented consistency
  train_rasf.py       train RASF as a conservative denoiser (own-clean-pred target)
  finetune_ib.py      train StableVLA IB-Adapter baseline
  eval_libero_v.py    unified eval: vanilla | sib | ib | baseline | aegis (+TE)
  eval.py             clean / corruption / action-noise eval + recording
configs/              on86 configs: sib_on86, rasf_on86, ib_on86, libero_v ...
results/
  RESULTS_TRACKER.md  internal source-of-truth for the deck (verified numbers only)
  ib_on86/libero_v/{baseline,aegis}/   LIBERO-V eval outputs
  rasf_on86/          RASF clean + robustness evals
multivla/             NanoVLA / TinyVLA / substitute cross-arch briefs
presentation/         spectral_filter_vla.pptx
```

Companion docs: **`architecture.md`** (the exact mechanism), `scenario.md`
(decision playbook for ambiguous gates), `contributions_and_novelty.md` (prior-work
positioning), `results/RESULTS_TRACKER.md` (numbers + external-labeling rule).
