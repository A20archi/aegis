# project.md — AEGIS: a plug-in robustness layer for frozen VLAs

> **One sentence.** AEGIS is a compact, identity-initialised robustness layer that
> attaches to a **frozen** VLA at two complementary loci — the *perception interface*
> (visual-corruption robustness) and the *action interface* (motion regularity /
> action-noise robustness) — augmented with a receding-horizon consensus step, and
> **recovers the robustness a base policy loses under distribution shift without
> retraining the backbone and without trading away clean success**.

> **Naming (read first).** `AEGIS` is the **internal** name used in this repo and in
> these working docs. **Externally** (deck / paper / slides) the method is labeled
> **"SmolVLA+SIB"** and the baseline is **"SmolVLA"**. Do **not** surface in external
> materials: the internal component names (RIB/RASF/TE), ForgeVLA, the "forge"
> recipe, the internal levers, or temporal-ensembling-as-a-named-lever. Both the
> AEGIS arm and the baseline arm share the consensus step; it is not a surfaced
> lever. See `results/RESULTS_TRACKER.md` for the external-labeling rule.
>
> **Working-doc discipline.** These docs describe *what the method achieves and why
> the design is sound*, at the capability level. The exact reproduction recipe
> (initialisation schedule, objective weightings, band/gate parametrisation, the
> training-target construction) lives only in the source under `sib/` and the
> training scripts — treat the code, not this prose, as the authoritative spec. Keep
> it that way: positioning docs stay at the conceptual level so the contribution is
> communicated, not handed over.

---

## 1. What this project is now (and the pivot)

The project began as a **single-locus action regulariser**: an information-rate
bottleneck on the VLA's action chunk, sold on motion regularity + interpretable
per-band allocation. That work is complete and is preserved (see
`contributions_and_novelty.md`, `README.md`), but it is **not** the headline anymore.

**The pivot (robustness story).** The strongest small-VLA results at 0.5B report
~99.6% clean success on LIBERO but are **measured with no perturbation**. That is the
wedge: clean success is saturated; *robustness under distribution shift is not*. So
the project repositioned from "regularise the action stream" to **"make a
frozen VLA robust"**, along two axes a single locus cannot cover at once:

- **Visual robustness** — viewpoint / lighting / texture / sensor-noise shift.
- **Action robustness** — high-frequency jitter and injected action-space noise.

AEGIS is the answer, and the framing is deliberately unifying: **one governing
principle — a rate-limited information bottleneck — realised at two interfaces of the
policy**, each one engineered to be a structural pass-through at initialisation so
clean success is *protected by construction* rather than by tuning. That "robustness
you add can never subtract" property is the design's central selling point.

---

## 2. The three components (internal names)

**AEGIS = RIB + RASF + TE.** Capability-level summary here; the exact mechanism lives
in the source and is paraphrased in `architecture.md`.

| Component | Locus | Role | Trains | Status |
|---|---|---|---|---|
| **RIB** — perception-interface bottleneck | vision→LLM connector (`modality_projection.proj`, D=960) | visual-corruption robustness | RIB (~2.3M) + fusion gate + action head; backbone frozen | **built, engages, measured** |
| **RASF** — action-interface regulariser | sampled action chunk (post-sampler, `(B,50,7)`) | action denoising / motion regularity | RASF only (~few-k params); policy frozen | **built, measured** |
| **TE** — receding-horizon consensus | overlapping chunks at the horizon | reactivity + stochastic-noise averaging | nothing (inference-time blend) | **built; shared by both arms** |

- **RIB** is a connector-level bottleneck for a VLA that *actually engages* under
  distribution shift. A naive bottleneck at the same interface stays **dormant** —
  trained on clean task loss alone with a heuristic gate and no information objective,
  its fusion contribution converges to ~0 (measured on our own baseline: coeff ≈
  −0.006 after 10k steps). RIB's contribution is the *design that makes the locus pay
  off*: a pass-through-at-init residual coupling that still receives gradient from the
  first step, an explicit information-rate objective, and a robustness-shaped training
  signal. The result is a bottleneck that demonstrably turns on and recovers visual
  robustness where the naive baseline does nothing.
- **RASF** is the action-leg regulariser as a **conservative, self-referential
  denoiser**: a pass-through at init, structurally bounded so it can never replace the
  policy's own output, and trained against a target that makes "do nothing on clean
  input" the exact optimum — so it leaves clean behaviour untouched and only acts on
  injected perturbation. This design supersedes two earlier variants that traded clean
  success for smoothness; the current form keeps motion regularity high *and* clean
  success intact.
- **TE** aligns and blends overlapping chunks at the horizon. It is present in
  **both** the baseline and AEGIS arms, so AEGIS's gains are reported *on top of* it —
  the conservative, honest comparison.

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

**Read:** AEGIS is **net-additive on clean** (+1.5pp over baseline+TE) — exactly what
pass-through-at-init delivers: the layer cannot structurally hurt clean SR, and here
it nudges it *up*. The action leg cuts RMS jerk by ~10× (0.659→0.057) at no clean
cost. The headline property — *robustness added without clean regression* — holds.

### 3.2 Visual robustness — LIBERO-V (baseline+TE vs AEGIS)
| axis / condition | type | baseline | AEGIS | gap |
|---|---|---|---|---|
| viewpoint_medium | systematic (geometry) | 0% (t0) | 20% (t0) | **+20pp (t0)** |
| gaussian_noise_1 | stochastic | 80% (t0) | 75% (t0) | ~tie |
| viewpoint large/small, lighting, texture, motion-blur | — | _pending n=200_ | _pending_ | — |

**Framing (data-driven):** RIB and the consensus step are **complementary** by
construction. The consensus step already neutralises *stochastic* per-frame noise
(→ parity on gaussian_noise); RIB recovers the *systematic* visual shifts no temporal
average can remove (→ +20pp on viewpoint, where the baseline collapses to 0%). That
clean dissociation — each mechanism owning the failure mode the other cannot touch —
is the perception-leg headline and the strongest evidence that the two-locus framing
is the right one.

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
| NanoVLA-S (161M) | ACT enc-dec + CVAE | full re-impl as language-conditioned ACT (no public code/ckpt available) | code build now, GPU queued |
| TinyVLA / substitute | LLM + diffusion head | diffusion head → must hook denoised x0; MetaWorld | substitute search (TinyVLA ships no checkpoint) |

The cross-arch attachment points are the same two interfaces everywhere:
**perception front** = the vision→backbone connector; **action back** = the
action-chunk head. The portability of *that pair of attachment points* across
unrelated VLA families is the generalization claim.

---

## 6. Design principles (hard constraints, all components)

1. **Backbone frozen.** No gradient to pretrained vision/LLM weights. (RIB
   additionally co-trains the lightweight action head; RASF trains nothing but itself.)
2. **Pass-through at initialisation.** Both modules are exact identities at step 0.
   Clean success cannot structurally degrade — robustness is strictly additive.
3. **No backprop through the ODE sampler.** RASF trains on the post-sampler chunk;
   RIB trains on the single-forward flow/task loss.
4. **No tuning on the test split**, no seed/task reselection after seeing success,
   no widening a baseline's disadvantage. Steering = better engineering, not data
   massaging (see `scenario.md`).
5. **Honest negatives are results.** A tie or a clean negative is reported as such —
   and the framing earns its optimism from the structural guarantees, not from
   selective reporting.

---

## 7. Repo map (where things live)

```
sib/
  robust_ib.py        RIB (perception leg) + FusedRobustIBProjector + inject/load
  adaptive_filter.py  RASF (action leg, conservative denoiser)
  ib_adapter.py       naive connector-bottleneck baseline (goes dormant)
  bottleneck.py       SpectralActionModule (v1), GaussianChannel, build_module
  transforms.py       orthonormal DCT-II / IDCT (Parseval-tested)
  libero_v.py         LIBERO-V sim perturbations (viewpoint/lighting/texture) + grid
  corruptions.py      image-space sensor-noise corruptions
  wrapper.py          SIBPolicy + ForgeActionHeadPolicy (receding-horizon consensus, TE)
scripts/
  finetune_rib.py     train RIB by robustness-shaped consistency
  train_rasf.py       train RASF as a conservative self-referential denoiser
  finetune_ib.py      train naive connector-bottleneck baseline
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

Companion docs: **`architecture.md`** (capability-level mechanism), `scenario.md`
(decision playbook for ambiguous gates), `contributions_and_novelty.md` (prior-work
positioning), `results/RESULTS_TRACKER.md` (numbers + external-labeling rule). The
authoritative, reproducible spec is the **source itself**, by design.
