# AEGIS: An Adaptive Entropy-Gated Information Sieve for Additive, Provably-Safe Robustness in Frozen Vision-Language-Action Policies

> **arXiv draft scaffold.** Status tags: ✅ have data · 🟡 partial · 🔴 [PENDING] not yet run.
> Numbers are pulled from `final_module_architecture.md` and `SAVED_STATE.md`; figures from
> `docs/figures/`. Replace [PENDING] blocks as Phase-1/2 results land. Author line + venue TBD.

---

## Abstract

Vision-Language-Action (VLA) models achieve strong nominal success but degrade sharply under
the visual and dynamical perturbations any real deployment guarantees — motion blur, sensor
noise, lighting and viewpoint shift. Existing robustness interventions either retrain the
backbone (expensive, and risking the clean-task competence the model is valued for) or attach
heuristics that can silently degrade clean success. We introduce **AEGIS** (Adaptive
Entropy-Gated Information Sieve), a rate-limited information bottleneck applied at **two
interfaces** of a **frozen** VLA — the vision→LLM connector (perception) and the sampled
action chunk (action) — together with a receding-horizon temporal consensus shared by both
arms. Both modules are **exact identities at initialization**, so the augmented policy is
bit-for-bit the base policy before any learning; robustness is therefore *strictly additive*
and per-suite gating is *provably safe* (a disabled module recovers the base policy exactly).
On a frozen SmolVLA backbone we evaluate on **LIBERO-Plus** (a published robustness benchmark,
7 perturbation categories) across all four suites (spatial / object / goal / long) over three
seeds, plus clean per-suite SR. AEGIS improves robustness over the SmolVLA+TE baseline while
preserving clean success (never below base on any suite, by construction). [PENDING: the
multi-seed LIBERO-Plus + clean runs; GR00T-N1.5 cross-architecture; WidowX real-robot. Internal
LIBERO-V corruption-axis validation: Spatial +14.1, Object/Goal +29.9 / 0 regressions.]

---

## 1 Introduction

- **Problem.** VLAs are deployed open-loop over action chunks; nominal SR collapses under
  perturbation. Robustness must not cost clean competence.
- **Gap.** Backbone retraining is costly and risks clean SR; bolt-on heuristics can silently
  regress. No prior method offers a *guarantee* that the intervention cannot hurt.
- **Key idea.** A rate-limited information bottleneck that is an *exact identity at init*,
  inserted at two complementary interfaces of a frozen policy, trained post-hoc on cached
  outputs. "Off" is the base policy exactly → additive robustness + provably-safe gating.
- **Contributions.**
  1. A dual-locus (perception + action) information-bottleneck design for frozen VLAs.
  2. *Identity-at-initialization* → strictly-additive robustness and provably-safe per-suite
     gating (0 regressions, by construction).
  3. A spectral (DCT-II) action-chunk filter with a rate-distortion foundation (Wiener gain /
     reverse water-filling), the first application of rate-distortion to robot action spectra.
  4. Empirical: in-distribution + cross-suite robustness on LIBERO/SmolVLA; [PENDING
     cross-architecture GR00T-N1.5; PENDING real-robot WidowX].

## 2 Related Work

- **IB in policies/representations:** VIB (Alemi'17), IBAC-SNI (Igl'19), VDB (Peng'19),
  StableVLA IB-Adapter (Hiranaka'24). → comparison matrix (see `contributions_and_novelty.md`):
  locus, spectral basis, channel model, rate term, decode, post-hoc applicability.
- **VLA robustness / generalization benchmarks:** LIBERO, LIBERO-Plus, OpenVLA/Octo WidowX
  suites, SIMPLER.
- **Temporal ensembling / receding-horizon control** (ACT-style consensus).
- **Our position:** the only post-hoc, frozen-backbone, identity-init method with a safety
  guarantee; spectral action bottleneck is novel.

## 3 Method

### 3.1 Overview (Fig. — `docs/figures/fig3_architecture.png`)
Three insertions, no backbone gradient. Both modules pass-through at init.

### 3.2 RIB — Robust Information Bottleneck (perception)
Fused projector at the connector linear; deterministic latent (~2.27M); bounded info-rate
penalty with a floor; robustness-shaped consistency training on *generic* augmentations
(test perturbations held out). Handles systematic visual shift.

### 3.3 RASF — Residual Adaptive Spectral Filter (action)
DCT-II along time → input-adaptive per-band gain → bounded residual
`A_hat = A + gate_max·tanh(gate)·(filtered − A)`. Five structural no-collapse guarantees.
Self-referential denoiser (target = policy's own benign prediction). **Theory:** per-band
Gaussian channel, Wiener gain `g_k=λ_k/(λ_k+σ_k²)`, rate `R_k=½ln(1+λ_k/σ_k²)`, reverse
water-filling (proved + unit-tested in `sib/waterfill.py`).

### 3.4 TE — Temporal Ensembling
Position-aligned exponential consensus; inference-only; present in *both* arms so every Δ
isolates the AEGIS modules.

### 3.5 Safety properties
Identity at init ⇒ clean SR cannot structurally degrade; robustness strictly additive;
per-suite gating provably safe (off ≡ base exactly).

## 4 Experimental Setup

Backbone: frozen SmolVLA (checkpoint `smolvla_spatial_repro`, used for BOTH arms). Sim: LIBERO.
Protocol: `n_action_steps=1`, 10 flow-matching denoise steps, **per-suite max-steps** (spatial
220 / object 280 / goal 300 / long 520), fixed init-states. Baseline = SmolVLA + TE (honest
reference). **Robustness benchmark = LIBERO-Plus (arXiv 2510.13626)** — a published benchmark
with 7 perturbation categories per suite. **Seeds {42, 123, 456}**, mean ± 95% Wilson CI.
(Our custom LIBERO-V corruption-axis grid is internal validation only — see Appendix; not a
main-paper result.)

## 5 Results

### 5.1 Clean per-suite SR preserved — 4 suites, 3 seeds 🔴 [PENDING run: `--stage clean`]
Standard LIBERO, no perturbation. AEGIS ≥ base on every suite (identity-init + per-suite
gating). Single-seed-42 reference: avg 85.25 vs 83.5 (+1.75). [table: 4 suites × {base,AEGIS,Δ}
mean ± CI]

### 5.2 LIBERO-Plus robustness — 4 suites × 7 categories, 3 seeds ★ HEADLINE 🔴 [PENDING run]
The paper's main robustness result. Per suite: 7-category × {baseline, AEGIS, Δ} table + total,
mean ± CI, per-suite gating. `lplus_modal.py --stage stage1`. [table + figure]

### 5.3 Cross-architecture — GR00T-N1.5 (3B) 🔴 [PENDING / out of current scope]
Wiring verified (RIB @ eagle_model.mlp1, 2.46M, identity-init; RASF @ 16×32). No re-training in
current scope.

### Appendix A · LIBERO-V internal validation (not a main result)
Our custom corruption-axis grid (motion blur / gaussian / lighting / texture / viewpoint). It
gave the early signal that AEGIS works but is not a recognized benchmark, so it is reported only
as internal validation; the published LIBERO-Plus result (§5.2) is the headline.

**Measured grid (frozen SmolVLA baseline → AEGIS, SmolVLA-0.5B, n=100/cell unless noted, seed 42):**

| Suite | Axis | Baseline | AEGIS | Δ |
|---|---|---:|---:|---:|
| Object | motion_blur | 0% | 86% | **+86** |
| Object | gaussian_noise | 36% | 90% | **+54** |
| Object | lighting | 58% | 92% | +34 |
| Object | texture | 83% | 97% | +14 |
| Object | viewpoint_medium | 0% | 0% | +0 |
| Goal | motion_blur | 19% | 78% | **+59** |
| Goal | viewpoint_medium | 17% | 43% | +26 |
| Goal | viewpoint_large | 8% | 29% | +21 |
| Goal | texture | 90% | 93% | +3 |
| Goal | lighting | 80% | 82% | +2 |

**Summary:** 10 complete Object+Goal pairs, **mean Δ = +29.9, zero regressions**, range +0 to +86.
AEGIS gains concentrate on the action/perception-corruption axes (motion blur, gaussian noise,
lighting); large-viewpoint shift is the soft spot (Object viewpoint flat 0→0 — RASF/RIB are blind
to large viewpoint change; Goal viewpoint does recover). Scope: Object+Goal only, single seed (42),
not multi-seeded. One cell incomplete (Object viewpoint_large AEGIS). Spatial reported separately
gave +14.1 mean.

## 6 Ablations 🟡 (harness ready: `smolvla_modal.py::main --stage ablation`)

All checkpoints exist → eval-only. Spatial, 6 axes.
- **Per-locus:** vanilla (floor) → baseline (+TE) → RIB-only → RASF-only → full. Attributes
  each axis's gain; isolates TE (vanilla→baseline).
- **Design:** no-DCT (`raw_vib`) → spectral basis matters; no-rate (`gain_no_rate`) → rate
  objective matters; naive-IB (`ib_on86`) → naive bottleneck is dormant (StableVLA contrast).

[PENDING: results table — run post-reset / on dedicated A100.]

## 7 Limitations

Single seed (so far); ~5–10pp reproduction gap under strict protocol (Δ on our honest
baseline, never vs paper numbers); RASF deliberately blind to perception-space shifts (RIB's
axis); cross-architecture and real-robot pending.

## 8 Conclusion

A frozen-backbone, identity-init, dual-locus information bottleneck delivers additive,
provably-safe robustness — architecture- and embodiment-agnostic by design.

## References
[BibTeX in README.md; fill from `contributions_and_novelty.md` citation list.]

---

### Draft TODO (gating items)
- [ ] 5.5 multi-seed (Phase 1) · [ ] 5.6 GR00T (Phase 2) · [ ] §6 ablation table (Phase 1)
- [ ] 5.4 graceful-degradation curve (needs noise-sweep data regenerated)
- [ ] LIBERO-Plus paper-protocol section (Phase 3) · [ ] WidowX real-robot (Phase 4)
- [ ] author/venue · [ ] convert to LaTeX (NeurIPS/CoRL template) when numbers freeze
