# AEGIS on VLA-Adapter Pro — robustness under LIBERO-Plus

AEGIS wraps a **frozen** VLA-Adapter Pro policy with identity-initialised robustness modules and evaluates
robustness on the 7-axis [LIBERO-Plus](https://arxiv.org/abs/2510.13626) perturbation benchmark:

- **RIB** — a fused robust info-bottleneck on the vision→LLM projector (perception). Zero-init decoder ⇒
  exact pass-through at init; it learns to correct *photometric* corruption and is a **measured exact
  identity on clean inputs** (0 residual, see below).
- **RASF** — an adaptive spectral filter on the predicted action chunk (temporal action denoising).

Both are exact no-ops at initialisation, so injecting them never changes the frozen policy until trained.

## Headline result — LIBERO-Plus, 3 training seeds {42, 123, 456}

Raw success rate, paired task-bootstrap 95% CI on Δ vs the frozen base. `*` = CI excludes 0.

| Suite | base | **AEGIS** | Δ AEGIS [95% CI] | StableVLA-native | Δ StableVLA [95% CI] |
|-------|------|-----------|------------------|------------------|----------------------|
| Object | 45.7 | **51.4** | **+5.7 [+4.6, +6.9]** \* | 41.7 | −4.0 [−5.2, −2.8] \* |
| Goal | 54.6 | **57.9** | **+3.2 [+0.7, +5.8]** \* | 49.3 | −5.4 [−8.2, −2.5] \* |
| Long | 54.0 | 54.2 | +0.2 [−2.5, +2.7] | 37.8 | −16.2 [−19.4, −12.9] \* |
| Spatial | 86.8 | 87.0 | +0.2 [−1.4, +1.8] | 81.4 | −5.4 [−7.3, −3.6] \* |
| **Net** | | | **AEGIS +2.3** | | **StableVLA −7.8** |

AEGIS: two significant wins, **zero regressions**. StableVLA-native (the same fused info-bottleneck at the
same projector locus, trained with StableVLA's clean-anchor objective): significant harm on **all four**
suites. AEGIS is **+10.1 net ahead** of the competitor. See [`results/FINAL_3WAY_TABLE.txt`](results/FINAL_3WAY_TABLE.txt).

## Clean no-regression (standard LIBERO)

The perception module must not tax clean performance. Measured:

- **RIB residual on clean = 0.000000** across 1,433 forward passes → the perception module is a **perfect
  identity on in-distribution inputs** (it fires only under corruption).
- Full-pipeline clean net Δ = **−0.4** (held-out per-axis gated) / **−0.7** (ungated) — both **within the
  ±1.5 clean binomial-noise band ⇒ a statistical tie, no clean tax**. The small residual is RASF
  action-side variance, not perception.

Reporting is honest: no oracle, no best-of-seeds; base is the paired deterministic reference. The per-axis
gate is **held-out** (decide on task_id%2==0, report on the disjoint half) and openly disclosed —
gate-off equals baseline exactly. See [`results/CLEAN_VLA_REPORT.txt`](results/CLEAN_VLA_REPORT.txt) and
[`results/CLEAN_VLA_REPORT_gated_heldout.txt`](results/CLEAN_VLA_REPORT_gated_heldout.txt).

## Code (evaluation & reporting)

| File | Role |
|------|------|
| [`code/build_3way_ci.py`](code/build_3way_ci.py) | 3-way table + paired task-bootstrap 95% CIs |
| [`code/clean_report.py`](code/clean_report.py) | clean no-regression report |
| [`code/clean_gated_report.py`](code/clean_gated_report.py) | held-out per-suite gated clean report |
| [`code/gated_results.py`](code/gated_results.py) | disclosed held-out per-axis gate |

The VLA-Adapter policy itself is the upstream [OpenHelix-Team/VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter);
this directory contains the evaluation/reporting code and final results.
