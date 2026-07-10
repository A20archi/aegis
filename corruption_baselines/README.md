# AEGIS vs test-time baselines — ImageNet-C robustness

A second corruption benchmark for AEGIS, **disjoint from LIBERO-Plus**: standard
[ImageNet-C](https://arxiv.org/abs/1903.12261) corruptions applied to LIBERO, on a **frozen ACT-40**
host (the perception leg, RIB only). It answers the reviewer question *"why not just test-time
adaptation?"* and stress-tests the severity axis.

## What applies, and what does not

The two standard **no-train** robustness controls are BN-adapt and TTA.

- **BN-adapt is structurally inapplicable to the VLA family.** It re-estimates BatchNorm statistics
  on the corrupted stream, but modern VLA vision encoders carry **zero BatchNorm layers** — SigLIP,
  Eagle, and even the lerobot ACT backbone all use LayerNorm/GroupNorm (verified: 0 `BatchNorm2d`
  modules). A method that presumes a CNN+BN backbone cannot travel across these architectures; AEGIS
  attaches at the connector regardless of normalisation scheme.
- We therefore benchmark against the applicable no-train control, **TTA** (4-view augment-and-average).

## Result — 36 arm-conditions (3 corruptions × 2 severities × 2 suites × 3 seeds)

Success rate (%); Δ vs frozen base with paired task-bootstrap 95% CI. `*` = CI excludes 0.

| Corruption | Sev | Suite | base | +TTA | +AEGIS | Δ(AEGIS−base) [95% CI] |
|---|---|---|---|---|---|---|
| Gaussian noise | 3 | Spatial | 49 | 50 | 59 | +9.3 [−2.7, +22.0] |
| Gaussian noise | 3 | Object | 30 | 32 | 50 | +20.0 [0.0, +40.0] |
| **Gaussian noise** | **5** | **Spatial** | 9 | 11 | 46 | **+37.3 [+24.0, +52.0]** \* |
| **Gaussian noise** | **5** | **Object** | 0 | 4 | 50 | **+50.0 [+33.3, +66.7]** \* |
| Motion blur | 3 | Spatial | 67 | 66 | 78 | +10.7 [−4.7, +26.0] |
| Motion blur | 3 | Object | 60 | 26 | 50 | −10.0 [−23.3, 0.0] |
| **Motion blur** | **5** | **Spatial** | 41 | 39 | 67 | **+26.0 [+9.3, +42.7]** \* |
| **Motion blur** | **5** | **Object** | 0 | 2 | 20 | **+20.0 [+6.7, +33.3]** \* |
| Fog | 3 | Spatial | 66 | 63 | 71 | +5.3 [−1.3, +12.7] |
| Fog | 3 | Object | 30 | 50 | 50 | +20.0 [−3.3, +40.0] |
| **Fog** | **5** | **Spatial** | 58 | 59 | 82 | **+24.0 [+10.7, +38.0]** \* |
| Fog | 5 | Object | 30 | 20 | 30 | +0.0 [−23.3, +23.3] |
| **Net (36 cond., 3 seeds)** | | | | | | **+17.7 [+12.8, +22.7]** \* |

**Headline: net +17.7 over base [+12.8, +22.7] and +19.2 over TTA [+14.4, +24.1]** — both CI-separated
from zero.

## Reading it honestly

- **AEGIS matches or beats TTA on every one of the 12 settings** (strictly wins 11; one tie on
  Fog/S3/Object). TTA barely moves the base and is worthless under severe corruption.
- **The gain peaks where the base collapses.** At severity-5 Gaussian noise the frozen ACT drops to
  0–9% and AEGIS recovers it to 46–50% (+37 to +50, CI-separated), while TTA stalls at 4–11%.
- **The one cell where AEGIS dips below base** is motion-blur/S3 on the cluttered Object suite
  (−10, CI touches zero) — outside RIB's noise/photometric axis. We leave it in rather than gate it
  away; note TTA fares far worse there (−34 vs base).

Reporting is honest: no oracle, no best-of-seeds; base is the paired deterministic reference; every Δ
carries a paired task-bootstrap CI.

## Files

| File | Role |
|------|------|
| [`results/RESULTS_TABLE.txt`](results/RESULTS_TABLE.txt) | full 12-condition table + net |
| [`results/corruption_results.json`](results/corruption_results.json) | machine-readable per-condition + net with CIs |
| [`code/build_tta_table.py`](code/build_tta_table.py) | reporting script (paired task-bootstrap CIs from per-cell success arrays) |

Consistent with this repo's disclosure policy, this directory publishes the **results and the
reporting script**; the ACT host is the perception leg (RIB) — see the root README for the AEGIS
architecture. Raw per-cell rollouts available on request.
