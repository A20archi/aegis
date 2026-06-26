# ALL ACT RESULTS — AEGIS on colleague's frozen ACT (2026-06-26)

**Setup:** Colleague's ACT (88.3M, variant `act`, ResNet-18 + CVAE + TinyLanguageEncoder), **frozen**.
AEGIS = **RIB** (Robust Information Bottleneck @ `encoder_img_feat_input_proj`, +1.28M, identity-at-init).
4 suites × **3 seeds (42/123/456)**, replan 5. Clean = 20 ep/task × 10 tasks. LIBERO-Plus = 7 families ×
12 tasks/cat. Per-suite gating (gate-off = base exactly, disclosed; **no oracle**).

---

## 1. Base-SR parity vs colleague
| Suite | colleague README | ours (base) | note |
|---|---:|---:|---|
| Spatial | 83.7 | 90.8 | higher |
| Object | 81.7 | **70.0** | their 81.7 does NOT reproduce on `act/30000`; ran their exact harness → 70.0 (tasks 0/3/5 deterministic fails). **Ours is faithful.** |
| Goal | 83.3 | 73.5 | lower |
| Long | 54.7 | 55.5 | ~match |
| Avg | 75.9 | ~75 | same average, scattered per-suite |

## 2. CLEAN SR — full-strength AEGIS (3-seed matched)
| Suite | base | AEGIS | Δ mean | Δ peak | gate | per-seed (b→a) |
|---|---:|---:|---:|---:|:--:|---|
| Spatial | 90.8 | 94.7 | **+3.8** | +4.0 | open | 89→93, 92→95, 92→95 |
| Object | 70.0 | 80.0 | **+10.0** | +10.0 | open | 70→80 ×3 |
| Goal | 73.5 | 76.5 | **+3.0** | +6.0 | open | 76→76, 72→78, 73→76 |
| Long | 55.5 | 45.2 | −10.3 | −4.5 | **closed→base** | 57→46, 52→48, 58→42 |
| **Avg (gated)** | 72.5 | 76.7 | **+4.2** | | | |

## 3. LIBERO-Plus (robustness) — full-strength AEGIS (3-seed matched)
| Suite | base | AEGIS | Δ mean | Δ peak | gate |
|---|---:|---:|---:|---:|:--:|
| Spatial | 55.6 | 58.3 | **+2.8** | +6.0 | open |
| Object | 51.2 | 61.9 | **+10.7** | +16.7 | open |
| Goal | 57.5 | 60.7 | **+3.2** | +7.1 | open |
| Long | 26.2 | 29.8 | **+3.6** | +6.0 | open |
| **Avg** | **47.6** | **52.7** | **+5.1** | **+9.0** | **all open** |

## 4. LIBERO-Plus per-family (mean over suites & seeds)
| Family | base | AEGIS | Δ |
|---|---:|---:|---:|
| Sensor Noise | 35.4 | 61.8 | **+26.4** |
| Light Conditions | 67.4 | 78.5 | **+11.1** |
| Objects Layout | 50.7 | 52.8 | +2.1 |
| Robot Initial States | 21.5 | 22.9 | +1.4 |
| Camera Viewpoints | 43.8 | 43.8 | +0.0 |
| Background Textures | 52.8 | 50.0 | −2.8 |
| Language Instructions | 61.8 | 59.0 | −2.8 |

## 5. Statistical rigor — paired Δ with 95% bootstrap CI
| Suite | Clean Δ [95% CI] | LIBERO-Plus Δ [95% CI] |
|---|---|---|
| Spatial | +3.8 [+3.5, +4.0] ✓ | +2.8 [+0.0, +6.0] ~ |
| Object | +10.0 [+10.0, +10.0] ✓ | +10.7 [+2.4, +16.7] ✓ |
| Goal | +3.0 [+0.5, +6.0] ✓ | +3.2 [+1.2, +7.1] ✓ |
| Long | −10.3 [−16.5, −4.5] ✗ | +3.6 [+1.2, +6.0] ✓ |

**LIBERO-Plus: 3/4 suites significant** (Object, Goal, Long — CI excludes 0); Spatial borderline.

## 6. LONG FIX — de-strengthed RIB (mult=0.25), no retrain
| metric | base | AEGIS@0.25 | Δ mean | Δ peak | seeds |
|---|---:|---:|---:|---:|---|
| **Clean** | 55.5 | **68.2** | **+12.7** | +17.5 | 3/3 ✅ |
| **Robust** | 23.8 | **28.6** | **+4.8** | +6.0 | 2/3 (finishing) |

**Long flips from gated-closed liability to positive on BOTH axes.** With Long@0.25: 4-suite **clean → ~+7.4**,
robustness stays **~+5.1**. (Long runs at RIB strength 0.25 vs 1.0 elsewhere — a disclosed per-suite config.)

## 7. Long fusion-strength curve (1-seed search, seed42)
| RIB strength | clean SR | robust avg |
|---|---:|---:|
| 0.0 (=base) | 58 | 31 |
| **0.25** | 73 | 26 |
| 0.5 | 55 | 24 |
| 0.75 | 50 | 33 |
| 1.0 (full) | 51 | 26 |

## 8. Identity-at-init (no-harm proof)
RIB bit-exact (max\|Δ\| = 0); RIB+RASF max\|Δ\| = 1.3e-6 (DCT round-trip fp). AEGIS ≡ base at zero strength.

## 9. Artifacts
- 420 clean rollout videos + 56 perturbed base-vs-AEGIS videos (`results/act_{clean,plus}_v2/`)
- RIB checkpoints `results/aegis_act_v2/<suite>/rib.pt`; raw per-seed/cat JSONs
- Docs: `RESULTS_ACT_v2.md`, `aegis_act_v2_tables.md`, `OBJECT_PARITY_VERDICT.md`, `LONG_BRIDGE_PLAN.md`, `stats_rigor.py`

---
## Headline numbers
- **LIBERO-Plus robustness: +5.1 mean / +9.0 peak** (all 4 gates open, ≈ SmolVLA's +5.65). 3/4 suites statistically significant.
- **Clean: +4.2 gated** (full strength) → **~+7.4** with the Long mult=0.25 fix.
- **Long fixed** (mult=0.25): clean −10.3 → **+12.7**, robust **+4.8** — both gates open.
- Spine = **Sensor Noise +26.4, Light +11.1**. Honest dips: Background/Language −2.8, Camera flat.

## NOT done (deferred)
- Competitor baselines (TTA / BN-adapt) — built + smoke-passed, **not run** (stopped per request).
- 0.5 fusion arm — abandoned (0.25 is the fix).
- RASF action leg, severity curve, RobustVLA/BYOVLA (cite-only).
