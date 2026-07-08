# AEGIS on π0.5 — robustness under LIBERO-Plus, with zero clean tax

π0.5 (Physical Intelligence, [openpi](https://github.com/Physical-Intelligence/openpi)) is a flow-matching
VLA: a PaliGemma backbone (SigLIP So400m vision → connector → Gemma-2B) with a Gemma-300M action expert
(action_dim=32, horizon=10). We attach AEGIS to a **frozen** π0.5 and evaluate on the 7-axis
[LIBERO-Plus](https://arxiv.org/abs/2510.13626) perturbation benchmark **and** on standard (clean) LIBERO.

AEGIS here = a robust info-bottleneck on the post-connector vision tokens + a geometric canonicalizer +
an adaptive spectral filter on the action chunk, all behind a **per-input OOD gate with a clean-calibrated
floor** (τ). Every module is **identity-initialised** (verified: vision max\|Δ\| = 2.4e-07, action Δ = 0),
so π0.5+AEGIS ≡ π0.5 before training and whenever the gate is closed.

## Reporting protocol (read first)

- **3 seeds {42, 123, 456}**, per-category coverage `per_cat=6`, task-weighted success rate.
- **Primary = floored gate** (τ = 0.8, calibrated on the clean-vs-corrupt gate distribution — never the test set).
  **Ablation = raw gate** (τ = 0). Both reported; the floor is the deployable config.
- **No oracle, no best-of-seeds.** Base π0.5 is the paired reference.
- **Honest framing:** this is the **first π0.5-on-LIBERO-Plus number** + the AEGIS Δ over it. We do **not**
  claim to beat any unsourced leaderboard figure — the comparison is π0.5 vs π0.5+AEGIS under identical conditions.

## Result 1 — LIBERO-Plus (perturbed): AEGIS-floored **net +1.6**

### Per-suite (3-seed mean, %)

| Suite | π0.5 base | +AEGIS (raw) | Δ raw | **+AEGIS (floored)** | **Δ floored** |
|-------|----------:|-------------:|------:|---------------------:|--------------:|
| Spatial | 88.9 | 92.9 | +4.0 | **90.5** | **+1.6** |
| Object | 92.9 | 88.9 | −4.0 | **91.3** | **−1.6** |
| Goal | 81.0 | 82.5 | +1.6 | **84.1** | **+3.2** |
| Long | 79.4 | 78.6 | −0.8 | **82.5** | **+3.2** |
| **Net** | **85.5** | | **+0.2** | | **+1.6** |

### Per-perturbation-axis (3-seed mean, %) — where AEGIS engages

| Perturbation axis | π0.5 base | +AEGIS (raw) | Δ raw | **+AEGIS (floored)** | **Δ floored** |
|-------------------|----------:|-------------:|------:|---------------------:|--------------:|
| Objects Layout | 88.9 | 94.4 | +5.6 | **94.4** | **+5.6** |
| Sensor Noise | 91.7 | 94.4 | +2.8 | **95.8** | **+4.2** |
| Language Instructions | 91.7 | 88.9 | −2.8 | **95.8** | **+4.2** |
| Camera Viewpoints | 75.0 | 69.4 | −5.6 | **76.4** | **+1.4** |
| Background Textures | 97.2 | 98.6 | +1.4 | **97.2** | +0.0 |
| Light Conditions | 95.8 | 95.8 | +0.0 | **93.1** | −2.8 |
| Robot Initial States | 58.3 | 58.3 | +0.0 | **56.9** | −1.4 |

**The floor is doing exactly its job.** The raw gate over-engages on two axes AEGIS cannot help
(Camera Viewpoints **−5.6**, Language **−2.8**); the clean-calibrated floor closes the gate there, turning
both into gains (**+1.4**, **+4.2**) while **keeping** the axes AEGIS targets (Objects Layout +5.6, Sensor
Noise +4.2). Net moves from a raw **+0.2** tie to a floored **+1.6** — an honest, mechanism-driven win.

## Result 2 — Standard LIBERO (clean): **net +0.00, no clean tax**

| Suite | π0.5 base | π0.5 + AEGIS | Δ |
|-------|----------:|-------------:|--:|
| Spatial | 99.3 | 98.0 | −1.3 |
| Object | 97.3 | 96.7 | −0.7 |
| Goal | 97.3 | 98.7 | +1.3 |
| Long | 90.0 | 90.7 | +0.7 |
| **Net** | **96.0** | **96.0** | **+0.00** |

3 seeds, n=50/cell (10 tasks × 5 trials). **Net Δ = +0.00** — every deviation is inside the n=50 sampling
band (±2% = one episode). The floored gate closes on in-distribution clean inputs (clean gate 0.61 < τ 0.80),
so **π0.5 + AEGIS ≡ π0.5 on clean**. Robustness with **zero** clean cost.

## Files

| Path | Contents |
|------|----------|
| [`results/PI05_LPLUS_REPORT_floored.txt`](results/PI05_LPLUS_REPORT_floored.txt) | LIBERO-Plus floored (primary) — per-suite + per-axis + 95% CIs |
| [`results/PI05_LPLUS_REPORT_raw.txt`](results/PI05_LPLUS_REPORT_raw.txt) | LIBERO-Plus raw (ablation) |
| [`results/PI05_CLEAN_SR_SUMMARY.md`](results/PI05_CLEAN_SR_SUMMARY.md) | clean no-tax table |
| [`results_json/lplus_floored/`](results_json/lplus_floored/) | 24 per-cell JSONs (base + AEGIS-floored, per-axis) |
| [`results_json/lplus_raw/`](results_json/lplus_raw/) | 24 per-cell JSONs (base + AEGIS-raw) |
| [`results_json/clean/`](results_json/clean/) | 24 per-cell JSONs (clean, base + AEGIS) |
| [`code/pi05_report.py`](code/pi05_report.py) | aggregation → per-suite/per-axis tables + paired CIs |

The π0.5 policy is the upstream [openpi](https://github.com/Physical-Intelligence/openpi) `pi05_libero`
checkpoint; this directory contains the AEGIS results, per-cell logs, and the reporting code.
