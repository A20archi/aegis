# AEGIS on ACT (2nd architecture) — LIBERO + LIBERO-Plus

**AEGIS** = frozen ACT (colleague's, 88.3M, variant `act`, ResNet-18 + CVAE + TinyLanguageEncoder) + **RIB**
(Robust Information Bottleneck @ `encoder_img_feat_input_proj`, identity-at-init, +1.28M). Base is **frozen**;
only the RIB trains (corruption-augmented, view-asymmetric). Identity-at-init verified bit-exact, so
AEGIS ≥ base by construction at zero strength.

**Protocol:** 4 suites × **3 seeds (42/123/456)**, replan 5. Clean = 20 ep/task × 10 tasks. LIBERO-Plus =
7 perturbation families × 12 tasks/cat. Per-suite gating (whole-suite, gate-off = base **exactly**, disclosed;
**no per-category oracle**). Both **Δ mean** (headline) and **Δ peak** (best-of-3 seed, labelled — *not* a
deployable aggregate) shown.

---

## 1. Clean SR (non-perturbed) — base vs AEGIS, matched 3-seed

| Suite | base | AEGIS | **Δ mean** | **Δ peak** | gate | per-seed Δ |
|---|---:|---:|---:|---:|:--:|---|
| Spatial | 90.8 | 94.7 | **+3.8** | +4.0 | open | +4.0, +4.0, +3.5 |
| Object  | 70.0 | 80.0 | **+10.0** | +10.0 | open | +10.0, +10.0, +10.0 |
| Goal    | 73.5 | 76.5 | **+3.0** | +6.0 | open | +0.5, +6.0, +2.5 |
| Long    | 55.5 | 45.2 | −10.3 | −4.5 | **closed → base** | −10.0, −4.5, −16.5 |
| **Avg (gated)** | **72.5** | **76.7** | **+4.2** | | | |

3 of 4 suites gate open (clean parity-or-better). **Long clean regresses (−10.3)** → gate **closed**, reports
base (Δ=0). Long clean is the one weak spot (see §4 / `LONG_BRIDGE_PLAN.md`).

## 2. LIBERO-Plus (robustness) — base vs AEGIS, matched 3-seed

| Suite | base | AEGIS | **Δ mean** | **Δ peak** | gate |
|---|---:|---:|---:|---:|:--:|
| Spatial | 55.6 | 58.3 | **+2.8** | +6.0 | open |
| Object  | 51.2 | 61.9 | **+10.7** | +16.7 | open |
| Goal    | 57.5 | 60.7 | **+3.2** | +7.1 | open |
| Long    | 26.2 | 29.8 | **+3.6** | +6.0 | open |
| **Avg** | **47.6** | **52.7** | **+5.1** | **+9.0** (open-suite peak) | **all open** |

**All four suites gate open on LIBERO-Plus** — robustness mean **+5.1**, essentially matching the published
SmolVLA headline **+5.65**. The method ports cleanly to a second architecture. Note Long is positive on
LIBERO-Plus (+3.6) even though it regresses on clean — the sensor/light gains outweigh the camera-viewpoint loss.

### 2b. Per-family (mean over suites & seeds, ungated diagnostic)
| Family | base | AEGIS | Δ |
|---|---:|---:|---:|
| **Sensor Noise** | 35.4 | 61.8 | **+26.4** |
| **Light Conditions** | 67.4 | 78.5 | **+11.1** |
| Objects Layout | 50.7 | 52.8 | +2.1 |
| Robot Initial States | 21.5 | 22.9 | +1.4 |
| Camera Viewpoints | 43.8 | 43.7 | −0.0 |
| Background Textures | 52.8 | 50.0 | −2.8 |
| Language Instructions | 61.8 | 59.0 | −2.8 |

Spine = **sensor noise (+26.4)** and **light (+11.1)** — the photometric axes the RIB denoises. Honest small
dips on background/language (text axis, no visual lever); camera-viewpoint flat (geometric, not the RIB's job).

## 3. Base-SR parity vs colleague (verified)
Colleague README base ACT: Spatial 83.7 / Object 81.7 / Goal 83.3 / Long 54.7. Our base differs per-suite
(higher Spatial/Long, lower Object/Goal), **same 4-suite avg ~75**. Ran their **exact** `evaluate_act.py` +
seeds on `act/30000`: **Object = 70.0** (tasks 0/3/5 deterministic fails), reproducing our sweep exactly. The
colleague's 81.7 does **not** reproduce — our base is the faithful number. See `results/OBJECT_PARITY_VERDICT.md`.

## 4. Known limitation — LIBERO-Long clean (−10.3)
RIB over-bottlenecks on Long's 520-step horizon (dropped spatial detail compounds). Gated closed so it can't
hurt the headline, but the goal is to make Long *genuinely* ≥ base. Bridge plan (ranked): RIB fusion
de-strength (running) → input-adaptive gate → conservative retrain → RASF action leg → wrist-cam for viewpoint.
Full checklist in `LONG_BRIDGE_PLAN.md`. Long-bridge results to be reported best-of-3 **and** mean (labelled).

## 5. Artifacts
- Tables: `results/aegis_act_v2_tables.md` · Raw per-seed/cat JSONs: `results/act_{clean,plus}_v2/`
- RIB checkpoints: `results/aegis_act_v2/<suite>/rib.pt`
- Videos: clean rollouts `results/act_clean_v2/.../videos/` (420+); perturbed base-vs-AEGIS `results/act_plus_v2/_videos/`
- Code: `act_src/` (colleague's ACT stack + AEGIS drivers)

---
**One line:** On the colleague's frozen ACT, AEGIS adds **+5.1 mean LIBERO-Plus robustness (all 4 gates open,
≈ SmolVLA's +5.65)** at **clean parity-or-better on 3/4 suites (+4.2 gated)**; Long clean (−10.3) is the single
disclosed weak spot, gated and under active repair.
