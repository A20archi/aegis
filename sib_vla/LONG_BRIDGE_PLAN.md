# LONG_BRIDGE_PLAN — closing the LIBERO-Long gap (AEGIS-on-ACT)

**Date:** 2026-06-26 · **Context:** colleague's frozen ACT (88.3M, variant `act`). 3 suites strong
(Spatial/Object/Goal: clean +3.8/+10.0/+3.0, LIBERO-Plus +2.8/+10.7/+3.2). **Long is the only weak suite.**

## Diagnosis (from the v2 sweep)
| Symptom | Number | Cause |
|---|---|---|
| Long **clean** regression | base 56.5 → AEGIS 46.5 (**−10**, seed42 complete) | RIB over-bottlenecks; 520-step horizon compounds dropped spatial detail. fusion +0.581 (strong). |
| Long **Camera Viewpoints** | base 29.2 → AEGIS 4.2 (**−25**) | RIB is a *photometric denoiser*, not geometric-invariance; strips spatial cues under viewpoint shift. |
| Long **Sensor Noise** | base 16.7 → AEGIS 50.0 (**+33**) | RIB works as intended on photometric axes — the leg isn't broken, it's mis-scoped for Long. |

Both failures = "RIB removes info Long needs." Fix = keep more info on clean + don't touch geometry.

## Methodologies (ranked)

### Tier 1 — no retrain, RUNNING TONIGHT
- [ ] **RIB fusion-strength sweep** (`run_long_fusion_sweep.sh`, mult ∈ {0,0.25,0.5,0.75,1.0}; mult=0 ≡ base).
      Eval flag added: `--fusion-mult` in `clean_eval_aegis.py`/`plus_eval_aegis.py` + `load_aegis_policy`.
      **Goal:** smallest mult with **clean ≥ base** AND robustness retained. Expect ~0.25–0.5.
      Output: `results/long_bridge/{clean,plus}/m*/`.

### Tier 2 — the real fix (one retrain each, ~half day)
- [ ] **Input-adaptive RIB gate** (AEGIS-v2, see `project_aegis_v2_blackwell`): gate engages only when
      corruption detected (high-freq feature energy); clean Long → gate≈0 → **identity, zero clean regression
      by construction**. The principled cure for the −10.
- [ ] **Conservative Long-RIB retrain**: β 1e-3→1e-4 (keep more info), raise free-bits, `corrupt_frac`
      0.6→0.4 (more clean samples protect clean), and try the **step-3000** ckpt (6000 may over-train).

### Tier 3 — action leg + viewpoint (~1 day)
- [ ] **RASF on Long** (action leg, not yet exercised on ACT): `SpectralActionModule` + `SIBPolicy` are
      wired; train/enable on Long to smooth the T=100 chunk → stop long-horizon error compounding.
- [ ] **Wrist-camera emphasis for viewpoint**: agentview takes the perturbation, wrist is stable;
      up-weight wrist features (or spatial-transformer canonicalize agentview) to fix Camera −25.
- [ ] **Temporal ensembling**: replan 5-of-100 → heavy overlap; average overlapping chunk predictions.

## Order of attack tomorrow
1. Read tonight's fusion-sweep curve → pick mult (free win if clean recovers).
2. If mult alone insufficient → adaptive gate (Tier 2 #1) as the headline fix.
3. Add RASF (Tier 3 #1) for the long-horizon action smoothing.
4. Camera −25 is the last mile (wrist emphasis) — only if targeting that axis specifically.

## Reporting convention for the Long bridge (user directive 2026-06-26)
Run each methodology at **3 seeds (42/123/456)**, report the **best-of-3 seed** as the Long headline —
but **LABELLED "best-of-3 (Δpeak)"** with the 3-seed **mean shown alongside**. This matches our existing
Δpeak column; the only rule (from the retired silent-concealment standard) is *never present best-of-3 as a
silent mean* — keep it labelled so a reviewer sees both. Best-of-3 is the method's ceiling, not a deployable
aggregate.

Tonight's fusion sweep stays **1-seed (search only)** to find the strength curve cheaply; the *chosen* config
+ Tier-2/3 methods get the full **3-seed, best-of-3-labelled** treatment tomorrow.

## Honest framing (unchanged)
Per-suite gating already protects the headline: if Long stays below base it gates closed (Δ=0), never
negative. The bridge work is to make Long *genuinely* ≥ base, not to mask it.
