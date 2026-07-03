# RIGOR_SUMMARY — honesty audit trail

Plain-English ledger. Each headline claim maps to the exact measured number, the
file/command that produced it, and a label: **PROVEN-IDEALIZED** (mathematically
exact for the idealized model) vs **DEPLOYED-MEASURED** (an actual measurement of
the deployed/eval system).

Deployment reality, stated once so nothing below blurs it:
- Default **RASF** = a *learned* per-band sigmoid gain trained with a 4-term
  denoising MSE toward a task target A* ≠ A.
- **Wiener/MMSE decode** = a Track-B variant (closed-form), not the default.
- **RIB** (perception, vision→LLM connector, `sib/robust_ib.py`) = a
  deterministic-L2 / rate-penalised bottleneck, **not** a variational-KL channel,
  despite the historical "VIB" naming. The water-filling theorems do NOT apply to it.

---

## 1. θ = β/2 water-filling theorem

- **Claim:** the β-penalised Gaussian compression channel is exactly reverse
  water-filling at level θ = β/2; active bands carry constant distortion D_k = β/2.
- **Number:** max |D_k − β/2| = **5.135e-16** over β∈{0.05,0.2,0.5,1.0,2.0,3.0,5.0}
  on bands λ=(0.1…4.0); active-band counts 40/39/38/35/30/25/15; dropped bands
  keep D_k=λ_k (atol 1e-9); rate↔level inversion max|err| = **3.553e-15**.
- **Source:** `scripts/verify_theta_beta2.py`; unit suite `tests/test_waterfill.py`
  reports **6 passed** (incl. `test_beta_equals_two_theta_theorem`).
- **Label:** **PROVEN-IDEALIZED** (compression objective, reconstruct source A,
  Wiener decode). This is the analytic ANCHOR, not the deployed default RASF.

## 2. Wiener/MMSE decode optimality + operational rate–distortion identity

- **Claim:** the scalar Wiener gain g_k = λ_k/(λ_k+σ_k²) is the global MMSE
  decoder (over all measurable estimators); its MSE is the posterior variance
  D_k = λ_k σ_k²/(λ_k+σ_k²); and R_k = ½ln(1+λ_k/σ_k²) = ½ln(λ_k/D_k) — so the
  water-filling rate is *operational* (achieved, not merely a lower bound).
- **Number:** identity terms are algebraic; anchored by the same
  max|D_k−β/2|=5.135e-16 / max|err|=3.553e-15 numerics as claim 1.
- **Source:** proof (Theorem 2); `scripts/verify_theta_beta2.py`;
  `tests/test_waterfill.py` (6 passed).
- **Label:** **PROVEN-IDEALIZED**. Referee note: Step-6 minimizer argument uses
  unique-critical-point + boundary behaviour (J is NOT globally convex in s;
  strictly convex in the gain g). Deployed default is still the learned sigmoid gain.

## 3. RASF = deterministic-L2 bottleneck on the action chunk; identity-at-init

- **Claim:** RASF is the ACTION-side DCT spectral module (`sib`/`raw_vib`,
  `sib/transforms.py`); θ=β/2 and the allocation are ACTION-side. Identity-at-init
  holds two DIFFERENT ways: idealized RASF via σ²→0 (g_k=1 ⇒ Â=A); the deployed
  perception RIB (`sib/robust_ib.py`) via zero-init decoder + tanh(fusion_coeff)
  residual fuse + per-sample gate (out = z_mlp at init).
- **Number:** algebraic identity; L2 rate proxy is `z.pow(2).mean()`
  (`robust_ib.py` L77); residual fuse L88–90, L169; zero-init L57–58.
- **Source:** Theorem 3; `sib/robust_ib.py`, `sib/transforms.py`.
- **Label:** **PROVEN-IDEALIZED** for the algebra; the locus/naming (action-side
  RASF vs perception-side RIB) is a **DEPLOYED** code fact. RIB is L2, not KL.

## 4. Learned allocation matches water-filling SHAPE (Track A)

- **Claim:** the learned per-band rate tracks the analytic reverse-water-filling
  shape tightly and monotonically in β; the LEVEL does NOT transfer.
- **Numbers (Pearson r | θ | β/2 | total-rate nats | relL1):**
  - sib_b1e-4: r=**0.945** | θ=0.1233 | β/2=5e-05 | 79.13 | 0.517
  - sib_b3e-4: r=**0.960** | θ=0.2012 | β/2=1.5e-04 | 62.18 | 0.408
  - sib_b1e-3: r=**0.974** | θ=0.3941 | β/2=5e-04 | 44.70 | 0.286
  - sib_b1e-2: r=**0.991** | θ=2.1057 | β/2=5e-03 | 16.99 | 0.153
  - raw_vib_b1e-3: r=**0.889** | θ=0.2676 | β/2=5e-04 | 198.76 | 0.196
- **Source:** `scripts/validate_allocation.py`.
- **Label:** **DEPLOYED-MEASURED** shape validation. Honest split: SHAPE transfers
  (high r, monotone), LEVEL does NOT (θ is 3–4 orders of magnitude above β/2
  because the deployed module trains a denoising target with an MMSE decode).

## 5. Track B Wiener/MMSE decode — smoothness upside, SR-neutral

- **Claim:** the Wiener/MMSE decode (sib_b1e-4) is SR-neutral-to-slightly-positive
  under action noise while cutting RMS jerk ~4–5.5×; a naive fixed low-pass of
  comparable smoothing instead HURTS SR — so the smoothness gain is principled.
- **Numbers** (n=200 per cell, libero_spatial, 10 tasks × 20 init):
  - ΔSR sib_b1e-4 − vanilla: **+4.0 / +0.5 / −2.0** pp at noise 0.05/0.1/0.2
    (SR 0.585/0.555/0.525 vs vanilla 0.545/0.550/0.545).
  - low-pass ΔSR: **−0.5 / −2.5 / −3.5** pp (hurts).
  - jerk ratio vanilla/sib_b1e-4: **4.07× / 4.36× / 5.50×** lower
    (0.1108/0.1121/0.1141 vs 0.4511/0.4886/0.6280).
  - clean: SR 0.595 vs 0.570; jerk 0.1109 vs 0.4089 (~3.7×); HF-frac 0.00231 vs
    0.02570 (~11×).
- **Source:** Track-B eval tables (action-noise SR / RMS jerk / clean jerk ref;
  `eval_jerk.json` for the RASF-variant clean row).
- **Label:** **DEPLOYED-MEASURED**. Reported as a SMOOTHNESS result, NOT an SR headline.

## 6. Statistical rigor — which deltas exclude zero

- **Claim:** report only per-suite deltas whose 95% paired bootstrap CI (n=3 seed
  means, seeds 42/123/456) excludes zero as "real"; flag the rest.
- **EXCLUDES zero (real):**
  - CLEAN Spatial **+3.8 [+3.5,+4.0]**
  - CLEAN Object **+10.0 [+10.0,+10.0]** (n=3-degenerate CI)
  - CLEAN Goal **+3.0 [+0.5,+6.0]**
  - PLUS Object **+10.7 [+2.4,+16.7]**
  - PLUS Goal **+3.2 [+1.2,+7.1]**
  - PLUS Long **+3.6 [+1.2,+6.0]**
  - Long-bridge mult=0.25 CLEAN **+12.7 [+10.0,+17.5]**
- **EXCLUDES zero but is a REGRESSION:** CLEAN Long **−10.3 [−16.5,−4.5]**.
- **INCLUDES zero (inconclusive):** PLUS Spatial **+2.8 [+0.0,+6.0]** (LB = +0.0);
  Long-bridge mult=0.25 ROBUST **+3.2 [+0.0,+6.0]**.
- **INCOMPLETE (0/3 seeds, no CI fabricated):** Long-bridge mult=0.5 CLEAN & ROBUST.
- **IQM (base/AEGIS):** CLEAN Spatial 99.0/99.3, Object 86.7/100.0, Goal 83.0/85.3,
  Long 58.7/39.7; PLUS Spatial 61.1/66.7, Object 52.4/73.8, Goal 65.1/71.4, Long 2.4/9.5.
- **Source:** AEGIS-on-ACT stats script over existing JSONs (stratified-bootstrap
  CI + paired Δ CI + IQM + Wilson). No new rollouts.
- **Label:** **DEPLOYED-MEASURED**. Narrow/degenerate CIs are an n=3 artifact.

## 7. AEGIS-on-ACT reconciliation (honest proper base)

- **Claim:** on the PROPER (colleague's frozen) ACT base, AEGIS adds
  **+5.06 mean / +8.93 (≈+9.0) peak** LIBERO-Plus robustness (47.62% → 52.68%),
  4 suites × 3 seeds, all gates open.
- **Numbers (base → AEGIS, Δmean, Δpeak):**
  - Spatial 55.56→58.33, +2.78, +5.95
  - Object 51.19→61.90, +10.71, +16.67
  - Goal 57.54→60.71, +3.17, +7.14
  - Long 26.19→29.76, +3.57, +5.95
  - **Average 47.62→52.68, +5.06 mean, +8.93 peak** (peak = mean of 4 per-suite Δpeak).
- **Provenance:** `results/act_plus_v2/*/{base,aegis}/seed{42,123,456}/result.json`.
- **Superseded OLD disposable base** (`aegis_act_sweep_cap220`): +7.74 mean /
  +11.01 peak (17.16→24.90) — retained ONLY for provenance, NOT the headline.
- **Label:** **DEPLOYED-MEASURED**. Δmean = mean of per-suite paired 3-seed mean
  deltas (gate-off = baseline exactly, no oracle); Δpeak = best-of-3-seed, labelled.

---

## Open issues a referee could still attack

- **PLUS Spatial +2.8 CI lower bound = +0.0** → not significant; do not quote as a win.
- **CLEAN Long −10.3** is a real regression on full-strength RIB; the mult=0.25
  bridge fixes clean (+12.7) but robust Long stays inconclusive (+3.2 [+0.0,+6.0]).
- **Long-bridge mult=0.5 cells are 0/3 seeds** — incomplete, no CI. A referee will
  want these filled or dropped from any claim.
- **n=3 seeds** → paired-Δ CIs bootstrapped over 3 means; degenerate CIs (Object
  +10.0 [+10.0,+10.0]) are an artifact, not a certainty claim.
- **Single ACT base** — no official combined ACT base exists; cross-base
  generality of +5.06/+8.93 not established.
- **RASF blind to blur/viewpoint**; viewpoint at floor even for perception RIB.
- **θ=β/2 is compression-only** — deployed learned RASF matches shape (r up to
  0.991), not level (θ 0.123–2.106 vs β/2 5e-05–5e-03).
