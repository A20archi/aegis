# AEGIS → workshop-credible: research scoping + roadmap

*Synthesized from a 4-track literature sweep (ARXIV / NeurIPS / CoRL / ICLR / RSS + their workshops). Goal: make the documentation and results genuine, honest, and credible to a workshop reviewer. All arxiv IDs below were returned by the research pass; a handful of 2026 IDs are flagged "verify-before-cite".*

---

## 0. The one-sentence thesis to defend
> "A parameter-efficient module on a **frozen** VLA, **identity-at-init so it provably cannot degrade clean behaviour**, buys large, statistically-supported robustness gains under simulated perturbation (LIBERO-Plus), reported per-perturbation with confidence intervals and the clean-task cost shown openly."

The bolded clause is our **novelty wedge** — *no competitor proves module-off = baseline.* Lead with it.

---

## 1. Where AEGIS stands vs the field

**Our perturbation suite mirrors a real, standard benchmark.** The 7 axes we test (sensor noise, viewpoint, lighting, background, layout, language, robot-init) are essentially **LIBERO-Plus** (Fei et al., arXiv:2510.13626). So we are *on a recognized benchmark*, not a homemade one — state this explicitly and cite it.

**Closest related work (cite + differentiate by design — NO head-to-head benchmark):**
- **StableVLA / IB-Adapter (arXiv:2605.18287)** — an Information-Bottleneck adapter on the frozen vision→policy connector, conceptually adjacent to our RIB leg. **We cite it as related work and differentiate on design, not by an empirical comparison:** (i) corruption-trained to drop a corruption subspace, (ii) **formal identity-preservation** (gate-off = baseline *exactly*) which it does not provide, (iii) an action-side leg (RASF) + temporal ensembling. We position RIB via the IB lineage (Deep VIB, BC-IB), not as "beats StableVLA." *(Decision: StableVLA head-to-head comparison dropped.)*

**Our three legs each have a clear precedent we must cite (and out-frame):**
- RIB ← **Deep VIB** (Alemi et al., ICLR 2017, 1612.00410) + **BC-IB** (Bai et al., ICML 2025, 2502.02853, IB-in-behaviour-cloning on LIBERO) + **DRIBO** (ICML 2022, 2102.13268, IB compresses visual distractors → robustness).
- RASF ← **FAST** (Pi, RSS 2025, 2501.09747, DCT on action chunks) — our edge: we *denoise* in DCT space over a frozen VLA rather than tokenize.
- Temporal ensemble ← **ACT** (Zhao, RSS 2023, 2304.13705). **Caveat:** **RTC** (Black, NeurIPS 2025, 2506.07339) shows naive temporal-ensemble averaging can blend valid actions into invalid ones — we must pre-empt this (our identity-init conservative filter is the answer).

**Action-fragility supports RASF:** **RobustVLA** (ICLR 2026, 2510.00037) independently finds *action* is the most fragile modality — strong external motivation, but it's also a baseline reviewers will demand we beat on the action axis.

**Frozen-VLA robustness sibling / validation exemplar:** **BYOVLA** (2410.01971) reports nominal (clean) SR as an explicit ceiling and shows recovery to it — the gold-standard "doesn't hurt clean" methodology. Our gate-off=baseline guarantee is the stronger version.

---

## 2. The credibility gaps to close (this is what the internal feedback caught)

Ordered; **P0 are blocking — integrity, not just rigor.** The `max(method,baseline)` oracle and outcome-chosen gating are the *exact pattern reviewers classify as misleading reporting* (the malpractice LIBERO-PRO, 2510.03827, was written to expose). These must be **removed, not supplemented.**

| # | Gap (from feedback) | Correct practice | Justifying paper |
|---|---|---|---|
| **P0-a** | per-category `max(AEGIS,baseline)` oracle reported as a result | Label as **"oracle skyline / upper bound"** only, never a deployable number; show the gap to the real system | Agarwal 2108.13264; Pineau 2003.12206 |
| **P0-b** | per-suite gating that masks a clean loss | **3-row ladder per suite**: always-on (ungated) / input-adaptive-gated / baseline. The gate must decide from **inference-time inputs**, not the success label. Show the ungated clean Δ openly | Tsipras 1805.12152; Agarwal 2108.13264 |
| **P0-c** | clean SR loss hidden by gating (e.g. Long −8) | **Report the true ungated clean number incl. the loss.** Own a small clean cost; show robustness gain dwarfs it | Tsipras 1805.12152 |
| **P1** | single-seed headline numbers | **≥3 seeds (target 5)**, report **IQM + 95% stratified-bootstrap CI** (rliable), per-seed disclosed, **paired per-seed Δ** with CI. Retire "best-of-seed" (it's the seed-analogue of the oracle) | Henderson 1709.06560; Agarwal 2108.13264; Colas 1806.08295 |
| **P1b** | gains carried by one axis (motion-blur 4→50) | **Per-axis table mandatory**; aggregate **baseline-normalized** (ImageNet-C style) so no axis dominates; **state where we don't help** | Hendrycks 1903.12261 |
| **P2** | weak/no baselines, no ablation | Add cheap baselines (test-time aug, BN/norm-stats adaptation); ablate **RIB / RASF / TE / gate-on-off**; clean vs perturbed breakdown | CoRL/NeurIPS reviewer norms |
| **P3** | reproducibility / scope | Release code+configs+seeds(42/123/456)+compute; **Limitations** section (sim-only, clean cost, blind axes); scope title to "simulated perturbations on LIBERO-Plus" | NeurIPS checklist; CoRL author instructions |

**SR confidence intervals:** use the **Wilson score interval** per cell (good at small n / near 0–1). Note: n=25 gives ~±20pp at p≈0.5 — which is exactly why our Long n=25 numbers need CIs and ≥50 rollouts/task.

---

## 3. Must-cite / must-compare paper list

**MUST CITE (8 core):**
1. LIBERO — Liu et al., NeurIPS 2023 D&B — **2306.03310** (benchmark)
2. LIBERO-Plus — Fei et al. — **2510.13626** (our robustness benchmark; verify 2026 status)
3. OpenVLA — Kim et al., CoRL 2024 — **2406.09246** (the eval protocol everyone copies: 3 seeds × 500 rollouts/suite, mean±SE, Long max_steps=520)
4. Deep VIB — Alemi et al., ICLR 2017 — **1612.00410** (the VIB we implement)
5. BC-IB — Bai et al., ICML 2025 — **2502.02853** (IB-in-BC on LIBERO; our key precedent for RIB)
6. FAST — Pi, RSS 2025 — **2501.09747** (DCT-on-action-chunks; precedent for RASF)
7. ACT — Zhao et al., RSS 2023 — **2304.13705** (temporal ensembling + chunking)
8. StableVLA / IB-Adapter — **2605.18287** (closest related IB-on-frozen-VLA; cited as related work, **not benchmarked**; *verify-before-cite*)

**MUST COMPARE / BASELINES a reviewer will demand:**
- **RobustVLA (2510.00037, ICLR 2026)** — under the same perturbation axes.
- **BYOVLA (2410.01971)** — frozen, no-retrain robustness baseline + clean-preservation protocol.
- Cheap controls: **test-time augmentation**, **BN/normalization-stats adaptation**, and a **naive low-pass / CAPS (2012.06644)** action smoother (to show DCT-spectral beats trivial smoothing).
- *(StableVLA head-to-head comparison dropped — cited as related work only.)*
- **Ablations:** RIB-only, RASF-only, RASF−TE, gate on/off, and **module-off = baseline on CLEAN LIBERO** (the identity-preservation headline).

**SUPPORT / DEFENSE CITES:**
- DRIBO (2102.13268), RTC (2506.07339, pre-empt the averaging critique), BID (2408.17355), Diffusion Policy (2303.04137), COLOSSEUM (RSS 2024, 2402.08191, the per-factor table format), SIMPLER (CoRL 2024, 2405.05941, sim↔real correlation precedent), LIBERO-PRO (2510.03827, evaluation-integrity reference — citing it pre-empts the cherry-pick objection).

**METHODOLOGY (the honest-reporting backbone):**
- Henderson "Deep RL that Matters" AAAI 2018 — **1709.06560**
- Agarwal "Statistical Precipice" / **rliable** NeurIPS 2021 — **2108.13264** (IQM + stratified bootstrap CI + performance profiles)
- Colas "How many seeds" — **1806.08295**; "Hitchhiker's Guide" — **1904.06979**
- Pineau ML-reproducibility — **2003.12206**
- Hendrycks ImageNet-C — **1903.12261** (per-corruption breakdown + baseline-normalized aggregation)
- Tsipras "Robustness May Be at Odds with Accuracy" ICLR 2019 — **1805.12152** (own the trade-off)

---

## 4. The roadmap (phased, for the existing project)

**Phase A — Integrity reset (blocking, do first).**
- [ ] Delete every oracle/max() artifact (done: local_lplus/SUMMARY.json, modal_snapshot/CLEAN_PERTASK_GATED.json) + neutralize the oracle-generating script.
- [ ] Strip all "oracle" mentions from README/RESULTS_STATUS/paper; replace with the honest-standard one-liner (§5).
- [ ] Re-state every table as **ungated, no-oracle, single fixed policy**. Show the clean Δ including Long's loss.
- [ ] Fix the Long clean-SR regression *at the module level* (in progress: RASF validated-conservative → RIB residual accumulation is the cause; make RIB identity-on-clean so the raw module is ≥ baseline, not gated to it).

**Phase B — Statistical rigor.**
- [ ] 3 seeds (42/123/456), target 5, for **every** cell incl. baselines, ≥50 rollouts/task/condition.
- [ ] Report **IQM + 95% stratified-bootstrap CI** (rliable) for aggregates; **Wilson interval** per SR cell; **paired per-seed Δ** with CI.
- [ ] Per-axis robustness table, baseline-normalized aggregation; state the blind axes.

**Phase C — Completeness.**
- [ ] Baselines: test-time-aug / BN-adaptation controls, + (if reproducible) RobustVLA/BYOVLA. *(StableVLA cited as related work, not benchmarked.)*
- [ ] Ablations: RIB / RASF / TE / gate; module-off=baseline on clean (the identity headline).
- [ ] Severity-sweep **trade-off curve** (clean→high corruption) showing the crossover — the single most convincing "effect is real" artifact.

**Phase D — Framing & submission.**
- [ ] Limitations section (sim-only, clean cost, blind axes, no real robot).
- [ ] Scope title/abstract to "robustness under simulated perturbations on LIBERO/LIBERO-Plus".
- [ ] Release code/configs/seeds/compute.
- [ ] Target a **CoRL/NeurIPS/ICLR/RSS workshop** (sim-only + module + 3 seeds + ablation is exactly in-lane; CoRL explicitly does not require beating SOTA or a real robot).

**Phase E — (stretch) real-robot / SIMPLER.** Already scoped separately: SIMPLER-WidowX + OpenVLA-7B (cached). Not required for a workshop robustness paper, but converts it toward a main-conference submission.

---

## 5. The honest-reporting standard (put this verbatim in the README/paper methods)
> *Headline = a single fixed policy, IQM over ≥3 disclosed seeds (42/123/456) with 95% stratified-bootstrap CIs; per-cell success rates carry Wilson intervals; robustness reported per-perturbation-axis (baseline-normalized) with the clean-task delta shown including any loss; any gating is reported alongside its ungated number and decides only from inference-time inputs; oracle/skyline numbers are labelled upper bounds, never results.*

This single paragraph retires all four flagged practices at once.

---
*Verify-before-cite (2026 search-index IDs): 2605.18287, 2510.00037, 2510.13626, 2604.18107. The methodology + foundational cites (1709.06560, 2108.13264, 1903.12261, 1805.12152, 1612.00410, 2502.02853, 2304.13705, 2501.09747, 2406.09246, 2306.03310) are independently confirmed.*
