# scenario.md — Decision Playbook for the SIB Build

Companion to `project.md`. That file says *what to build and in what order*; this
file says *what to do when a gate's outcome is ambiguous or negative*, and which
ordered steering actions give the best honest shot at landing the three legs on
the positive side.

Read this alongside `project.md`. The stage spine is identical (Stage 0 → 4).
Each scenario is keyed `S<stage>.<letter>`. Two cross-cutting scenarios (`SP`
power, `SB` budget) apply at every stage.

---

## 0. Prime directive (read first — it constrains everything below)

The goal is to land **Leg 1, Leg 2, Leg 3 positive**. "Positive" is earned, not
forced. Concretely, for every scenario:

1. Run the **ordered steering actions** before accepting any fallback. A negative
   finding is only valid *after* the cheap, legitimate fixes are exhausted.
2. Never tune on the test split, never reselect seeds/tasks after seeing success,
   never widen a baseline's disadvantage to make SIB win. Steering = better
   engineering and experimental design, not data massaging.
3. If a steering action would only flip the sign by exploiting noise, **stop and
   report the honest result** instead. Reviewers punish fishing; a clean negative
   beats a fragile positive.
4. When a gate genuinely blocks the whole story (Stage 0 fails, or the core
   claim is contradicted after steering), **escalate to the human** rather than
   improvising scope.

If forced to triage under budget: **Leg 1 + Leg 2 are the safe bank** (they don't
require beating a baseline on a binomial metric), **Leg 3 is the upside**. Spend
remaining GPU-hours on whichever leg is closest to flipping, not evenly.

---

## 1. The three legs — positive criteria (the targets)

| Leg | Positive condition (all must hold) | Decided in |
|---|---|---|
| **Leg 1 — Frontier** | β-sweep yields a monotonic rate–distortion curve with a **flat-success plateau**: a contiguous β range where success stays inside the vanilla Wilson CI while `ΣR_k`, RMS jerk, and HF-energy-fraction all drop meaningfully (jerk ↓ ~20%+ as a guide, not a hard gate). | Stage 2 (mini), Stage 3 (full) |
| **Leg 2 — Allocation** | (a) learned `{R_k}` tracks the analytic reverse-water-filling allocation **under the MSE surrogate** (visual + quantitative correspondence); **and** (b) task-conditioned allocation — precision/contact-rich tasks put more bits in higher bands than coarse tasks, difference statistically separable. (b) requires `sigma_mode: context`. | Stage 1 (feasibility), Stage 4 (proof) |
| **Leg 3 — Active ingredient** | SIB beats **`raw_vib`** on ≥1 of {corruption robustness, jerk-at-matched-success}, gap clearing a two-proportion z-test (robustness) or clearly outside CI overlap (jerk); **and** SIB beats grid-searched low-pass and the β=0 learned gain on that same axis. The "spectral basis matters" claim rests on the `raw_vib` win specifically. | Stage 3 |

---

## 2. Operating rules for the coder

- Gates are **sequential**. Do all steering for a stage *within* that stage before
  moving on; do not carry an unresolved gate forward silently.
- Every run logs GPU-hours to `results/gpu_hours.csv` and saves its resolved config.
- Continuous metrics (jerk, HF energy, bits-per-band, allocation correlation) are
  **not** binomial-limited — prefer them when success differences sit near the
  noise floor. They are often where Leg 1 and Leg 2 are actually won.
- Run `tests/` green before trusting any negative result. A failing transform or
  a gradient not reaching `log_var` masquerades as a real finding.
- When a scenario says "check X", that means add a logged assertion or a one-off
  diagnostic to `results/`, not a mental note.

---

## Stage 0 — Reproduce vanilla baseline

### S0.A — Success in [0.65, 0.80]
**Trigger:** clean vanilla success inside the documented reproduction band.
**Path:** none needed. Record the value; this is the "tie" reference and the
center of every later Wilson-CI comparison. Proceed to Stage 1.

### S0.B — Success below 0.65 (BLOCKS EVERYTHING)
**Trigger:** vanilla success < 0.65.
**Risk:** all three legs — no trustworthy reference to tie against.
**Path (ordered):**
1. Verify the eval harness, not the model: action de-normalization, chunk
   extraction shape/dtype (`predict_action_chunk` output), and that `H`/`chunk_size`
   match the checkpoint. A wrong denorm or transposed chunk reads as a bad policy.
2. Confirm the LIBERO env stack (robosuite/libero versions) matches the
   checkpoint's training setup.
3. Try an alternate community LIBERO checkpoint (project.md Stage 0 fallback).
4. Re-run on a second seed set to rule out an unlucky draw.
**Positive exit:** success enters [0.65, 0.80] → treat as S0.A.
**Escalate:** if still < 0.65 after the above, stop and surface to the human.
Do not build on a broken baseline.

### S0.C — Success above 0.80
**Trigger:** vanilla success > 0.80.
**Path:** good, but verify it is a real held-out eval (no train/eval leakage,
correct task split). Then recenter the "tie" band on this value — the Leg 1
plateau is judged against *this* CI, not the paper's number.

---

## Stage 1 — Spectral pre-check (decides Leg 2 feasibility)

### S1.A — Energy spread across bands, low off-diagonal
**Trigger:** per-band DCT energy is distributed (not collapsed), off-diagonal
covariance ratio low.
**Path:** ideal. Leave `rotation: none`. Leg 2's allocation story has room to
exist. Proceed.

### S1.B — Energy collapses into the lowest ~2–3 bands (Gate A)
**Trigger:** nearly all energy in 2–3 low bands across dims.
**Risk:** **Leg 2** — "allocation" degenerates into "drop almost everything."
**Path (ordered):**
1. Re-run the spectrum **per task group**, not pooled — precision/contact-rich
   tasks (LIBERO-Long stages, insertion) often carry more high-band energy than
   the pooled average suggests. Bias the primary suite toward these.
2. Inspect **per-dim** spectra separately; the gripper dim and rotational DoF
   frequently have richer spectra than translation.
3. If the policy's `chunk_size` is short, a longer chunk exposes more resolvable
   bands — only if the checkpoint supports it without retraining.
**Positive exit:** at least the precision-task / per-dim view shows usable
high-band energy → Leg 2 (b) becomes testable; note the suite bias in the report.
**Honest fallback:** if energy is genuinely ~2–3 bands everywhere, record in
`results/stage1.json` that the allocation story is weak, and reframe Leg 2 as a
*characterization of action-chunk spectra* rather than a rich allocation result.
Lean the paper onto Leg 1 + Leg 3.

### S1.C — High off-diagonal energy: bands correlated (Gate B)
**Trigger:** off-diagonal covariance ratio high — the independent-channel
precondition is violated.
**Risk:** **Leg 1 and Leg 2** cleanliness — rate-per-band and water-filling
assume decorrelated bands.
**Path (ordered):** this is a *designed remedy*, not a failure.
1. Set `rotation: pca` (eigenvectors of the band covariance) for all later runs;
   re-run `decorrelation_check.py`.
2. If still correlated, set `rotation: learned` (Cayley-parameterized orthogonal).
**Positive exit:** post-rotation off-diagonal ratio drops to the S1.A regime →
proceed with rotation enabled and report it as a principled preprocessing step.
**Note:** rotation on/off becomes a Stage 4 ablation; keep both configs.

---

## Stage 2 — Train SIB at one β (the MVE; mini-test of Leg 1)

### S2.A — Success within ~2pp, jerk + HF down, rate decreased
**Trigger:** the project.md Stage 2 gate passes.
**Path:** green. Save `results/sib_mve.json` and the first bits-per-band heatmap.
Leg 1 is on track; proceed to the sweep.

### S2.B — Success drops > 2pp
**Trigger:** SIB at mid β loses more than ~2pp success.
**Risk:** **Leg 1** plateau may not exist at this β.
**Path (ordered):**
1. Lower `beta` (the curve's flat region may sit at a smaller rate cost).
2. Confirm `decode: mmse` at eval (stochastic decode at inference injects noise
   the robot shouldn't see).
3. Confirm the distortion target is **`A_star`**, not the model's own output —
   a self-distillation target can drag success down.
4. Verify `lambda_k` is warm (initialized from `estimate_lambda.py`); a cold EMA
   distorts the Wiener gain early and destabilizes training.
**Positive exit:** success returns within ~2pp at some β with rate still reduced.
**Honest fallback:** if success only recovers when nothing is suppressed, that is
an S3.B signal (no plateau) — handle there.

### S2.C — Nothing is suppressed (rate not decreasing)
**Trigger:** `ΣR_k` flat at init; gains all ~1.
**Risk:** **Leg 1** — no compression means no frontier.
**Path (ordered):**
1. Raise `beta`.
2. Run `test_bottleneck.py` — confirm gradient reaches `log_var` and `R` is
   actually in the loss with the right sign.
3. Check `log_var` init / `softplus` so `sigma2` can grow (it must be able to
   move bands toward suppression).
**Positive exit:** at least the high bands begin to suppress as β rises.

### S2.D — Success fine, but jerk/HF not reduced
**Trigger:** success holds, smoothness metrics unchanged.
**Risk:** **Leg 1** (the jerk/HF half of the plateau claim).
**Path (ordered):**
1. Run `test_transforms.py` (Parseval + IDCT∘DCT) — a wrong transform suppresses
   the wrong bands.
2. Inspect which bands are being suppressed; if the model is cutting low bands and
   keeping high ones, λ or the gain wiring is inverted.
3. If HF λ is already ~0 (nothing to cut), this is the S1.B collapse surfacing —
   route to S1.B's task-bias remedy.
**Positive exit:** suppression concentrates in high bands → jerk/HF drop.

---

## Stage 3 — Frontier + isolating baselines (decides Leg 1 full + Leg 3)

### S3.A — Clean monotonic frontier with a flat-success plateau
**Trigger:** β-sweep gives monotonic `ΣR_k` ↓ with success flat-then-falling.
**Path:** **Leg 1 positive.** This is the headline figure. Proceed to Leg 3 tests.

### S3.B — No plateau: success falls immediately for any β > 0
**Trigger:** success drops as soon as the rate penalty turns on.
**Risk:** **Leg 1.**
**Path (ordered):**
1. Refine the β grid near zero (`1e-5, 3e-5, 1e-4, ...`) — the plateau may be
   narrow, not absent.
2. **Warm-start** SIB from the converged `gain_no_rate` (β=0) solution, then
   anneal β up — avoids a destructive cold start.
3. Train longer (the bottleneck is tiny; a few more steps is cheap).
4. Switch to `sigma_mode: context` — task-conditioned allocation often preserves
   success at a given rate better than a global allocation.
5. As the project.md option, **co-tune the action head** briefly (only if frozen
   head is demonstrably the bottleneck; log the extra GPU-h).
**Positive exit:** a contiguous flat-success β range appears → Leg 1 positive.
**Honest fallback:** if no plateau survives, report the frontier as "monotonic
trade-off with no free lunch on this suite" — still a valid Leg 1 curve, weaker
claim; flag for the human before spending more budget.

### S3.C — SIB vs raw_vib (THE make-or-break for Leg 3)

#### S3.C.green — SIB beats raw_vib on robustness or jerk-at-matched-success
**Trigger:** gap clears the z-test (robustness) or sits clearly outside CI overlap
(jerk), **compared at matched success**, with rotation applied if S1.C fired.
**Path:** **Leg 3 positive** — the spectral basis is the active ingredient. This
is the claim the paper hinges on. Lock the config; do not keep fishing for a
larger gap.

#### S3.C.red — raw_vib matches SIB everywhere
**Trigger:** no separable SIB advantage over raw_vib on any axis.
**Risk:** **Leg 3** — without this, the spectral story collapses to "an IB on
actions."
**Path (ordered — exhaust before declaring negative):**
1. **Compare at matched success, not matched β.** raw_vib and SIB hit different
   rates at the same β; align them on the success axis before reading the
   robustness/jerk gap. A mismatched comparison can hide a real effect.
2. Confirm **rotation is applied** to SIB if Stage 1 flagged correlated bands —
   the spectral advantage may only appear once bands are decorrelated.
3. **Push corruption severities higher.** The information-limit benefit often
   shows only under stronger observation noise; a flat result at low severity may
   be a ceiling, not a null.
4. Switch SIB to `sigma_mode: context` — the spectral basis may only pay off when
   the per-band budget is task-conditioned (this is the strongest variant).
5. Check **power** (see `SP`): a real-but-small gap can be invisible at 30
   episodes; if continuous metrics (jerk) already lean SIB's way, add episodes
   within budget on the corruption comparison only.
**Positive exit:** a defensible SIB win on ≥1 axis against raw_vib → Leg 3 positive.
**Honest fallback (valid only after 1–5):** write the negative finding —
"per-band information bottleneck on actions helps; the frequency basis is not the
active ingredient." Per project.md this is a clean publishable result. Stop
expanding scope and inform the human that the paper's framing has shifted.

### S3.D — SIB loses to low-pass or to the β=0 gain (hole in the claim)
**Trigger:** grid-searched low-pass or unconstrained gain matches/beats SIB.
**Risk:** **Leg 3** hygiene — implies the rate constraint isn't earning its keep.
**Path (ordered):**
1. Verify the comparison is fair both ways: the low-pass cutoff grid is honest
   (not sandbagged), and `gain_no_rate` is genuinely unconstrained (β=0, gain free
   in [0,1], no channel).
2. Check the β range actually spans the useful regime (an all-too-aggressive grid
   can make SIB look worse than a mild low-pass).
3. Confirm rate is reducing where it should (ties to S2.C).
**Positive exit:** SIB beats both on the axis where Leg 3 is claimed.
**Honest fallback:** if SIB genuinely cannot beat a tuned low-pass, the core
contribution is in question — **escalate to the human**; this is more serious than
the raw_vib negative.

---

## Stage 4 — Robustness + ablations + allocation (proves Leg 2)

### S4.A — Learned {R_k} matches water-filling under the MSE surrogate (Leg 2a)
**Trigger:** comparing learned per-band rate against the analytic reverse-water-
filling allocation computed from `λ_k` and the β-implied water level.
**Path (ordered to secure the positive):**
1. Compute the analytic allocation **under the MSE surrogate**, not the full flow
   loss — the water-filling prediction only holds for squared error. Compare on
   that footing.
2. Confirm `λ_k` EMA has converged before reading the allocation.
3. Overlay analytic vs learned `R_k`; report correlation and the visual match.
**Positive exit:** clear correspondence → Leg 2 (a) positive (the validation the
plan calls the strongest single result).
**If mismatch:** most often a wrong comparison basis (full-loss vs MSE) or an
unconverged λ — fix and recompute before concluding the mechanism diverges.

### S4.B — Task-conditioned allocation: precision tasks buy more HF bits (Leg 2b)
**Trigger:** per-task `R_k` under `sigma_mode: context`.
**Risk:** **Leg 2** money result.
**Path (ordered):**
1. This requires **`sigma_mode: context`** — a global σ cannot condition on task,
   so it cannot show this effect. Train/eval the context variant explicitly.
2. Group tasks into precision/contact-rich vs coarse; compare HF-band bit share
   across groups with a stat (not eyeballing).
3. If S1.B fired (energy-collapsed suite), use the precision-biased suite from
   that remedy — the effect is most visible there.
**Positive exit:** separable HF-bit difference between task groups → Leg 2 (b)
positive; this is the most quotable interpretability finding.
**Honest fallback:** if context σ shows no task structure, drop the
context-conditioning claim and report Leg 2 on (a) alone (still a valid result);
note context-conditioning gave no lift, per the plan's fallback.

### S4.C — Supporting ablations
**Trigger:** context vs global σ, stochastic vs MMSE decode, rotation on/off.
**Path:** report all regardless of direction — these contextualize the legs and
are expected content, not gated. MMSE should beat stochastic at eval (sanity);
if not, re-check the inference path drops the noise.

---

## Cross-cutting scenarios

### SP — A comparison lands within noise (power check)
**Trigger:** any success-rate gap (tie claim or win claim) sits near the CI floor.
**Path (ordered):**
1. Report the **Wilson CI width** at the current n; a Leg-1 "tie" needs the gap
   inside the CI, a Leg-3 "win" needs it outside.
2. Prefer the **continuous metrics** (jerk, HF energy, allocation correlation) —
   they are not binomial-limited and often carry the legs that success can't.
3. Only if a continuous metric already leans the right way, add episodes (within
   budget) to the *specific* comparison that matters — never broadly, never after
   peeking to pick a favorable subset.
**Guardrail:** if the effect only exists by exploiting noise, it does not exist.
Report the tie/negative honestly.

### SB — Budget pressure (GPU-hours)
**Trigger:** `results/gpu_hours.csv` trending toward the 80h ceiling.
**Path (ordered):**
1. Spend remaining hours on the leg **closest to flipping positive**, not evenly.
2. Cut the lowest-value item first: typically the redundant ablation, then extra
   β-grid points already on the flat part of the frontier.
3. Protect the make-or-break runs: Stage 0 reference, the β-sweep plateau region,
   the SIB-vs-raw_vib robustness comparison, and the allocation figures.
4. If a steering action (co-tune head, more episodes, learned rotation) would
   blow the budget, surface the trade-off to the human before spending.

---

## 3. Definition of the positive paper (all three legs green)

When everything lands on the positive side, the results set is:

- **Leg 1:** a rate–distortion frontier figure with a labeled flat-success plateau;
  success inside the vanilla Wilson CI across that β range; jerk and HF energy
  reduced; rate monotonically decreasing.
- **Leg 2:** the bits-per-band heatmap; an analytic-vs-learned allocation overlay
  matching under MSE; and a per-task allocation comparison showing precision tasks
  buying more high-band bits, with a reported statistic.
- **Leg 3:** SIB beating raw_vib on robustness or jerk-at-matched-success
  (z-test / CI), plus beating the tuned low-pass and the β=0 gain on that axis —
  licensing the "spectral basis is the active ingredient" claim.

All success comparisons carry Wilson CIs; corruption comparisons carry the
two-proportion z-test; every run logs its config and GPU-hours. No claim of
closed-form rate-distortion optimality anywhere — the rate is a variational upper
bound, as `project.md` requires.

**Triage order if the budget forces a choice:** secure Leg 1 and Leg 2 first (they
don't ride on a binomial win), then spend the remainder chasing the Leg 3
raw_vib gap. A paper with Legs 1 + 2 positive and Leg 3 as an honest negative is
still publishable; a forced Leg 3 is not.
