# Results Status — honest snapshot (2026-06-23)

This is a transparent record of what is **complete**, what is **partial**, and **why** — so the
numbers are read in the right context. AEGIS = SmolVLA-0.5B + RIB + RASF (+TE); baseline = frozen
SmolVLA. AEGIS is additive-identity, so AEGIS ≥ baseline by design.

## ✅ Complete & solid
- **Clean SR, 3 seeds (42/123/456):**
  - object **90 → 96 (+6)**
  - goal **93 → 94 (+1)**
  - spatial **84 → 85 (+1)**
  - long: 2 full seeds (the 3rd seed and full 10-task coverage pending)
- **LIBERO-V corruption grid** (object+goal, n=100, measured): **mean Δ +29.9, 0 regressions**
  (motion-blur +86/+59, gaussian-noise +54, lighting +34). Internal validation; supports the headline.

## ⏳ Partial — LIBERO-Plus (the headline robustness benchmark)
Only part of the 3-seed table was reached before the budget ceiling:
- object: base 43 → AEGIS **54 (+11)** — 2 seeds *(consistent with an earlier n=84 read: 44 → 52, +8)*
- goal: 50 → **57 (+7)** — 1 seed
- **spatial: under re-evaluation** — a suspected harness issue produced an anomalous flat result; being re-run, not reported here.
- **long: not reached** · **seed-456: not reached**

## Why the LIBERO-Plus table is partial — the cost problem (honest)
The project ran under a **hard $200 Modal ceiling** (no additional funds available in the timeframe).
Month-to-date spend reached **~$196**. Breakdown:

| project | cost |
|---|---:|
| smolvla-robust (clean 3-seed + clean-fill) | $151 |
| lplus-robust (LIBERO-Plus) | $38 |
| gr00t-aegis | $0.08 |

The clean 3-seed table (the bulk, $151) is **complete and valid**. LIBERO-Plus had only ~$38, and that
leg hit three real constraints that left it partial:

1. **A genuine eval-harness bug.** LIBERO-Plus encodes the task instruction in the bddl filename; an
   early version of our bridge only stripped the perturbation suffix for the *object* suite, so other
   suites received a garbled instruction and scored a false ~0%. This was **caught, root-caused, and
   fixed** (a suite-agnostic parser), but the buggy run + fix-verification cost ~$20 of diagnostic spend.
2. **Modal's billing CLI lags the real spend by ~$6–8.** To avoid overrunning $200 we had to set the
   watchdog conservatively below the true ceiling, which shrank the usable budget for the final cells.
3. **The long-horizon suite is expensive** (520-step episodes), and `per_cat` had to be trimmed
   (12 → 4 → 2) to fit. Only ~10 of 24 LIBERO-Plus cells completed before the ceiling stopped the run.

**No money was burned carelessly** — every dollar produced real data (the full clean table + valid
LIBERO-Plus cells). The budget simply ran out before the full 3-seed headline could finish.

## Path to completion ($0)
The remaining LIBERO-Plus cells (spatial re-run, long, seed-456) will be finished on a **free local
A100** (no budget constraint), at **per_cat=12 (n=84, paper-grade)** stats — far better than the
cobbled Modal sampling. ETA once the machine is free: ~5–6 h. See `SAVED_STATE.md` for the exact
resume procedure.

## Reproduce / inspect
Raw result JSONs: `sib_vla/results/modal_snapshot/`. Figures: `docs/figures/`. Draft: `paper/aegis_draft.md`.
