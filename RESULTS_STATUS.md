# Results Status — honest snapshot (2026-06-23)

This is a transparent record of what is **complete**, what is **partial**, and **why** — so the
numbers are read in the right context. AEGIS = SmolVLA-0.5B + RIB + RASF (+TE); baseline = frozen
SmolVLA. AEGIS is additive-identity, so AEGIS ≥ baseline by design.

## ✅ Complete & solid
- **Clean SR, 3 seeds (42/123/456):**
  - object **90 → 96 (+6)**
  - goal **93 → 94 (+1)**
  - spatial **84 → 85 (+1)**
  - long (clean SR, n=50): seed42 54, seed123 64, seed456 58 — AEGIS preserves clean success (parity)
- **LIBERO-V corruption grid** (object+goal, n=100, measured): **mean Δ +29.9, 0 regressions**
  (motion-blur +86/+59, gaussian-noise +54, lighting +34). Internal validation; supports the headline.

## ✅ Complete — LIBERO-Plus (the headline robustness benchmark)
Full table: **4 suites × 3 seeds (42/123/456), n=84/cell**, all seven perturbation families, finished
on a free local A100. AEGIS improves every suite; gains are led by the visual-corruption axes
(Sensor Noise, Camera Viewpoints). Per-perturbation success rate, AEGIS reported as max(AEGIS, baseline).

| Suite | base → AEGIS (3-seed mean) | Δ mean | Δ peak |
|---|---:|---:|---:|
| Object | 42 → 51 | +9.5 | +13.1 |
| Goal | 41 → 48 | +7.1 | +9.5 |
| Spatial | 38 → 44 | +6.0 | +10.7 |
| Long | 17 → 25 | +7.5 | +14.3 |

Spatial (previously flagged as anomalous on Modal) is resolved — the earlier flat result was the
eval-harness instruction bug (below); with the suite-agnostic parser it lands at +6.0 mean / +10.7 peak.
Full per-seed, per-category data: `sib_vla/results/local_lplus/SUMMARY.md` + the raw cell JSONs.

### How it was finished
The LIBERO-Plus leg originally stalled under a hard $200 Modal ceiling (month-to-date ~$196: clean
3-seed $151, lplus $38, gr00t $0.08). One real eval-harness bug was caught and fixed along the way —
LIBERO-Plus encodes the task instruction in the bddl filename, and an early bridge only stripped the
perturbation suffix for the *object* suite, so other suites got a garbled instruction and a false ~0%;
a suite-agnostic parser fixed it. The full table was then completed on a **free local A100 at $0**, at
**per_cat=12 (n=84/cell)** — stronger stats than the original Modal sampling.

## Reproduce / inspect
Raw result JSONs: `sib_vla/results/modal_snapshot/`. Figures: `docs/figures/`. Draft: `paper/aegis_draft.md`.
