# LIBERO-Plus — the paper's robustness evaluation (3-seed, 4 suites)

**Decision:** LIBERO-V (our custom corruption axes) **is dropped from the paper** — it proved
the module works internally, but it is not a recognized benchmark. The paper's robustness
evidence is **LIBERO-Plus** (arXiv 2510.13626), a published benchmark, evaluated **paper-worthy**:
all 4 suites, 3 seeds, base vs AEGIS. **Eval-only** (reuse `smolvla_spatial_repro` +
`rib_on86`/`rasf_on86`; no training).

---

## Two deliverables (both multi-seed: 42, 123, 456)

### A · Clean per-suite SR — "we preserve clean success"
Standard LIBERO, no perturbation. 4 suites × {baseline, aegis} × 3 seeds = **24 cells**.
Currently only seed 42 exists → this adds 123, 456 and re-runs 42 into `seed<N>/` for a
uniform table. Stage: `smolvla_modal.py --stage clean`.

### B · LIBERO-Plus robustness — the headline
7 perturbation categories/suite. 4 suites × {baseline, aegis} × 3 seeds = **24 cells**,
each `per_cat` tasks/category. Stage: `lplus_modal.py --stage stage1`.

## Protocol (paper-worthy)
- **Arms:** baseline (SmolVLA + TE) vs AEGIS (RIB + RASF + TE). Δ = AEGIS − base.
- **Checkpoint:** `smolvla_spatial_repro` for BOTH arms (the one RIB/RASF were trained against;
  no-training constraint). Absolute clean SR runs ~5–10pp under the paper's official number —
  so we report **Δ on our honest baseline**, never against a paper figure. (Switching to the
  official `HuggingFaceVLA/smolvla_libero` ckpt would need RIB/RASF retraining — out of scope.)
- **Per-suite max-steps (critical):** spatial 220, object 280, goal 300, **long/libero_10 520**
  (truncating Long silently kills its SR — now baked per-cell).
- **Per-suite gating (your rule):** if AEGIS < base on a suite, gate off → report base.
  Identity-init makes this provably safe (0 regressions by construction).
- **Stats:** mean ± 95% Wilson CI over 3 seeds, per category and per-suite total.
- **Persistence:** `<suite>/seed<N>/<arm>.json`; resume-skip + 2-min commit daemon.

## Work size & cost

**B · LIBERO-Plus (24 cells):**
| per_cat | eps/cat/arm (×3) | Modal $ | Modal wall | A100 wall |
|---:|---:|---:|---:|---:|
| 30 | 90 | ~$56 | ~14 h | ~14 h |
| **40 (recommended)** | **120** | **~$74** | ~19 h | ~19 h |
| 50 | 150 | ~$94 | ~23 h | ~23 h |

**A · Clean SR (24 cells):** ~$17 Modal / ~5 h (clean is fast; no perturbation).

**Full paper run (A + B):** per_cat=40 → **~$91 Modal (one $100 reset, tight)** or **~1 day on a
dedicated A100 ($0)**. per_cat=30 → ~$73 (comfortable in one reset).

## Run it  (limit $200 → guard HARD_TOTAL=190, auto-abort with ~$15 margin)
```bash
cd sib_vla/multivla
# 0. smoke — calibrate per-task time + confirm the 7 category names (~$0.5)
BUDGET_GUARD=2   HARD_TOTAL=190 ./safe_modal_run.sh modal run lplus_modal/lplus_modal.py::main --stage smoke
# A. clean SR (4 suites × 3 seeds)
BUDGET_GUARD=22  HARD_TOTAL=190 ./safe_modal_run.sh modal run smolvla_modal/smolvla_modal.py::main --stage clean --episodes 10
# B. LIBERO-Plus robustness (4 suites × 3 seeds, per_cat 12)
BUDGET_GUARD=80  HARD_TOTAL=190 ./safe_modal_run.sh modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 12
# C. LIBERO-Plus-native pairwise videos (object suite, 7 cats × {base,aegis})
BUDGET_GUARD=6   HARD_TOTAL=190 ./safe_modal_run.sh modal run lplus_modal/lplus_modal.py::main --stage video
```
A100: `--stage clean` via local clean runner; LIBERO-Plus via `run_stage2_liberoplus.sh`
(needs the 4-suite + seed loop — local wiring TODO; Modal path is complete).

## Deliverable tables (what goes in the paper)
1. **Clean SR** — 4 rows (suite) × {base, AEGIS, Δ}, mean ± CI. Shows ΔSR ≈ 0 (preservation).
2. **LIBERO-Plus** — per suite, 7-row category table × {base, AEGIS, Δ}, + per-suite total,
   mean ± CI, gating note. This is the headline robustness result.
3. Figure (extend `make_readme_figures.py`); fill `paper/aegis_draft.md` §LIBERO-Plus.

## Modifications made for paper-worthiness
- LIBERO-Plus cells: **all 4 suites** (was object+goal), **per-suite max-steps** (Long=520),
  **3 seeds**, per_cat default 40.
- Clean-SR multi-seed stage added (`--stage clean`, 4 suites × 3 seeds).
- `--seed` + `--out` JSON on the eval (persistence + resume).
- Paper draft repositioned: LIBERO-Plus = headline; LIBERO-V → dropped (internal validation only).

## Open decisions (need your call)
1. **per_cat:** 40 (recommended, ~$74) vs 30 (cheaper, ~$56) vs 50 (strongest, ~$94).
2. **Compute:** Modal (one reset, ~$91 for A+B) vs dedicated A100 (~1 day, $0) — A100 preferred
   given the budget freeze.
3. **Spatial in the table?** You listed all 4 suites — included by default; say if you want
   object/goal/long only.
