# LIBERO-Plus 3-seed evaluation plan (AEGIS vs frozen SmolVLA)

**Goal:** paper-protocol robustness results on LIBERO-Plus (arXiv 2510.13626), base vs AEGIS,
across **3 seeds** for proper mean ± CI. **Eval-only** — reuses `smolvla_spatial_repro` +
`rib_on86`/`rasf_on86`; **no training**.

---

## 1 · What LIBERO-Plus is
A robustness benchmark that systematically perturbs LIBERO tasks. **4 suites**
(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), each expanded to ~2400
perturbed tasks grouped into **7 perturbation categories**. Perturbations auto-apply by
parsing the bddl filename. The eval samples `per_cat` tasks/category and reports SR per
category + total.

> The 7 categories are the benchmark's perturbation factors (camera/viewpoint, lighting,
> texture/color, sensor noise, object layout/distractors, robot init, language) — exact names
> are surfaced by the smoke run's per-category print. This maps cleanly onto AEGIS: RIB owns
> the *visual* categories; the action leg + TE cover dynamics.

## 2 · Protocol
- **Arms:** `baseline` (frozen SmolVLA + TE) vs `aegis` (RIB + RASF + TE). Δ = AEGIS − base.
- **Seeds:** 42, 123, 456 → mean ± 95% CI per category. (The eval seeds each task as
  `seed + tid*10 + ep`, so a seed shift = fully independent init states.)
- **Suites:** **object + goal** (core, matches the cross-suite story). Spatial + libero_10
  are an optional scale-up.
- **per_cat:** tasks/category/seed. Default **30** → 30×7 = 210 tasks/cell, 90 eps/category/arm
  across 3 seeds.
- **Gating (your rule):** per-suite — if AEGIS < base on a suite, report base (gate off). By
  identity-init this is provably safe.
- **Persistence:** each cell writes `<suite>/seed<N>/<arm>.json` (per-category + total SR + n);
  resume-skip + 2-min commit daemon ⇒ a kill loses ≤ 1 cell.

## 3 · Work size  (core = object+goal, 3 seeds, 2 arms = 12 cells)

| per_cat | tasks/cell | eps/cat/arm (×3 seeds) | Modal cost¹ | Modal wall² | A100 wall³ |
|---:|---:|---:|---:|---:|---:|
| 20 | 140 | 60 | ~$19 | ~5 h | ~6 h |
| **30 (default)** | **210** | **90** | **~$28** | **~7 h** | ~9 h |
| 50 | 350 | 150 | ~$47 | ~12 h | ~15 h |

¹ L4 all-in ~$1/h × cell-hours. ² Modal cap=4 containers. ³ A100, ~4 parallel, sim-bound.
Per-task ~40 s (full episode rollout) — **±40%**; the smoke run calibrates this exactly.

**All-4-suites scale-up** (spatial+object+goal+libero_10, 3 seeds, 2 arms = 24 cells):
double the row above (per_cat=30 → ~$55, ~14 h).

## 4 · Run it

**Modal (guarded):**
```bash
cd sib_vla/multivla
# smoke first — 1 task/cat, calibrates per-task time + confirms category names (~$0.5)
BUDGET_GUARD=2  HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run lplus_modal/lplus_modal.py::main --stage smoke
# full 3-seed core (12 cells, per_cat 30)
BUDGET_GUARD=32 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 30
```

**Local A100:**
```bash
cd sib_vla
SEEDS="42 123 456" PERCAT=30 bash run_stage2_liberoplus.sh   # (after seed-loop wiring)
```

## 5 · Deliverable (what the results table looks like)
Per suite, a 7-row category table:

| category | base (mean±CI) | AEGIS (mean±CI) | Δ |
|---|---|---|---|
| camera/viewpoint | … | … | … |
| lighting | … | … | … |
| … (7 rows) | | | |
| **TOTAL** | … | … | … |

…over 3 seeds, for object + goal, with a per-suite gating note. Plus a figure
(`make_readme_figures.py` extended) and the `paper/aegis_draft.md` §LIBERO-Plus block filled.

## 6 · Risks / notes
- **Cost driver = per_cat × 7 × 12 cells.** Start at per_cat=30; the smoke run's per-task
  time tells us if 50 is affordable before committing.
- **Category names** are confirmed by the smoke print (don't hard-code until then).
- **libero_10 (Long)** uses 520 max-steps — if added to suites, set `--max-steps 520` for it.
- Eval-only; identity-init guarantees AEGIS ≥ base per suite after gating.

---
**Status:** Modal cells wired + seed-aware (12-cell 3-seed gen verified). Local seed-loop
wiring is the only remaining bit. Smoke → calibrate → full run.
