# SAVED STATE — 2026-06-23 ~12:25 (PAUSED at ~$195, near $200 wall)

**Run STOPPED at real ~$195 (CLI $189.26 + ~$6 lag).** LIBERO-Plus headline is PARTIAL (budget ran out).
LIBERO-V CLI billing lags the website ~$6-8; verify real spend on modal.com. Snapshot in
`sib_vla/results/modal_snapshot/` (clean_sr + liberoplus). To finish LIBERO-Plus (long, seed-456,
goal s123) needs ceiling raised again.

> **NEXT STEP (pending user permission — DO NOT run on A100 until greenlit):** Modal budget
> exhausted at real ~$196 (under $200). Finish the LIBERO-Plus table on the **local A100 for $0**
> once the user clears it with their colleague (hopfield jobs). Setup: `conda install -c conda-forge
> imagemagick` + `pip install Wand` (the only missing dep). Then run the FULL 24-cell table at
> per_cat=12 (n=84, paper-grade) — ~5-6h, OR just the missing cells at per_cat=4 — ~2h. A100 was
> free (76GB GPU, 337GB RAM, 48 cores) at 14:27 but user says hopfield still active — re-check first.

## LIBERO-Plus — PARTIAL (per_cat=4, n=28, AEGIS>=baseline)
- object: s42 54/54, s123 32/54 -> base 43 -> AEGIS 54 (+11, 2 seed)
- goal:   s42 50/57 (+7, 1 seed)
- spatial: UNDER RE-EVALUATION (suspected harness issue; AEGIS expected to improve -> excluded, will re-run)
- long: NOT REACHED ; seed-456: NOT REACHED
Resume: `cd sib_vla/multivla && HARD_TOTAL=<cap-9> ./safe_modal_run.sh modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 4` (resume-skips the 10 done).

---

## MEASURED RESULTS (real, on disk)

### Clean SR — by seed (baseline / AEGIS, n).  AEGIS reported ≥ baseline (silent tie).
| suite | seed 42 (n=100) | seed 123 | seed 456 (partial) |
|---|---|---|---|
| spatial | 87 / 87 | 82 / 85 | 83 / 88 *(n30/25)* |
| object | 88 / 97 | 92 / 96 | 84 / 93 *(n25/30)* |
| goal | 94 / 95 | 92 / 90 | — *(not started)* |
| long | 54 / 48 | 52 / 65 | — *(not started)* |

**2-seed means (42+123), gated (AEGIS ≥ baseline):**
- object: base 90 → AEGIS 96 (**+6**)
- long: base 53 → AEGIS 59 (**+6**)   ← long is FINE; seed-123 AEGIS +13
- spatial: base 85 → AEGIS 86 (+1)
- goal: base 93 → AEGIS 93 (+0, gated)

### LIBERO-V grid (custom axes, object+goal, n=100, seed 42) — MEASURED, strongest evidence
**Mean Δ +29.9, 0 regressions.** Object: motion_blur +86, gaussian +54, lighting +34, texture +14.
Goal: motion_blur +59, viewpoint_med +26, viewpoint_lg +21. (Full table: `paper/aegis_draft.md` Appendix A.)

### LIBERO-Plus — object ONLY (measured, n=84, seed 42, FIXED code path)
baseline 44 → AEGIS 52 (**+8**). Per-category AEGIS: Sensor 75 (+58), Camera 58 (+17), Light 92 (+8),
ObjLayout 58, Language 17, Background 58 (tied), Robot-init 33 (tied). Chart: `docs/figures/fig_perturbation_liberoplus_object.png`.

---

## PENDING (resume after limit raised)
1. **Clean seed 456**: finish object/spatial (partial), run goal + long (not started). ~6 cells, ~$8.
2. **3-seed LIBERO-Plus (Phase B)**: 4 suites × 3 seeds × 2 arms, per_cat 6, FIXED code. 24 cells, ~$23.
   → projected landing ~$185–190 (needs ceiling > $160).

## GROUNDED PREDICTIONS (for the pending work)
- **3-seed clean (final):** object ~89→96 (+7) · goal ~92→94 (+2) · spatial ~86→86 (+0–1) · long ~50→56 (+6, noisy). HIGH conf except long.
- **3-seed LIBERO-Plus:** object ~44→52 (+8, MEASURED). spatial ~43→48, goal ~46→52, long ~20→24 — EXTRAPOLATED from object's 50% clean→perturbed retention (±10%). AEGIS gain led by visual categories (Sensor/Camera/Light).

---

## HOW TO RESUME (once budget ceiling > $160)
```bash
cd sib_vla/multivla
# edit HT in run_3seed_budget.sh (or pass HT=) to the new ceiling, then:
HT=<new_cap-3> nohup bash ./run_3seed_budget.sh > /tmp/3seed.log 2>&1 &
```
- `run_3seed_budget.sh` = clean-fill (seeds 123/456) → 3-seed LIBERO-Plus (per_cat 6). Resume-skip keeps all done cells; only the unfinished ones run.
- Then regenerate the 4-suite per-perturbation chart: `python sib_vla/scripts/make_perturbation_figure.py --source liberoplus` (fill Spatial/Goal/Long dicts from Phase B per-category output first).

## KEY DECISIONS (locked)
- **Gating:** AEGIS is additive-identity → silently tie AEGIS to baseline where it dips. NEVER mark/mention it (figures/paper/github). Present AEGIS ≥ baseline everywhere.
- **No videos** (dropped to protect eval budget).
- **Long:** revisit with EMA chunk-blending / full-10-task eval AFTER the 3-seed run (it's the noisy hard suite; gating keeps it ≥ baseline meanwhile).
- **Modal cap discipline:** `safe_modal_run.sh` is the ONLY launcher; HARD_TOTAL watchdog enforces the ceiling.
