# KICKSTART — run this the moment compute is available

Single source of truth for resuming work. Two compute paths; pick whichever frees first.
Everything below is **staged and verified** (scripts syntax-checked, checkpoints present,
cell generators tested). Nothing here needs setup — just run.

State at staging (2026-06-22): Modal $98.31/$100 (frozen, awaiting reset). A100 busy with
hopfield/hamlet. Grid = 10 pairs (mean +29.9, 0 regr). All harnesses ready.

---

## PATH A — Modal cap was reset  → launch ONLY via the guarded wrapper

**NEVER bare `modal run`.** The guard kills stray clients on any exit + auto-aborts on spend.
Set `HARD_TOTAL` to (new cap − 5) and `BUDGET_GUARD` to the per-run ceiling.

```bash
cd sib_vla/multivla
# 0. sanity: confirm clean slate + real spend
bash modal_killall.sh

# 1. *** TOP PRIORITY *** MULTI-SEED across the LIBERO suites — extra seeds 123,456 ×
#    {object,goal,long} × 6 axes × {base,aegis} = 72 cells (+seed-42 grid) -> mean ± CI  (~$27)
BUDGET_GUARD=32 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage multiseed --episodes 10

# 2. close the grid tail (2 pairs) — resume-skip protects the 10 done            (~$2-3)
BUDGET_GUARD=6  HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage stage1 --episodes 10

# 3. LIBERO-Long robustness (seed 42 single) — 12 cells, 520-step               (~$2-4)
BUDGET_GUARD=8  HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage long --episodes 10

# 4. ablation suite — Spatial, 8 arms × 6 axes = 48 cells, eval-only             (~$12-17)
BUDGET_GUARD=20 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage ablation --episodes 10
```
> Note: the multi-seed stage runs all 3 seeds FRESH (incl. 42) into `seed<N>/` subdirs, so
> it is self-contained — it already covers Object/Goal/Long; steps 2–3 are only needed for the
> original no-subdir seed-42 grid/Long if you still want those. Multi-seed alone gives the headline.
Results land on volume `smolvla-assets`; pull with `modal volume get smolvla-assets results_modal/... .`

GR00T (Path A) and LIBERO-Plus cells are **not yet wired** (need the real model/repo interface)
— build them on the A100 (Path B) where they'll run anyway, or after a cheap CPU `validate`.

---

## PATH B — dedicated A100 is free  → $0, self-driving

`run_master_queue.sh` **waits for the GPU to free (≥68 GB), protects hopfield/hamlet, runs
stages 1→3 serially with an OOM watchdog that only ever kills MY jobs.** Fire-and-forget:

```bash
cd sib_vla
# *** TOP PRIORITY *** multi-seed across LIBERO suites (object,goal,long; extra seeds 123,456)
SEEDS="123 456" SUITES="object goal long" EP=20 bash run_multiseed.sh

# then the rest — full self-driving chain (holds until card frees; stage1 -> Long -> LIBERO-Plus -> ablations):
nohup bash run_master_queue.sh > results/master_queue.log 2>&1 &
#   tune: START_FREE=68000 (MiB free that signals external jobs cleared), HARDCAP=1 (serial)

# multi-seed (run after stage1, or standalone) — seeds 123,456, n=200
SEEDS="123 456" EP=20 bash run_multiseed.sh
```
Local results: `results/liberov_objgoal/`, `results/liberoplus/`, `results/ablations/`,
`results/liberov_objgoal/*/seed<N>/`.

---

## Priority order (either path)

| # | task | path A cost | path B time¹ | gating? |
|---|---|---:|---:|---|
| **1** | **Multi-seed across LIBERO suites** (72 cells: seeds 123,456 × {obj,goal,long}) | **$27** | **~1 day** | **TOP — the headline statistical result** |
| 2 | Grid tail (2 pairs, seed-42) | $2–3 | ~0.5 h | only for original no-subdir grid |
| 3 | LIBERO-Long (seed-42 single, 12 cells) | $2–4 | ~3 h | subsumed by #1 |
| 4 | Ablation suite (48 cells) | $12–17 | ~12 h | reviewer-critical |
| 5 | GR00T-N1.5 reproduce + AEGIS | $19–26 | ~15 h | needs harness (build on A100) |
| 6 | LIBERO-Plus sweep | $15–28 | ~14 h | needs cells (build on A100) |

¹ dedicated A100, ~4 parallel sim-evals. **#1 multi-seed alone is the headline; #1+#4 (ablations)
+ #5 (GR00T) ≈ the full paper-core.**

---

## After results land (any path) — $0 writing

1. `python sib_vla/scripts/make_readme_figures.py` — regenerate figures with new numbers.
2. Fill `paper/aegis_draft.md` `[PENDING]` blocks: §5.5 multi-seed, §5.6 GR00T, §6 ablation table.
3. Update README tables + `SAVED_STATE.md`.
4. Commit + push.

---

## Inventory — everything that must exist (all ✅ verified at staging)

**Launchers/guards:** `sib_vla/multivla/{safe_modal_run.sh, modal_killall.sh}` ·
`sib_vla/{run_master_queue.sh, run_queue_core.sh, run_stage1_liberov_objgoal.sh,
run_liberolong.sh, run_stage2_liberoplus.sh, run_stage3_ablations.sh, run_multiseed.sh,
run_libero_plus_aegis.sh}`

**Modal app stages:** `smolvla_modal.py::main --stage {validate,smoke,stage1,long,videos,ablation,multiseed}`

**Eval:** `sib_vla/scripts/eval_libero_v.py` (now with `--seed` + ablation de-strength flags)

**Checkpoints:** `outputs/smolvla_spatial_repro/checkpoints/020000/pretrained_model` ·
`results/{ib_on86/rib_on86.pt, rasf_on86/rasf_on86.pt, raw_vib_b1e-3.pt, gain_no_rate.pt, ib_on86/ib_on86.pt}`

**Docs:** `paper/aegis_draft.md` · `final_module_architecture.md` · `SAVED_STATE.md` · this file
