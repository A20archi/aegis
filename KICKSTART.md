# KICKSTART — run this the moment compute is available

Single source of truth for resuming work. Two compute paths; pick whichever frees first.
Everything below is **staged and verified** (scripts syntax-checked, checkpoints present,
cell generators tested). Nothing here needs setup — just run.

State at staging (2026-06-22): Modal $98.31/$100 (frozen, awaiting reset). A100 busy with
hopfield/hamlet. Grid = 10 pairs (mean +29.9, 0 regr). All harnesses ready.

> ### ⭐ ACTIVE SCOPE (current decision): only TWO tasks, both EVAL-ONLY (no re-training)
> 1. **Multi-seed** (extra seeds 123,456 × {object,goal,long}) — 72 cells
> 2. **LIBERO-Plus eval** (object+goal × {baseline,aegis}) — 4 cells
>
> Everything else below (grid tail, Long-only, ablations, GR00T, WidowX) is **deferred** —
> kept as reference but NOT in the active run. No checkpoints are trained; all runs reuse
> `smolvla_spatial_repro` + `rib_on86`/`rasf_on86`.

---

## PATH A — Modal cap was reset  → launch ONLY via the guarded wrapper

**NEVER bare `modal run`.** The guard kills stray clients on any exit + auto-aborts on spend.
Set `HARD_TOTAL` to (new cap − 5) and `BUDGET_GUARD` to the per-run ceiling.

```bash
cd sib_vla/multivla
# 0. sanity: confirm clean slate + real spend
bash modal_killall.sh

# === ACTIVE (the only two we run) ===
# 1. MULTI-SEED across LIBERO suites — extra seeds 123,456 × {object,goal,long} = 72 cells (~$27)
BUDGET_GUARD=32 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage multiseed --episodes 10

# 2. LIBERO-Plus eval — object+goal × {baseline,aegis}, per-cat 50, eval-only       (~$15-28)
BUDGET_GUARD=30 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run lplus_modal/lplus_modal.py::main --stage stage1 --per-cat 50

# === DEFERRED (reference only — NOT in the active run) ===
#   stage1 (grid tail) · long · ablation  -> see git history / prior KICKSTART revisions
```
> Multi-seed combines the 2 extra seeds with the existing seed-42 grid for a 3-seed mean ± CI.
> LIBERO-Plus smoke first if unsure: `... lplus_modal.py::main --stage smoke` (1 task/cat, ~$0.5).
Results land on volume `smolvla-assets`; pull with `modal volume get smolvla-assets results_modal/... .`

LIBERO-Plus Modal cells are now **wired** (`lplus_modal.py::main --stage {validate,smoke,stage1}`,
eval-only). GR00T is **deferred** (out of active scope; needs a harness + training, which we're
not doing now).

---

## PATH B — dedicated A100 is free  → $0, self-driving

`run_master_queue.sh` **waits for the GPU to free (≥68 GB), protects hopfield/hamlet, runs
stages 1→3 serially with an OOM watchdog that only ever kills MY jobs.** Fire-and-forget:

```bash
cd sib_vla
# ACTIVE TASK 1 — multi-seed across LIBERO suites (object,goal,long; extra seeds 123,456)
#   EP=10 = n=100 (~1 day); EP=20 = n=200 (~2 days)
SEEDS="123 456" SUITES="object goal long" EP=10 bash run_multiseed.sh

# ACTIVE TASK 2 — LIBERO-Plus eval (object+goal × {baseline,aegis}, per-cat 50, eval-only)
PERCAT=50 bash run_stage2_liberoplus.sh
#   (DEFERRED: run_master_queue.sh runs the full chain incl. grid/Long/ablations — not in active scope)

# multi-seed (run after stage1, or standalone) — seeds 123,456, n=200
SEEDS="123 456" EP=20 bash run_multiseed.sh
```
Local results: `results/liberov_objgoal/`, `results/liberoplus/`, `results/ablations/`,
`results/liberov_objgoal/*/seed<N>/`.

---

## Priority order (either path)

**ACTIVE (the only two):**
| # | task | path A cost | path B time¹ | |
|---|---|---:|---:|---|
| **1** | **Multi-seed across LIBERO suites** (72 cells: seeds 123,456 × {obj,goal,long}) | **$27** | **~1 day** (n=100) | headline statistical result |
| **2** | **LIBERO-Plus eval** (object+goal × {base,aegis}, per-cat 50) | **$15–28** | ~14 h | paper-protocol robustness |

**DEFERRED (reference, not running):** grid tail · Long-only · ablations · GR00T · WidowX.

¹ dedicated A100, ~4 parallel sim-evals. **Active scope total ≈ $42–55 / ~1.5–2 days.**

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
