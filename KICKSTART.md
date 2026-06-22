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

# 1. close the grid tail (2 pairs) — resume-skip protects the 10 done       (~$2-3)
BUDGET_GUARD=6  HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage stage1 --episodes 10

# 1b. LIBERO-Long robustness — same 6 axes × {base,aegis} = 12 cells, 520-step (~$2-4)
BUDGET_GUARD=8  HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage long --episodes 10

# 2. ablation suite — Spatial, 8 arms × 6 axes = 48 cells, eval-only          (~$12-17)
BUDGET_GUARD=20 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage ablation --episodes 10

# 3. multi-seed — seeds 123,456 on the headline grid = 48 cells              (~$14-18)
BUDGET_GUARD=22 HARD_TOTAL=<cap-5> ./safe_modal_run.sh \
    modal run smolvla_modal/smolvla_modal.py::main --stage multiseed --episodes 10
```
Results land on volume `smolvla-assets`; pull with `modal volume get smolvla-assets results_modal/... .`

GR00T (Path A) and LIBERO-Plus cells are **not yet wired** (need the real model/repo interface)
— build them on the A100 (Path B) where they'll run anyway, or after a cheap CPU `validate`.

---

## PATH B — dedicated A100 is free  → $0, self-driving

`run_master_queue.sh` **waits for the GPU to free (≥68 GB), protects hopfield/hamlet, runs
stages 1→3 serially with an OOM watchdog that only ever kills MY jobs.** Fire-and-forget:

```bash
cd sib_vla
# self-driving: holds until the card frees, then
#   stage1 (V object+goal) -> LIBERO-Long -> stage2 (LIBERO-Plus) -> stage3 (ablations)
nohup bash run_master_queue.sh > results/master_queue.log 2>&1 &
# or just the Long robustness sweep on its own:
EP=20 bash run_liberolong.sh
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
| 1 | Grid tail (2 pairs) | $2–3 | ~0.5 h | — |
| 1b | LIBERO-Long robustness (12 cells, 520-step) | $2–4 | ~3 h | long-horizon stress test |
| 2 | Ablation suite (48 cells) | $12–17 | ~12 h | reviewer-critical |
| 3 | Multi-seed (48 cells) | $14–18 | ~10 h | **#1 reviewer ask** |
| 4 | GR00T-N1.5 reproduce + AEGIS | $19–26 | ~15 h | needs harness (build on A100) |
| 5 | LIBERO-Plus sweep | $15–28 | ~14 h | needs cells (build on A100) |

¹ dedicated A100, ~4 parallel sim-evals. **Paper-core = #1–4 ≈ 2 days / ~$47–61.**

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
