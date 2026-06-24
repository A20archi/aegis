# State handoff → Blackwell box (2026-06-24)

Everything obtained so far is saved. This doc is the resume point. We pivoted off Modal
($ budget) the moment a **free Blackwell GPU** became available — run everything there for $0.

## What's saved & where (grab via TeamViewer file transfer)
| File (on `~/Desktop/`) | Size | Contents |
|---|---|---|
| `aegis_handoff.bundle` | 46M | **full repo + git history** incl. commit `9560a67` (AEGIS v2 gate code), 24 measured LIBERO-Plus cell JSONs, SUMMARY, figures, notebook |
| `aegis_checkpoints.tgz` | ~big | 720-wide base ckpt `smolvla_spatial_repro/020000/pretrained_model` + RIB-v1 `rib_on86.pt` (196M) + RASF-v1 `rasf_on86.pt` |
| `aegis_untracked_data.tgz` | 8K | `results/allsuites/` (n=200 clean-SR runs) + `report.csv` (Modal billing) |

Restore on Blackwell:
```bash
mkdir aegis && cd aegis && git clone /path/to/aegis_handoff.bundle .
tar xzf /path/to/aegis_checkpoints.tgz -C sib_vla      # base ckpt + v1 weights
tar xzf /path/to/aegis_untracked_data.tgz              # allsuites + report.csv
```

## ⚠️ Gotcha #1 — Blackwell is `sm_100`; current torch tops out at `sm_90`
The lplus env ships `torch 2.7.1+cu126` whose `arch_list` ends at `sm_90` → **no Blackwell
kernels**. First thing on the Blackwell box, in the eval env:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_arch_list())"   # must include sm_100
```
If stable cu128 still lacks sm_100, use the cu128 **nightly**. Nothing else in the stack changes.

## The work product: AEGIS v2 (commit 9560a67)
Diagnosis from the 3-seed LIBERO-Plus data: AEGIS **wins** on the corruption axes it targets
(Sensor Noise +17…+28, Camera +6…+19, all 4 suites) but **regresses** on the high-baseline
axes (Light −5…−11, Layout, Robot-Init) because the old RIB was **always-on** (global scalar
`tanh(fusion_coeff)`) and its info-loss is pure downside where the base is already strong.

Fix (implemented + CPU-verified):
- **#1 input-adaptive gate** (`sib/robust_ib.py`): per-sample `g(x)∈(0,1)` scales the RIB
  residual. Closed gate ⇒ residual 0 ⇒ **baseline exactly** (verified 9e-14). Identity-at-init
  preserved. `gate_head` saved/loaded in the checkpoint.
- **#2 clean/corrupt supervision** (`scripts/finetune_rib.py`): `--lambda-gate` adds
  BCE(open-on-corrupt / close-on-clean) using the existing per-image `cmask`; sparsity fallback
  if gate↔cmask shapes don't align. Logs `gate[bce]=`.

Target: turn the 4 regressions → ~parity while keeping the wins ⇒ **+6…+9 suite mean** (was
+2…+3.6), with +10 the stretch. The gate can only stop losses, never fabricate wins.

## Resume steps on Blackwell (all free, ~hours)
1. cu128 torch (above). Re-run the CPU gate smoke test to confirm the code survived the move:
   `cd sib_vla && python -c "from sib.robust_ib import FusedRobustIBProjector; print('ok')"`
2. **Train RIB-v2** (cheap on Blackwell). Make a config pointing at the base ckpt + LIBERO data,
   then:
   ```bash
   python scripts/finetune_rib.py --config <cfg>.yaml --steps 15000 \
     --d-z 448 --beta 1e-3 --free-bits 0.1 --corrupt-frac 0.6 --lambda-gate 0.05 --tag rib_v2_gate
   ```
   Watch `gate[bce]` separate (clean→~0.1, corrupt→~0.9) and `task=` flatten. 15k for margin;
   checkpoints every 3k so you can eval an early one.
3. **Cheap validation first** — Light + Sensor + Camera on Object+Goal, against the existing
   baseline JSONs (already measured). Go/no-go: did Light move toward 0 *without* losing Sensor?
   `scripts/libero_plus_aegis_eval.py --method aegis --ckpt <base> --rib-weights results/rib_v2/rib_v2_gate.pt --suite libero_object --cats "Sensor Noise" "Camera Viewpoints" "Light Conditions" --per-cat 6 --out ...`
4. **If validation passes** — full sweep, AEGIS arm only (baseline is on disk):
   ```bash
   CONC=16 ARMS=aegis SEEDS=42,123,456 RIB=results/rib_v2/rib_v2_gate.pt python scripts/run_local_lplus.py
   ```
   (edit `run_local_lplus.py` RIB path + ENV/paths for the Blackwell box; CONC can go high — 80GB+).
5. Rebuild SUMMARY + figure from the new JSONs (`scripts/build_results_summary.py`,
   `scripts/make_robustness_figure.py`).

## Honest results state at handoff (don't lose this)
- **Clean SR, per-suite gated (n=200):** Object 97.5 (gate closed=base), Goal 91.5→93.5 (+2.0),
  Spatial 80.5→85.5 (+5.0), Long 64.5 (gate closed=base). Avg 83.5→85.25 (+1.75), 0 regressions.
- **LIBERO-Plus per-perturbation (3-seed mean), AEGIS-open:** wins Sensor/Camera/Language(O,G);
  losses Light(all), Layout/Init(mixed). Per-suite means +3.6/+3.6/+2.4/+0.4. Raw cell JSONs in
  `results/local_lplus/<suite>/seed<N>/`.
- Modal is retired (pivoted to free Blackwell). Billing: $190.34 spent, budget eaters were
  `smolvla-robust` training/clean-SR ($151), not eval.

## Cost analysis (now moot, kept for the record)
LIBERO-Plus eval is cheap *if* run as **K concurrent rollouts per container** (~$5 full 3-seed on
T4) vs one-process-per-container (~$25, the historical $13–21 runs). On Blackwell this is all free.
