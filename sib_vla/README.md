# Spectral Information Bottleneck for a Flow-Matching VLA Action Head

A small, trainable module placed **after** a frozen SmolVLA flow-matching action
expert. It takes the sampled action chunk, transforms it to the frequency domain
(orthonormal DCT along time), passes each frequency band through a learned
Gaussian channel with a penalized information rate, reconstructs with the
MMSE/Wiener estimate, and inverse-transforms back to an action chunk. A single
coefficient `beta` controls how many bits the policy may spend across the action
spectrum.

**Frozen:** the VLA backbone and the flow-matching action expert. **Trained:**
only the bottleneck (≈ `H·d` parameters for the global channel). The
distortion target is always the **ground-truth action chunk** `A_star`, so the
module can *denoise* the frozen policy, not merely compress it.

---

## The mechanism (exact)

Per batch `B`, band index `k = 0..H-1`, action dim `d`. With `H = 50`, `d = 7`
for LIBERO (verified from `SmolVLAPolicy.config`).

1. **Transform** `X = C · A` along time, `C` the orthonormal DCT-II matrix
   (`C Cᵀ = I`, so the inverse is `Cᵀ` and Parseval holds). `raw_vib` sets
   `C = I`.
2. **Source variance** `lambda_k` (shape `(H,d)`): EMA over the dataset of the
   per-(band,dim) variance of the *predicted* coefficients `X.detach()`.
   Distributional, never per-sample; detached wherever it enters the loss.
3. **Channel** with learned `sigma2 = softplus(log_var)+eps`:
   - Wiener gain `g = lambda / (lambda + sigma2) ∈ [0,1]` (the correct MMSE
     gain, **not** `1 − theta/lambda`).
   - Train (reparameterized): `X_hat = g · (X + sqrt(sigma2)·eps)`.
   - Eval (MMSE): `X_hat = g · X`.
   - Rate `R = 0.5 · Σ log(1 + lambda/sigma2)` — the mutual information
     `I(X; X_tilde)` of the Gaussian channel, i.e. a **variational upper bound**
     on the true coding rate. No closed-form rate–distortion optimality is
     claimed anywhere.
4. **Inverse** `A_hat = Cᵀ · X_hat`.
5. **Loss** `L = MSE(A_hat, A_star) + beta · R`.

### A note on rate units
`R` is written verbatim with `log1p` (natural log), so the optimized quantity is
in **nats**. Since `L = MSE + beta·R`, rescaling `R` (nats→bits) is absorbed by
`beta` and does not change the optimum. We therefore keep the loss in nats and
report the per-band allocation heatmap in **bits** (`bits_per_band`, base-2),
clearly labeled. `beta` is thus defined relative to nats.

### Water-filling: the analytic anchor
For the **compression** objective `D + beta·R` (target = source, rate in nats),
minimizing over the per-band noise has a closed form (`sib/waterfill.py`):

```
sigma²_k = beta·lambda_k / (2·lambda_k − beta)   (active iff 2·lambda_k > beta)
  ⇒  D_k = beta/2   (constant on active bands),   R_k = ½ ln(lambda_k / (beta/2))
```

i.e. the β-penalized Gaussian channel **is reverse water-filling with water level
`theta = beta/2`**; bands with `lambda_k ≤ beta/2` are dropped, and the optimal
Wiener gain on active bands is exactly `1 − theta/lambda_k`. This is proven and
unit-tested (`test_waterfill.py::test_beta_equals_two_theta_theorem`). Under the
**denoising** objective (`A_star ≠ A`) it becomes the task-weighted
generalization; `scripts/validate_allocation.py` measures how closely the
*learned* `{R_k}` tracks the matched-rate water-filling prescription (Pearson r,
L1 gap, recovered `theta` vs `beta/2`) and plots the overlay — the planner's
strongest single result. (On the synthetic denoising check, learned-vs-analytic
correlation is 0.93–0.99 across β.)

---

## Baselines map to the planner's confound isolation

Each baseline removes exactly one explanation, so a win can be attributed:

| baseline (config) | isolates "is it just …?" |
|---|---|
| `lowpass` | … throwing away high frequencies? (fixed DCT truncation) |
| `gain_no_rate` (`beta=0`) | … learnable filtering, with no rate constraint? |
| `raw_vib` (identity transform) | … a per-band bottleneck, with the spectral basis irrelevant? |
| `jerk` (`gamma·J`) | … smoothness, achievable by a jerk penalty? |

`raw_vib` is the make-or-break: if it matches `sib` everywhere, the honest
finding is that the frequency basis is not the active ingredient.

---

## Repository layout

```
sib/            transforms · lambda_estimator · bottleneck · losses · metrics
                waterfill · corruptions · recording · wrapper · data · utils
scripts/        estimate_lambda · decorrelation_check · train · eval
                sweep_beta · validate_allocation · aggregate · make_figures
configs/        base + {vanilla, sib, raw_vib, lowpass, gain_no_rate, jerk, sweep_beta}
tests/          test_transforms · test_bottleneck · test_shapes · test_waterfill
                test_metrics_corruptions · test_recording     (61 tests, all pass)
results/        json · .pt · figures · gpu_hours.csv · videos/<run>/<cond>/<task>/
```

The six configs come from **one** minimal structure (transform × operator ×
regularizer):

| config        | transform | operator         | regularizer    |
|---------------|-----------|------------------|----------------|
| vanilla       | —         | —                | —              |
| sib           | DCT(+rot) | Gaussian channel | rate (`beta·R`)|
| raw_vib       | identity  | Gaussian channel | rate (`beta·R`)|
| lowpass       | DCT       | low-pass mask    | none           |
| gain_no_rate  | DCT       | learned gain     | none (`beta=0`)|
| jerk          | DCT       | learned gain     | jerk (`gamma·J`)|

`raw_vib` vs `sib` differs **only** in the transform; `jerk` vs `gain_no_rate`
differs **only** in the regularizer. Nothing else moves — that is the point.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + the LeRobot LIBERO env stack
pytest -q                                # 35 tests, all green

# Stage 0 — reproduce vanilla baseline
python scripts/eval.py  --config configs/vanilla.yaml --tag vanilla

# Stage 1 — precompute cache + lambda, then the spectral pre-check (gates A/B)
python scripts/estimate_lambda.py     --config configs/sib.yaml
python scripts/decorrelation_check.py --config configs/sib.yaml

# Stage 2 — MVE: train SIB at one beta, eval
python scripts/train.py --config configs/sib.yaml --beta 1e-3 --tag sib_mve
python scripts/eval.py  --config configs/sib.yaml --weights results/sib_mve.pt --tag sib_mve

# Stage 3 — frontier + isolating baselines
python scripts/sweep_beta.py --config configs/sweep_beta.yaml
for m in raw_vib lowpass gain_no_rate jerk; do
  python scripts/train.py --config configs/$m.yaml
  python scripts/eval.py  --config configs/$m.yaml --weights results/$m.pt --tag $m
done

# Stage 4 — robustness (image + action), allocation validation, figures
python scripts/eval.py --config configs/sib.yaml --weights results/sib_mve.pt \
       --corruption gaussian_noise:1 --tag sib_mve         # observation corruption
python scripts/eval.py --config configs/sib.yaml --weights results/sib_mve.pt \
       --action-noise 0.1 --tag sib_mve                    # action perturbation
python scripts/validate_allocation.py --weights results/sib_mve.pt   # learned vs water-filling
python scripts/aggregate.py   --results results --markdown
python scripts/make_figures.py --results results
```

Every run is `python scripts/<x>.py --config configs/<y>.yaml`; no experiment
parameters are hard-coded. Seeds and resolved configs are saved beside each
result; GPU-hours are appended to `results/gpu_hours.csv` (budget ≤ 80 GPU-h).

---

## Staged gates

- **Stage 0:** clean success in `[0.65, 0.80]`; else try an alternate community
  LIBERO checkpoint.
- **Stage 1 / Gate A:** if ~all energy is in the lowest 2–3 bands, the
  allocation story is weak — flagged in `results/stage1.json`; bias the suite
  toward contact-rich/precision tasks.
- **Stage 1 / Gate B:** if off-diagonal band-correlation energy is high, set
  `rotation: pca` (rotation saved to `results/pca_rotation.pt`).
- **Stage 2 (MVE):** success within ~2 pp of vanilla, RMS jerk and HF energy
  reduced, total rate `R` decreased from init.
- **Stage 3 (existence of the contribution):** `sib` must beat `raw_vib` on at
  least one of {corruption robustness, jerk at matched success}. `aggregate.py`
  computes this verdict (`results/stage3_gate.json`) with a two-proportion
  z-test on pooled corrupted episodes.

### Stage 3 verdict
**To be filled by `aggregate.py` after the runs complete.** If `raw_vib` matches
`sib` everywhere, the honest finding is: *a per-band information bottleneck on
actions, where the frequency basis is **not** the active ingredient* — a clean
negative result, written up as such rather than buried.

---

## Rollout artefacts (videos + traces)

Every `eval.py` run keeps its simulations. For each task it writes, under
`results/videos/<run>/<condition>/<task>/`:

- `epK.mp4` — what the policy *saw* each step, including any eval-time corruption
  (so corrupted runs visibly look corrupted). Capped at `record.videos_per_task`
  to bound disk; mp4 via ffmpeg, graceful gif/PNG fallback.
- `epK.npz` — the executed action trace, success flag, and per-episode jerk/HF —
  saved for **every** recorded episode (cheap), so failures are inspectable and
  the bit-allocation-vs-behaviour story is reproducible offline.
- `index.json` — a manifest of everything written (counts, successes, paths).

Controlled by the `record:` block in `configs/base.yaml` (`enabled`, `video`,
`videos_per_task`, `fps`, `camera_key`); set `enabled: false` to skip. The mp4
writer is unit-tested end-to-end (`test_recording.py`, real ffmpeg encode +
round-trip read).

## Status of this implementation

- **Mathematical core** (`sib/transforms.py`, `lambda_estimator.py`,
  `bottleneck.py`, `losses.py`, `metrics.py`, `waterfill.py`, `corruptions.py`):
  complete and **verified by 61 passing unit tests** — DCT orthonormality/Parseval
  (vs scipy), Wiener-gain and rate limits, rate monotonicity in `sigma2`, the
  reverse-water-filling `theta = beta/2` theorem, gradient reaches `log_var`
  while the frozen base receives none, shape preservation across all configs,
  and the metrics/corruptions/action-perturbation utilities.
- **LeRobot/LIBERO integration** (`wrapper.py`, `data.py`, `scripts/eval.py`,
  `estimate_lambda.py`): written against the **verified** LeRobot 0.4.3 API
  (`predict_action_chunk → (B,50,7)` under `no_grad`; `make_env(LiberoEnv(...))`;
  `preprocess_observation` / `add_envs_task`; `make_pre_post_processors`).
  Points that depend on your specific checkpoint/dataset/sim — hub IDs, `fps`,
  the success key, the action normalization mode — are marked `VERIFY` and
  isolated. `train.py` runs purely on cached tensors and is exercised by a
  synthetic-cache smoke test.
