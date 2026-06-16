# updated.md — Unified Spectral IB for VLAs (Week-2+ plan)

> **Status:** plan of record for next week onward. Do NOT start implementing the
> perception leg yet — the Week-1 action-leg runs are still completing. This file
> is the spec; the handoff block below grounds it in what Week 1 actually found so
> §1 ("what changes vs project.md") is based on measured results, not assumptions.

---

## Handoff — Week-1 action-leg results (measured, n=200/eval unless noted)

Baselines (LIBERO-Spatial, HuggingFaceVLA/smolvla_libero, frozen):
- **MAX** vanilla closed-loop (n_action_steps=1): **0.720** [0.654, 0.778], jerk 0.749
- **MIN** vanilla open-loop (n=25): **0.570** [0.501, 0.637], jerk 0.409
- Cadence curve: n=1→0.720, n=5→0.605, n=10→0.610, n=25→0.570 (reactivity payoff
  concentrated at n=1; the n=1→n=25 gap is a reactivity ceiling no chunk method
  can cross).

Action-leg SIB (DCT + Gaussian channel + rate), clean:
- **Leg 1 POSITIVE** — RD frontier with a flat-success plateau. β=1e-4: succ
  **0.595** (≥ MIN) at jerk **0.111 (−73% vs MIN)**, HF energy **−90%**. β rises →
  success falls (1e-3→0.535, 1e-2→0.335): clean monotonic frontier.
- **"DCT matters" POSITIVE** — SIB ≫ raw_vib (no DCT): 0.535 vs 0.360, jerk 0.11
  vs 0.37. Spectral basis is a real active ingredient.
- **S3.D caveat** — on CLEAN data SIB ≈ lowpass ≈ gain_no_rate (~0.53–0.55, jerk
  ~0.10–0.11). The rate machinery does not beat a fixed low-pass on clean.
- **Visual-corruption robustness FLAT/negative** — SIB ≤ vanilla under blur/
  brightness. Diagnosed mechanistically: visual corruption causes *semantic*
  errors (smoothly-wrong action chunks), which an action-output smoother cannot
  fix. This flatness is *expected* and is the motivation for the perception leg.

Week-1 follow-ons added (results pending / in flight, do not rebuild):
- `preserve_energy` (magnitude-preserving SIB) — RMS restored 0.876→1.0. Used in
  the cadence sweep; rescues low-n undershoot only partially (n=1 stays broken;
  the heavy low-pass kills closed-loop reactivity regardless of magnitude).
- `adaptive_sigma` (input-adaptive σ: estimate noise floor from signal-free high
  bands, suppress harder when noisier) + `train_action_noise` (denoiser training).
  Built, unit-tested, running on the **action-noise** robustness axis — the
  mechanism-matched test where the action SIB *should* win and where adaptive-σ
  can beat a fixed low-pass. These are action-leg robustness probes, distinct from
  the perception leg below.

Budget: **~37 GPU-h of 80 spent** by end of Week 1. ~43 GPU-h remain. The full
perception-leg budget map (§9) sums to ~70 GPU-h — MORE than remains — so the
realistic path is the **minimum-viable unified paper** (§9 last paragraph):
P0–P2 + `vis_raw_vib` + `vis_ibadapter` + the existing action leg + one joint P4.
Treat the §9 table as aspirational; gate spending on `results/gpu_hours.csv`.

Code already in place to reuse (do not re-architect): `sib/transforms.py`
(orthonormal DCT-II, Parseval-tested), `sib/bottleneck.py` (`GaussianChannel`,
`wiener_gain`, `LambdaEstimator`, `SpectralActionModule` with `preserve_energy` +
`adaptive_sigma`/`noise_band_frac` flags), `scripts/train.py` (warm-start λ,
`train_action_noise`), `scripts/eval.py` (`--tasks`, `--n-action-steps`,
`--action-noise`, `--corruption`, per-task build, recording). The 2D path in §2
extends these; the channel math is reused verbatim.

---

# project_unified.md — A Unified Spectral Information Bottleneck for VLAs

Extends the existing `project.md` (action-locus spectral IB). Same backbone
(`lerobot/smolvla_base`, frozen), same benchmark (LIBERO), same budget discipline.
This document adds the **perception locus** and the **unification**. The action
locus is unchanged; do not re-architect it.

Implement in the staged order. Each stage has a gate; honour its fallback.

---

## 0. Intent

One per-frequency-band rate-distortion instrument, applied to **two spectra**:

- **Action locus (existing):** 1D temporal DCT of the sampled action chunk. Buys
  smoothness + the interpretable temporal bit-allocation. Already built; keep as-is.
- **Perception locus (new):** 2D spatial DCT of the frozen vision encoder's patch
  tokens, at the modality-alignment interface. Buys visual-corruption robustness.

The paper claim is one principle, two spectra: a single `R_k = ½·log(1 + λ_k/σ_k²)`
budget governs both, and the learned allocation is interpretable on each. The
dissociation (smoothness vs robustness) lives *inside our own mechanism*, not
between our method and a borrowed one.

The novelty is unchanged and amplified: per-band rate allocation as the active
ingredient. StableVLA (channel-covariance sigmoid gating, no rate term, no
spectral transform) becomes a baseline we beat at the identical insertion point.

---

## 1. What changes vs `project.md`

- Action leg: **no change.** It already ties vanilla on success, "DCT matters"
  holds (beats raw_vib), robustness flat — that flatness is *expected* and is the
  motivation for the perception leg, not a failure to fix here.
- Perception leg: new module, new training path, new baselines, new figures.
- Unification: shared rate framework, two betas, joint-run figures.

Hard constraints carried over: backbone + action expert **frozen**; do **not**
backprop through the ODE sampler; distortion target is ground-truth `A_star`;
`lambda_k` distributional (EMA), never per-sample; strictly clean-train (no
augmentation, no corruption exposure during training).

New constraint (perception leg only): the perception module is upstream of the
sampler, so it trains on the **flow-matching training loss** (single forward,
sampled timestep + noise, gradients pass through the frozen expert to the
adapter). This is a single forward pass, NOT the ODE sampler, so it honours the
no-sampler-backprop rule in both letter and budget.

---

## 2. Perception-locus mechanism — exact, with shapes

### 2.1 Where it sits

After the frozen vision encoder, on the patch tokens, before they condition the
action expert. StableVLA's empirical finding is that corruption vulnerability
concentrates at this vision→LLM projection; we place our spectral budget there.

**First build step (verify, do not assume):** inspect the real LeRobot SmolVLA
forward. Locate the visual-token tensor feeding the policy. Record:
- token count `N`, feature dim `D`, dtype
- the spatial grid `(Hp, Wp)` with `Hp*Wp = N` (account for any pixel-shuffle /
  token-merging in SmolVLM2 — the grid mapping may not be a naive reshape)
- whether a clean residual stream is available to interpolate against
Write all of this into `sib/vision_wrapper.py` as a docstring. Gate P0 below
depends on a recoverable spatial grid.

### 2.2 2D spatial transform (`sib/transforms.py`, extend existing)

Reshape tokens to the spatial grid and apply a separable orthonormal DCT-II along
both spatial axes, per feature channel:

```python
# X_v: (B, N, D) -> grid (B, Hp, Wp, D)
g = X_v.reshape(B, Hp, Wp, D)
# spatial DCT: contract Ch over Hp, Cw over Wp
Xs = torch.einsum('uh, bhwd -> buwd', Ch, g)   # rows
Xs = torch.einsum('vw, buwd -> buvd', Cw, Xs)  # cols
# Xs indexed by 2D spatial band (u, v) per channel d
```

`Ch (Hp,Hp)`, `Cw (Wp,Wp)` are the same orthonormal DCT-II matrices as the action
leg, registered as buffers. Inverse uses transposes. Parseval/round-trip tested.

### 2.3 Per-spatial-band channel (`sib/bottleneck.py`, extend existing)

Reuse the existing channel verbatim, now indexed by 2D band `(u,v)` and channel `d`:

```python
sigma2 = softplus(log_var) + eps          # (Hp, Wp, D)
gain   = lam / (lam + sigma2)             # Wiener gain in [0,1]
# inference: X_hat = gain * Xs   (drop noise)
# training:  X_hat = gain * (Xs + randn_like(Xs)*sigma2.sqrt())
R_v = 0.5 * torch.log1p(lam / sigma2).sum()
```

`lam` is the EMA per-(spatial-band, channel) variance of the clean predicted
visual coefficients (detached), warm-started like the action leg.

**Critical init:** initialize `log_var` so `sigma2 -> 0`, hence `gain -> 1`. At
init the module is the identity, so clean success is preserved out of the box.
The rate penalty then pushes down only the gains of bands that don't earn their
bits — on clean data those are the low-energy nuisance spatial bands, which is
exactly where injected noise lives. Corruption robustness emerges from clean
training; we never train on corruption.

The Wiener gain **acts on the main feature path** (not an additive residual), so
it can actually attenuate corrupted bands. Identity-at-init keeps that safe.
Optional global safety blend `tanh(g_gate)` (init 0) only if P2 shows clean
regression; default off.

### 2.4 Inverse + reinjection

Inverse 2D DCT back to `(B, N, D)`, return to the frozen policy in place of the
original tokens. Assert the frozen base receives no gradient; only the perception
module's `log_var` (+ optional context head, + optional `g_gate`) trains.

### 2.5 Training signal

Flow-matching training loss on the frozen expert:

```python
# one forward, no sampler
t   = sample_timestep(B)
eps = torch.randn_like(A_star)
a_t = interpolate(A_star, eps, t)         # rectified-flow / CFM interpolant
v   = frozen_expert.velocity(a_t, t, cond=perceived_tokens)
L   = mse(v, target_velocity(A_star, eps, t)) + beta_v * R_v
```

Verify SmolVLA's exact interpolant + velocity target from the codebase; match it.
Gradients flow `L -> perceived_tokens -> perception module`. Frozen expert passes
gradient, updates nothing.

---

## 3. Unification

Total rate is additive across loci:

```
R_total = R_action + R_perception
L_unified = L_flow_or_mse + beta_a * R_action + beta_v * R_perception
```

Two betas, not one: the loci differ in scale and role, and a single knob would
conflate them. The unified *claim* is conceptual (one per-band RD instrument), not
that the two legs share an identical training path — be explicit about this in the
README so a reviewer isn't surprised:

- action leg trains on the **sampled chunk** (post-sampler, `no_grad`) with MSE,
  as in `project.md`.
- perception leg trains on the **flow loss** (single forward) as in §2.5.

They are the same instrument at two loci, with locus-appropriate objectives. That
is the honest framing; do not claim operational identity.

---

## 4. Repo additions

```
sib/
  transforms.py        # + separable 2D spatial DCT-II/IDCT
  bottleneck.py        # + 2D-band indexing (reuses the same channel math)
  vision_wrapper.py    # NEW: inserts the perception module at the token interface
  lambda_estimator.py  # + EMA over (spatial-band, channel) for visual coeffs
  baselines_vision.py  # NEW: lowpass-spatial, gain-no-rate, raw-channel-VIB, IB-Adapter
configs/
  sib_vision.yaml      # perception leg, one beta_v
  sib_unified.yaml     # both legs on
  sweep_beta_v.yaml    # perception frontier
  vis_lowpass.yaml     # fixed spatial low-pass truncation (grid-searched cutoff)
  vis_gain_no_rate.yaml# learned per-band spatial gain, beta_v = 0
  vis_raw_vib.yaml     # per-channel VIB at the locus, NO spatial DCT
  vis_ibadapter.yaml   # StableVLA Fused IB-Adapter, ported (baseline)
scripts/
  vision_spectral_check.py  # NEW: Stage P1 diagnostic
```

---

## 5. Perception-leg baselines (each isolates one confound)

1. **Vanilla** (frozen projector, no module) — floor.
2. **vis_lowpass** — fixed spatial low-pass truncation, cutoff grid-searched.
   Tests "is it just dropping high spatial frequencies?"
3. **vis_gain_no_rate** — learned per-band spatial gain, `beta_v = 0`, no rate.
   Tests "does the rate constraint matter, or just learnable spatial filtering?"
4. **vis_raw_vib** — per-channel Gaussian channel at the same locus, NO spatial
   DCT. Tests "does the spectral basis matter?" (the make-or-break, mirrors the
   action leg's `raw_vib`).
5. **vis_ibadapter** — StableVLA Fused IB-Adapter, ported from the reference repo,
   trained in our frozen setup via the same flow loss. The published competitor at
   the identical insertion point. Spectral-rate-IB vs channel-covariance-IB.

Gate (contribution exists): the spectral-IB must beat `vis_raw_vib` on robustness,
AND beat-or-match `vis_ibadapter` on robustness-at-matched-clean OR clearly win on
interpretability + frontier (the per-spatial-band bit map + the RD curve, neither
of which IB-Adapter produces). If `vis_raw_vib` matches the spectral version
everywhere, write the honest negative ("the locus matters; the spatial basis does
not") and stop expanding scope — same discipline as the action leg.

---

## 6. Staged execution with gates

### Stage P0 — Interface inspection
Locate the visual-token tensor and spatial grid; verify shapes; confirm a clean
residual stream exists. **Gate:** spatial grid `(Hp, Wp)` is recoverable from the
token layout. If pixel-shuffle destroys a usable grid, fall back to a 1D DCT over
the token sequence (note the weaker spatial interpretation) or to channel-VIB.

### Stage P1 — Vision spectral pre-check (make-or-break, cheap)
`vision_spectral_check.py`: DCT the clean predicted visual tokens; report
per-spatial-band energy and off-diagonal correlation. Then, **for analysis only**
(not training), pass held-out corrupted observations and measure which spatial
bands shift under each corruption family.
- **Gate A (story):** if noise-family corruption concentrates in identifiable
  (typically low-clean-energy, high-spatial-frequency) bands, the budget can
  suppress it clean-trained — proceed, expect the win to be strongest on noise.
- **Gate B (assumption):** if corruption is spread uniformly across bands with no
  structure the budget can exploit, the spatial-spectral story is weak; fall back
  to channel-VIB at the locus and reframe the perception leg around that.
- Honest expectation to record now: blur removes information, so a spatial budget
  cannot fully restore it — predict partial gains on blur, stronger on noise.

### Stage P2 — Train perception IB at one beta_v (the perception MVE)
`sib_vision.yaml`, frozen everything, gain-init-1, flow-loss training, one mid
`beta_v`. **Gate:** clean success within ~2 pp of vanilla AND `R_perception`
decreased from init (bands suppressed) AND beats vanilla on at least the noise
corruption family. If clean drops > 2 pp, lower `beta_v` or enable the `g_gate`
blend; if nothing suppressed, raise `beta_v`.

### Stage P3 — Perception frontier + isolating baselines
`sweep_beta_v.yaml` → perception RD frontier. Train/eval `vis_lowpass`,
`vis_gain_no_rate`, `vis_raw_vib`, and the ported `vis_ibadapter`. Apply the
contribution gate in §5.

### Stage P4 — Unify + figures
`sib_unified.yaml`: both legs on. Eval clean + corruption + smoothness. Produce
the headline figures (§7). Confirm the two legs compose without clean regression.

---

## 7. Metrics & figures

Carry over the action-leg metrics. Add:
- **Per-spatial-band bit map** `R^v_{u,v}` — the perception-locus headline figure;
  shows the policy spending its budget on corruption-robust spatial bands.
- **Robustness by corruption family** (noise vs blur vs digital), so the
  noise-strong / blur-partial asymmetry is visible and honest, with Wilson 95% CIs
  and two-proportion z-tests vs vanilla.
- **Two-spectrum allocation figure** — temporal action bits beside spatial
  perception bits, the visual of the unified claim.
- **Frontier overlay** — action RD frontier and perception RD frontier on shared
  axes (bits vs success), the one-instrument-two-spectra figure.

---

## 8. Risks & fallbacks (all publishable)

- **Blur asymmetry:** spatial budget can't restore removed information. Fallback:
  report robustness gains are concentrated in additive-noise families; frame blur
  as the known limit of an information-preserving filter.
- **Feature spectrum ≠ pixel spectrum:** the encoder has already mixed spatial
  content. P1 gates this; if feature-space bands are uninformative, fall back to
  channel-VIB and reframe.
- **Clean-train allocation:** a clean budget allocates by clean energy; this helps
  noise (low-energy bands) more than blur. Stated up front as an analysis result.
- **Spectral basis adds nothing over channel-VIB at the locus:** honest negative —
  "the locus is the active ingredient, not the spatial basis." Still a clean result
  and a useful correction to the StableVLA story.

---

## 9. Budget map (remaining-budget aware)

The action leg is largely spent. Prioritise the perception leg; keep unified runs
minimal. Indicative new spend:

| Run | Approx GPU-h |
|---|---|
| P0 interface inspection | <1 |
| P1 vision spectral pre-check | 1 |
| P2 perception MVE (1 train + eval) | 7 |
| P3 beta_v frontier (4 points) | 20 |
| P3 baselines: lowpass, gain_no_rate, raw_vib, ib_adapter | 22 |
| P4 unified + corruption + figures | 12 |
| Buffer / reruns | 7 |

If remaining budget is tight, the minimum viable unified paper is P0–P2 + the two
make-or-break baselines (`vis_raw_vib`, `vis_ibadapter`) + the existing action
leg, with the unified framing carried mostly by writing and one joint P4 run.
Log actual hours to `results/gpu_hours.csv`.

---

## 10. Open decisions (defaults chosen; flag if you disagree)

1. **Two betas vs one** — default two (`beta_a`, `beta_v`). One knob conflates loci.
2. **Main-path gain vs additive residual** — default main-path Wiener gain,
   identity-at-init. Residual can't suppress corruption in the main stream.
3. **Spatial-grid fallback** — if pixel-shuffle breaks the grid, default to 1D
   token-sequence DCT with the weaker spatial reading; channel-VIB only if P1 fails.
4. **IB-Adapter port** — include as a baseline (recommended) to convert the
   competitor into a comparison you win at the same locus.

---

## 11. First action for the agent

Do **not** write the module yet. First run Stage P0: inspect the real LeRobot
SmolVLA forward, find the visual-token interface, verify `N, D, (Hp, Wp)`, and
write the verified shapes into `sib/vision_wrapper.py` as a docstring. Everything
in §2 is contingent on what P0 finds.

---

## 12. Extension — per-band operators (the low-frequency fix)

The spectral IB only *suppresses*, so it cannot fix low-frequency corruption
(brightness, contrast, fog): those live in the high-energy low bands the budget
must keep. Fix: treat the band split as routing to band-appropriate *operators*,
not one global bottleneck. This generalizes the project from "bits per band" to
"operator per band" — richer, still spectral.

Operator per band:
- High band -> IB suppression (additive noise / HF artifacts). As in §2.3.
- Low band -> corrective normalization toward the clean prior. New, §12.2.
- Mid band -> pass-through.

### 12.1 Why normalization, not generation

Brightness is a DC offset, contrast a global gain; both are affine and
information-preserving, so an affine correction inverts them — no pixels need
regenerating. Generative restoration is the right family only when information is
*destroyed* (blur, occlusion, heavy compression), and it is not viable here: it
needs corruption exposure to learn the inverse (breaks clean-train), adds
diffusion-scale latency (breaks the control loop), needs its own training budget,
and hallucinates realistic-but-wrong features — the smoothly-wrong failure that
motivated this work, now harder to detect. Concede blur; no generator in the loop.

### 12.2 Reference-free correction (the workable "router")

At eval there is no clean reference, so we cannot detect "this band is corrupted."
Instead match the low-band statistics to the stored clean prior. Track `mu`
(EMA per-band clean mean) alongside the existing `lam` (EMA per-band clean
variance). The low-band operator standardizes toward that prior:

```python
# low-band coeffs X_lo: (B, bands_lo, D)
inst_mu = X_lo.mean(dim=(0, 1), keepdim=True)   # per-channel instance stats
inst_sd = X_lo.std(dim=(0, 1), keepdim=True)
X_norm  = (X_lo - inst_mu) / (inst_sd + eps)
X_lo_hat = gamma * (X_norm * lam_lo.sqrt() + mu_lo) + (1 - gamma) * X_lo
```

`gamma` learnable in `[0, 1]`, init 0 (identity, clean preserved). On clean data
the instance stats already match the prior, so the correction is ~null; under a
brightness/contrast shift the stats deviate and the map pulls them back.
Reference-free, uses only the clean prior — the feasible form of "realize it's off
and fix it." `mu`/`lam` are detached; only `gamma` (and the band cutoffs) train.

### 12.3 New baselines, gates, forecast

- Baseline `vis_global_norm`: plain whole-feature instance-norm, no band split.
  **Gate (routing matters):** the low-band corrective must beat this. If
  whole-feature norm ties it, the split adds nothing — report and drop it.
- **Gate (clean tie):** normalizing the low band can erase task-relevant global
  contrast. Must tie vanilla clean; if it drops, lower the `gamma` cap or narrow
  the low-band cutoff.
- Honest forecast: genuinely moves brightness/contrast (affine, invertible); fog
  partial (low-freq but not purely affine); blur still out of reach. This
  supersedes the "brightness structurally out of reach" line in §8 for the affine
  families only.

### 12.4 Staging

Fold into Stage P2/P3: add the low-band operator once the high-band IB MVE passes
P2, and add `vis_global_norm` to the P3 baseline set. Indicative extra spend:
~6 GPU-h (small module, mostly eval). If over budget, the low-band operator can
ship as a clean ablation on the brightness/contrast families alone.
