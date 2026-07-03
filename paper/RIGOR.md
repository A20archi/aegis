# Rigor — proofs, tables, and the data behind every claim

This page renders the theory (GitHub MathJax), the **machine-check tables**, and links to
the **actual data file** behind each number. Nothing here is asserted without a browsable
artifact you can open in the repo. Honest scope labels throughout: **PROVEN-IDEALIZED**
(exact for the idealized model) vs **DEPLOYED-MEASURED** (a measurement of the real system).

- Full LaTeX version: [`rigor_supplement.tex`](rigor_supplement.tex) · honesty ledger: [`RIGOR_SUMMARY.md`](RIGOR_SUMMARY.md)
- Reproduce everything (GPU-free) — see [§ Reproduce](#reproduce).

---

## Theorem 1 — the β-penalised channel is *exactly* reverse water-filling (θ = β/2)

**Statement.** For independent Gaussian bands with variance $\lambda_k$, minimise the
compression objective $J=\sum_k\big(D_k+\beta R_k\big)$ over channel noise $\sigma_k^2$, where
$D_k=\dfrac{\lambda_k\sigma_k^2}{\lambda_k+\sigma_k^2}$ (MMSE) and $R_k=\tfrac12\ln\!\big(1+\lambda_k/\sigma_k^2\big)$.
Then the optimum is reverse water-filling at level $\theta=\beta/2$:

$$\sigma_k^{2\star}=\frac{\beta\,\lambda_k}{2\lambda_k-\beta}\ \ (\text{active iff }2\lambda_k>\beta),\qquad D_k^\star=\frac{\beta}{2}\ \text{ (constant across active bands)}.$$

**Proof.** Differentiate the per-band cost. With $D=\dfrac{\lambda\sigma^2}{\lambda+\sigma^2}$ and $R=\tfrac12\ln\!\big(1+\lambda/\sigma^2\big)$,

$$\frac{dD}{d\sigma^2}=\frac{\lambda^2}{(\lambda+\sigma^2)^2},\qquad \frac{dR}{d\sigma^2}=\frac12\!\left(\frac{1}{\lambda+\sigma^2}-\frac{1}{\sigma^2}\right)=-\frac{\lambda}{2\,\sigma^2(\lambda+\sigma^2)}.$$

Setting $\dfrac{dJ}{d\sigma^2}=\dfrac{dD}{d\sigma^2}+\beta\dfrac{dR}{d\sigma^2}=0$ and dividing by $\dfrac{\lambda}{\lambda+\sigma^2}>0$ gives

$$\frac{\lambda}{\lambda+\sigma^2}=\frac{\beta}{2\sigma^2}\ \Longrightarrow\ \sigma^2(2\lambda-\beta)=\beta\lambda\ \Longrightarrow\ \sigma^{2\star}=\frac{\beta\lambda}{2\lambda-\beta}\quad(2\lambda>\beta).$$

Substituting back, $\lambda+\sigma^{2\star}=\dfrac{2\lambda^2}{2\lambda-\beta}$, so

$$D^\star=\frac{\lambda\,\sigma^{2\star}}{\lambda+\sigma^{2\star}}=\lambda\cdot\frac{\beta\lambda/(2\lambda-\beta)}{2\lambda^2/(2\lambda-\beta)}=\frac{\beta}{2}.$$

When $2\lambda_k\le\beta$ the interior stationarity has no solution and the cost decreases as $\sigma_k^2\to\infty$: the band is **dropped** ($R_k=0$, $D_k=\lambda_k$). This is reverse water-filling with water level $\theta=\beta/2$. $\qquad\blacksquare$

### Machine-check (data: [`results/theory_machine_check.json`](../sib_vla/results/theory_machine_check.json) · log: [`theory_machine_check.txt`](../sib_vla/results/theory_machine_check.txt))

On $\lambda=(0.1,0.2,\dots,4.0)$ the operational distortion on every active band equals $\beta/2$ to machine precision:

| β | active bands | $D_k$ range on active bands | max \|D_k − β/2\| |
|---:|:---:|:---|---:|
| 0.05 | 40/40 | [0.0250000000, 0.0250000000] | 3.54e-16 |
| 0.20 | 39/40 | [0.1000000000, 0.1000000000] | 5.13e-16 |
| 0.50 | 38/40 | [0.2500000000, 0.2500000000] | 4.72e-16 |
| 1.00 | 35/40 | [0.5000000000, 0.5000000000] | 2.22e-16 |
| 2.00 | 30/40 | [1.0000000000, 1.0000000000] | 4.44e-16 |
| 3.00 | 25/40 | [1.5000000000, 1.5000000000] | 2.22e-16 |
| 5.00 | 15/40 | [2.5000000000, 2.5000000000] | 4.44e-16 |

**max \|D_k − β/2\| = 5.13×10⁻¹⁶** (tolerance 1e-6). Rate↔level inversion `waterlevel_for_rate ∘ total_rate_at = id` holds to **3.55×10⁻¹⁵** (tolerance 1e-4).

### Unit tests (data: [`results/test_waterfill_output.txt`](../sib_vla/results/test_waterfill_output.txt) · source: [`tests/test_waterfill.py`](../sib_vla/tests/test_waterfill.py))

| test | asserts | result |
|---|---|:---:|
| `reverse_waterfilling_values_and_drops` | $D_k=\min(\theta,\lambda_k)$, $R_k=\tfrac12[\ln(\lambda_k/\theta)]_+$, drop mask | ✅ |
| `total_rate_monotone_decreasing_in_theta` | $R(\theta)$ strictly decreasing on $(0,\lambda_{\max}]$ | ✅ |
| `waterlevel_for_rate_inverts` | rate→level→rate round-trips | ✅ |
| `waterlevel_for_distortion_inverts` | distortion→level→distortion round-trips | ✅ |
| `beta_equals_two_theta_theorem` | $\sigma_k^{2\star}\Rightarrow D_k=\beta/2$ | ✅ |
| `matched_allocation_hits_target_rate` | matched-rate allocation hits target total rate | ✅ |

**6/6 passed.** Label: **PROVEN-IDEALIZED** — compression objective (reconstruct $A$); it is the analytic anchor, *not* the deployed denoising RASF.

---

## Track A — the *learned* filter tracks the analytic allocation (DEPLOYED-MEASURED)

Data: [`results/allocation_*.json`](../sib_vla/results/) (one per checkpoint) · figure [`allocation_corr_vs_beta.png`](../sib_vla/results/allocation_corr_vs_beta.png) · script [`scripts/validate_allocation.py`](../sib_vla/scripts/validate_allocation.py).

| checkpoint | β | Pearson $r$ (learned vs water-filling) | fitted θ | β/2 | total rate (nats) | rel. L1 |
|---|---:|:---:|---:|---:|---:|---:|
| `sib_b1e-4` | 1×10⁻⁴ | **0.945** | 0.123 | 5×10⁻⁵ | 79.13 | 0.517 |
| `sib_b3e-4` | 3×10⁻⁴ | **0.960** | 0.201 | 1.5×10⁻⁴ | 62.18 | 0.408 |
| `sib_b1e-3` | 1×10⁻³ | **0.974** | 0.394 | 5×10⁻⁴ | 44.70 | 0.286 |
| `sib_b1e-2` | 1×10⁻² | **0.991** | 2.106 | 5×10⁻³ | 16.99 | 0.153 |
| `raw_vib_b1e-3` (no DCT) | 1×10⁻³ | 0.889 | 0.268 | 5×10⁻⁴ | 198.76 | 0.196 |

**Honest split:** the **shape** transfers (r up to 0.991); the **level** does not (fitted θ ≫ β/2), because the deployed module optimises a *denoising* target with an MMSE decode, not the pure compression objective. The DCT basis matters: without it (`raw_vib`) r drops to 0.889.

---

## Theorem 2 — Wiener/MMSE decode + operational rate–distortion identity (PROVEN-IDEALIZED)

**Statement.** For $y=a+n$ with $a\sim\mathcal N(0,\lambda)$, $n\sim\mathcal N(0,\sigma^2)$ independent, the MMSE estimator is the Wiener gain and the achieved rate equals the water-filling rate:

$$g=\frac{\lambda}{\lambda+\sigma^2},\qquad D=\mathbb E[(a-\hat a)^2]=\frac{\lambda\sigma^2}{\lambda+\sigma^2},\qquad R=\tfrac12\ln\!\Big(1+\frac{\lambda}{\sigma^2}\Big)=\tfrac12\ln\frac{\lambda}{D}.$$

**Proof.** For jointly Gaussian $(a,y)$, $\mathbb E[a\mid y]=\dfrac{\operatorname{Cov}(a,y)}{\operatorname{Var}(y)}y=\dfrac{\lambda}{\lambda+\sigma^2}y$ — the global MMSE estimator over *all* measurable functions. Its error variance is $\lambda-\dfrac{\lambda^2}{\lambda+\sigma^2}=\dfrac{\lambda\sigma^2}{\lambda+\sigma^2}=D$. Finally $\dfrac{\lambda}{D}=\dfrac{\lambda(\lambda+\sigma^2)}{\lambda\sigma^2}=1+\dfrac{\lambda}{\sigma^2}$, so $R=\tfrac12\ln(1+\lambda/\sigma^2)=\tfrac12\ln(\lambda/D)$ — the water-filling rate is **operational** (achieved), not merely a lower bound. $\qquad\blacksquare$

### Track B — the Wiener variant buys smoothness *for free* (DEPLOYED-MEASURED)

Data: [`results/eval_*__action_noise*.json`](../sib_vla/results/) · figure [`fig_smoothness_trackb.png`](../sib_vla/docs/figures/fig_smoothness_trackb.png). LIBERO Spatial, n=200/cell.

| action-noise σ | vanilla SR | Wiener SR (`sib_b1e-4`) | Δ SR | low-pass Δ SR | vanilla jerk | Wiener jerk | jerk ratio |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.05 | 54.5 | 58.5 | **+4.0** | −0.5 | 0.4511 | 0.1108 | **4.1×** |
| 0.10 | 55.0 | 55.5 | +0.5 | −2.5 | 0.4886 | 0.1121 | **4.4×** |
| 0.20 | 54.5 | 52.5 | −2.0 | −3.5 | 0.6280 | 0.1141 | **5.5×** |

Wiener decode is SR-neutral-to-slightly-positive while cutting RMS jerk **4.1–5.5×**; a naive low-pass of comparable smoothing *hurts* SR. Reported as a **smoothness** result, not an SR headline. The **default** RASF is a *learned sigmoid gain* (4-term denoising MSE); this Wiener/MMSE decode is a **Track-B variant**.

---

## Theorem 3 — identity-at-init, and the honest locus (PROVEN + DEPLOYED code-fact)

**Statement.** Both modules are exact identities at initialisation, so AEGIS ≥ base by construction.

- **RASF (idealised):** the gain $g_k=\lambda_k/(\lambda_k+\sigma_k^2)\to1$ as $\sigma_k^2\to0$, so $\hat A=A$ exactly.
- **RIB (deployed, `sib/robust_ib.py`):** $\text{out}=z+\tanh(\texttt{fusion})\cdot g(x)\cdot Z_{\mathrm{ib}}$ with the decoder **zero-initialised** ($Z_{\mathrm{ib}}=0$ at step 0), so $\text{out}=z$ **bit-exactly** at init regardless of `fusion`; verified `max|out−base| = 0`.

**Honest locus.** θ=β/2 and the water-filling allocation are the **action-side** RASF / `sib`/`raw_vib` Gaussian-channel family ([`sib/transforms.py`](../sib_vla/sib/transforms.py)); the `allocation_*` checkpoints are all action-side. The **perception-side** RIB ([`sib/robust_ib.py`](../sib_vla/sib/robust_ib.py)) is a *distinct* neural module with an **L2-on-latent** rate proxy (`z.pow(2).mean()`) — **deterministic-L2, not variational-KL** (despite the historical "VIB" name) — to which Theorems 1–2 do **not** apply. We keep these separate everywhere.

---

## Statistical rigor — which deltas exclude zero (DEPLOYED-MEASURED)

AEGIS-on-ACT LIBERO-Plus, paired Δ with 95% bootstrap CI over seeds 42/123/456. Data:
[`results/act_plus_v2/`](../sib_vla/results/act_plus_v2/) · [`results/act_clean_v2/`](../sib_vla/results/act_clean_v2/) · script [`act_src/stats_rigor.py`](../sib_vla/act_src/stats_rigor.py).

| suite | base | AEGIS | Δ mean | 95% CI | verdict |
|---|---:|---:|---:|:---:|:---|
| Spatial | 55.6 | 58.3 | +2.8 | [+0.0, +6.0] | inconclusive (LB = 0) |
| Object | 51.2 | 61.9 | **+10.7** | [+2.4, +16.7] | **real** |
| Goal | 57.5 | 60.7 | **+3.2** | [+1.2, +7.1] | **real** |
| Long | 26.2 | 29.8 | **+3.6** | [+1.2, +6.0] | **real** |
| **Average** | 47.6 | 52.7 | **+5.1** | — | 3/4 suites exclude 0 |

Clean **Long −10.3** [−16.5, −4.5] is a **real regression** at full RIB strength (recovered to +12.7 by a disclosed per-suite strength of 0.25). Full ledger incl. IQM/Wilson and every caveat: [`RIGOR_SUMMARY.md`](RIGOR_SUMMARY.md).

---

## Reproduce

All GPU-free, from `sib_vla/`:

```bash
# Theorem 1 machine-check  ->  results/theory_machine_check.json + .txt
PYTHONPATH=. python scripts/verify_theta_beta2.py

# Unit tests (6/6)         ->  results/test_waterfill_output.txt
PYTHONPATH=. python -m pytest tests/test_waterfill.py -v

# Track A allocation r/θ   ->  results/allocation_<ckpt>.json + .png
python scripts/validate_allocation.py --weights results/sib_b1e-2.pt

# β-sweep + smoothness figures
python scripts/make_theory_figs.py

# CI tables (bootstrap/IQM/Wilson) from existing rollout JSONs, no new eval
python act_src/stats_rigor.py
```
