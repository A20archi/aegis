# AEGIS results tracker (internal source-of-truth for the deck)

**External-labeling rule (for the PPT/deck):** our method = **"SmolVLA+SIB"** externally
(AEGIS = internal name). Do NOT surface ForgeVLA, the forge recipe, L1/L3/L4/L5 levers,
or temporal-ensembling-as-forge in external materials. Baseline = "SmolVLA"; both arms
share temporal ensembling, which is *not* surfaced as a named lever.

Internal: **AEGIS = RIB (robust IB @ visual connector) + RASF (adaptive spectral filter
@ action chunk) + TE**. Verified numbers only; `(t0)` = single-task preview (n=20, noisy);
blanks = pending the Wed–Fri runs.

---

## 1. Clean SR (LIBERO-Spatial, n=200)
| config | clean SR | Wilson95 | note |
|---|---|---|---|
| SmolVLA + TE (baseline, base 020000) | **86.0%** | [80.5, 90.1] | measured |
| AEGIS (RIB+RASF+TE, base 020000) | **87.5%** | [82.2, 91.4] | measured — beats baseline, plug-in net-additive |
| AEGIS (RIB+RASF+TE, **v2 base**) | _TBD_ | | **target 92** (Wed) |

## 2. Robustness — LIBERO-V (baseline+TE vs AEGIS), n=200 target
| axis / condition | type | baseline | AEGIS | gap | note |
|---|---|---|---|---|---|
| viewpoint_medium | systematic (geom) | 0% (t0) | 20% (t0) | **+20pp (t0)** | baseline collapses; RIB warp-training recovers |
| viewpoint_large  | systematic (geom) | _TBD_ | _TBD_ | | |
| lighting_1       | systematic (photo) | _TBD_ | _TBD_ | | RIB should help (brightness in train aug) |
| texture_1        | systematic (photo) | _TBD_ | _TBD_ | | RIB partial (no direct train analog) |
| motion_blur_1    | systematic (blur) | _TBD_ | _TBD_ | | RIB should help (in train aug) |
| gaussian_noise_1 | stochastic | 80% (t0) | 75% (t0) | ~tie (t0) | TE already handles stochastic noise |
| **APPEARANCE-AVG** | | _TBD_ | _TBD_ | | headline |

**Framing (data-driven):** RIB **complements** TE — TE neutralizes *stochastic* per-frame
noise (tie), RIB recovers *systematic* visual shifts TE can't average away (viewpoint +20pp).

## 3. Noise graceful-degradation curve (gaussian_noise, both arms)
| std | baseline+TE | AEGIS | gap |
|---|---|---|---|
| 0.05 | _TBD_ | _TBD_ | |
| 0.12 | 80% (t0) | 75% (t0) | ~tie |
| 0.20 | _TBD_ | _TBD_ | |
| 0.30 | _TBD_ | _TBD_ | |
Hypothesis: gap ~0 at low std, grows to ~+5pp as std rises (TE breaks down, RIB denoises).

## 4. Cross-architecture generalization (NEXT WEEK, wk of 2026-06-22)
| host | arch family | AEGIS gain | status |
|---|---|---|---|
| SmolVLA-500M | decoder-LLM + flow | (this week) | in progress |
| NanoVLA-S (161M) | ACT enc-dec + CVAE | _TBD_ | full reimpl (no public code) |
| TinyVLA ~400M / substitute | LLM + diffusion | _TBD_ | diffusion head + MetaWorld |

## Artifacts
- Base retrain: `outputs/smolvla_spatial_v2/checkpoints/` (batch 384, 30k, full LIBERO)
- v2 pipeline: `run_v2_pipeline.sh {evalbase|modules|aegis|all} <ckpt>`
- Robustness: `run_libero_v_headline.sh` (env: CFG/RIB/RASF) ; noise: `run_noise_sweep.sh`
- Videos: `results/videos/libv_{baseline,aegis}/<condition>/`
- NanoVLA plan: `multivla/NANOVLA_IMPL_PLAN.md`
