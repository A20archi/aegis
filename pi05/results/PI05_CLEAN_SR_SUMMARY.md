# pi0.5 clean SR — no-clean-tax check (standard LIBERO, 3 seeds {42,123,456})

Base = frozen pi0.5. +AEGIS = floored gate (tau=0.8). n=50/cell (10 tasks × 5 trials). No oracle.

| Suite | pi0.5 base | pi0.5 + AEGIS | Δ |
|-------|-----------|---------------|---|
| Spatial | 99.3 | 98.0 | −1.3 |
| Object  | 97.3 | 96.7 | −0.7 |
| Goal    | 97.3 | 98.7 | +1.3 |
| Long    | 90.0 | 90.7 | +0.7 |
| **Net** | **96.0** | **96.0** | **+0.00** |

**Net Δ = +0.00 → exact tie, NO clean tax.** Deviations are within the n=50 sampling band
(±2% = one episode). The floored gate closes on in-distribution clean inputs (identity-at-init
verified, clean gate 0.61 < tau 0.80), so pi0.5+AEGIS ≡ pi0.5 on clean.

Companion result — LIBERO-Plus (perturbed): AEGIS-floored **net +1.6** (fixes raw-gate over-engagement
on Camera Viewpoints −5.6→+1.4 and Language −2.8→+4.2). Per-cell JSONs in clean_json/ and lplus_json/.
