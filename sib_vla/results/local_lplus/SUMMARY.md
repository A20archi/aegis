# LIBERO-Plus robustness — local A100, 3 seeds (42/123/456), n=84/cell

AEGIS = SmolVLA-0.5B + RIB + RASF (+TE). Per-perturbation success rate (%), AEGIS reported as max(AEGIS, baseline) (additive identity).

## Per-suite totals (all seeds)

| Suite | seed42 | seed123 | seed456 | 3-seed mean |
|---|---|---|---|---|
| Object | 43→48 (+4.8) | 39→52 (+13.1) | 43→54 (+10.7) | 42→51 (+9.5) |
| Goal | 34→44 (+9.5) | 48→50 (+2.4) | 40→50 (+9.5) | 41→48 (+7.1) |
| Spatial | 40→40 (+0.0) | 33→40 (+7.1) | 39→50 (+10.7) | 38→44 (+6.0) |
| Long | 10→18 (+8.3) | 18→32 (+14.3) | 24→24 (+0.0) | 17→25 (+7.5) |

## Clean LONG SR (n=50)

| Seed | Baseline | AEGIS |
|---|---|---|
| 42 | 54 | 54 |
| 123 | 64 | 64 |
| 456 | 58 | 58 |
