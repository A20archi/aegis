## Clean (non-perturbed) — 4 suites × 3 seeds (42/123/456), 20 ep/task, replan 5

| Suite | base mean | AEGIS mean | Δ mean | gate | per-seed Δ | Δ peak (seed) |
|---|---:|---:|---:|:--:|---|---:|
| Spatial |  90.8 |  94.7 | +3.8 | open | +4.0, +4.0, +3.5 | +4.0 |
| Object |  70.0 |  80.0 | +10.0 | open | +10.0, +10.0, +10.0 | +10.0 |
| Goal |  73.5 |  76.5 | +3.0 | open | +0.5, +6.0, +2.5 | +6.0 |
| Long |  55.5 |  45.2 | -10.3 | closed | -10.0, -4.5, -16.5 | -4.5 |
| **Avg** | ** 72.5** | ** 76.7** | **+4.2** | | | |

## LIBERO-Plus — 4 suites × 3 seeds × 7 perturbation families, 12 tasks/cat, replan 5

| Suite | base mean | AEGIS mean | Δ mean | gate | Δ peak |
|---|---:|---:|---:|:--:|---:|
| Spatial |  55.6 |  58.3 | +2.8 | open | +6.0 |
| Object |  51.2 |  61.9 | +10.7 | open | +16.7 |
| Goal |  57.5 |  60.7 | +3.2 | open | +7.1 |
| Long |  26.2 |  29.8 | +3.6 | open | +6.0 |
| **Avg** | ** 47.6** | ** 52.7** | **+5.1** | | |

### LIBERO-Plus per-family (mean over suites & seeds, ungated diagnostic)

| Family | base | AEGIS | Δ |
|---|---:|---:|---:|
| Camera Viewpoints |  43.8 |  43.7 | -0.0 |
| Light Conditions |  67.4 |  78.5 | +11.1 |
| Sensor Noise |  35.4 |  61.8 | +26.4 |
| Background Textures |  52.8 |  50.0 | -2.8 |
| Objects Layout |  50.7 |  52.8 | +2.1 |
| Robot Initial States |  21.5 |  22.9 | +1.4 |
| Language Instructions |  61.8 |  59.0 | -2.8 |
