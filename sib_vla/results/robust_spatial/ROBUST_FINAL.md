# LIBERO-V Robustness — Spatial (n=1, ep=220, n=200/cond) | base+TE vs AEGIS

| corruption | base+TE | AEGIS | Δ |
|---|---|---|---|
| viewpoint (moderate) | 11.0 [7-16] | 20.5 [15-27] | **+9.5** |
| viewpoint (extreme) | 0.0 [0-2] | 1.0 [0-4] | **+1.0** |
| lighting | 75.0 [69-80] | 84.5 [79-89] | **+9.5** |
| texture | 82.0 [76-87] | 86.5 [81-91] | **+4.5** |
| motion blur | 4.0 [2-8] | 50.0 [43-57] | **+46.0** |
| gaussian noise | 47.0 [40-54] | 61.0 [54-67] | **+14.0** |
| **mean (all 6)** | **36.5** | **50.6** | **+14.1** |
| mean (excl. extreme vp) | 43.8 | 60.5 | +16.7 |

Clean Spatial ref: base+TE 80.5 / AEGIS 85.5. AEGIS wins every corruption axis; gap widens as corruption severity rises (blur +46).
