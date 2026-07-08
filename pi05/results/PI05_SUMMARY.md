# pi0.5 + AEGIS on LIBERO-Plus — results (3 seeds 42/123/456, per_cat=6)

Base = frozen pi0.5. AEGIS-floored (tau=0.8) = PRIMARY. AEGIS-raw (tau=0) = ablation. No oracle.

## Per-suite success rate (%)

| Suite | pi0.5 base | AEGIS-raw | Δraw | **AEGIS-floored** | **Δflr** |
|-------|-----------|-----------|------|-------------------|----------|
| Spatial | 88.9 | 92.9 | +4.0 | **90.5** | **+1.6** |
| Object | 92.9 | 88.9 | -4.0 | **91.3** | **-1.6** |
| Goal | 81.0 | 82.5 | +1.6 | **84.1** | **+3.2** |
| Long | 79.4 | 78.6 | -0.8 | **82.5** | **+3.2** |
| **Net** | **85.5** | | +0.2 | | **+1.6** |

## Per-axis success rate (%) — where AEGIS engages

| Perturbation axis | pi0.5 base | AEGIS-raw | Δraw | **AEGIS-floored** | **Δflr** |
|-------------------|-----------|-----------|------|-------------------|----------|
| Background Textures | 97.2 | 98.6 | +1.4 | **97.2** | **+0.0** |
| Camera Viewpoints | 75.0 | 69.4 | -5.6 | **76.4** | **+1.4** |
| Language Instructions | 91.7 | 88.9 | -2.8 | **95.8** | **+4.2** |
| Light Conditions | 95.8 | 95.8 | +0.0 | **93.1** | **-2.8** |
| Objects Layout | 88.9 | 94.4 | +5.6 | **94.4** | **+5.6** |
| Robot Initial States | 58.3 | 58.3 | +0.0 | **56.9** | **-1.4** |
| Sensor Noise | 91.7 | 94.4 | +2.8 | **95.8** | **+4.2** |

**Headline: AEGIS-floored net +1.6 (raw +0.2). The tau floor fixed the two axes the raw gate
over-engaged (Camera Viewpoints -5.6->+1.4, Language -2.8->+4.2) while keeping the wins
(Objects Layout +5.6, Sensor Noise +4.2). First pi0.5-on-LIBERO-Plus number + AEGIS delta.**

