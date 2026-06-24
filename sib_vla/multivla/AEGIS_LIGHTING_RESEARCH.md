# Lighting-robustness research → AEGIS RIB engineering plan (2026-06-24)

Goal: kill the RIB regression on LIBERO-Plus **Light Conditions** (−5…−11 across suites) without
losing the Sensor/Camera wins. Four parallel research agents (arxiv + web). The findings converge
on one dominant lever that is *not* "more augmentation."

## The headline finding (changes the strategy)
**LIBERO-Plus (arXiv 2510.13626) attributes lighting robustness to the WRIST camera, not the
encoder or augmentation.** Lighting corrupts the *third-person (agentview)* image and global
appearance; the *wrist / eye-in-hand* view stays illumination-stable (close-range geometric cues).
- Their `3rd-black` ablation (mask 3rd-person, keep wrist): models still score 43.6 / 43.0 / 67.3.
- Third-person-ONLY models collapse on lighting (OpenVLA −60+pp, Nora −56.9, WorldVLA −49.7).
- Lighting leaders are all wrist-camera OpenVLA-OFT variants (92.7 / 94.9 Light SR).
- **Brute-force photometric augmentation has a LOW CEILING on lighting:** their aug helped
  viewpoint +37pp but lighting only **+2.2pp**. So "train RIB on more lighting aug" alone won't fix it.

**SmolVLA on LIBERO feeds BOTH views** (confirmed, arXiv 2506.01844 + lerobot docs):
`observation.images.image`=agentview, `observation.images.image2`=wrist, **64 tokens/view**,
concatenated at the MLP projector (= our RIB insertion point). So at the connector we have a clean
**two-block structure: [agentview 64 tokens | wrist 64 tokens]**.

→ **The highest-leverage RIB change is a per-VIEW gate that down-weights the agentview block under
lighting corruption and leans on the (stable) wrist block.** Directly justified by the source paper;
identity-at-init (gate=1); and *no published method does connector-level per-view gating to fix VLA
lighting* — that gap is AEGIS's novel wedge.

## Prioritized engineering plan for RIB

### Tier 1 — Per-view gate (DO FIRST; biggest, cheapest, novel)
- Restructure the RIB gate (currently per-sample scalar, AEGIS v2) to be **per-view-block**: one gate
  for the agentview 64-token block, one for the wrist 64-token block. Identity-init = 1.0.
- Architecture template: **BFA "Best-Feature-Aware" score net** (arXiv 2502.11161) — GAP each view's
  tokens → tiny linear head → per-view weight; reweight-and-sum (soft, not hard select). Adapt it to
  identity-init and train on corruption (not BFA's task-stage labels).
- **Train with view-dropout** (sensor/modality dropout; arXiv 2002.09107, 2410.03010): randomly mask
  or heavily lighting-corrupt the agentview block so the gate learns to ride the wrist view. Pure
  training-time, frozen-backbone-safe. The existing `cmask`/gate-BCE in finetune_rib.py generalizes:
  supervise the agentview gate toward "closed" when its block is the corrupted one.
- **VERIFY FIRST (on Blackwell):** confirm the token layout at `connector.modality_projection.proj`
  is per-view-separable (is RIB called once on [agentview|wrist] concatenated, or per-image?). This
  determines whether the per-view split happens inside RIB or one level up. Check
  `vlm_with_expert.vlm.model.connector` call sites.

### Tier 2 — Sigmoid Gram-channel gate (proven lighting gain)
- **StableVLA IB-Adapter (arXiv 2605.18287)** is the only connector-level module with a real lighting
  number (Contrast **+42pp**, sev-5). Mechanism: sigmoid (NOT softmax) gate over a **channel×channel
  Gram matrix** `G = QᵀK`, `A = σ(G·τ)`, `Z = V·A`. Global illumination injects a component
  *uncorrelated* with semantic structure → anomalous-covariance channels → `σ(small)≈0` → suppressed,
  **without stealing mass from good channels** (softmax can't do this; it collapsed CALVIN 2.13→0.46).
- Our RIB already has the fused identity-init form `out = z_mlp + tanh(coeff)·RIB(z_mlp)` — adopt the
  **sigmoid-Gram channel gating inside RIB's bottleneck**. Low-risk, <10M params, cache-compatible.
- Note: brightness/saturate are near-ceiling; **contrast/intensity is where the headroom is** — target it.

### Tier 3 — Selective feature whitening / IN-residual on the agentview stream
- **ISW / RobustNet (arXiv 2103.15597):** whiten only the feature-covariance directions that MOVE
  under photometric augmentation (clean vs aug covariance diff) → removes illumination "style", keeps
  content. Runs on cached FEATURES (not pixels). Add as an ISW loss + light whitening.
- Cheaper cousin: residual Instance-Norm `Z = X + g·(IN(X) − X)`, g init 0 (IBN-Net 1807.09441). IN
  strips the per-channel style statistic that encodes illumination. Apply **selectively + residual**
  (IN hurts if applied to deep content features blanket-style).

### Tier 4 — Lighting augmentation done right (pairs with Tier 1, not standalone)
- Recipe (robotics-validated, mild): brightness ±20%, **contrast ±30%** (the headroom axis), gamma
  [0.7,1.5], color-temperature ±15% (per-channel R/B gain = the "color" lighting factor), synthetic
  shadow (1–4 quad regions, darkness ~0.65; arXiv 2010.04767).
- **Pixel-space → must pass the frozen encoder.** Either re-cache features from augmented frames once,
  then train RIB as **noisy-feature → clean-feature consistency**, or apply on-the-fly. Don't expect
  this alone to fix lighting (the +2.2pp ceiling); it's the *signal that trains the Tier-1 gate*.

### Tier 5 — (optional) gradient-free test-time feature renorm
- Cache clean per-view per-channel mean/cov of connector features; at test, re-center/whiten the live
  agentview block toward clean stats (TTN 2205.10210 / DUA, gradient-free). Identity-at-init via a
  blend coefficient. Well-grounded in vision TTA; unproven in VLA (= novel, but optional).

## Strategic notes
- **The lever is architecture (per-view gate), not augmentation.** LIBERO-Plus proves aug's lighting
  ceiling is ~+2pp; the wrist-camera asymmetry is worth ~45–86pp between models.
- **Novelty wedge:** no existing method is BOTH post-hoc/frozen AND demonstrated on the lighting axis
  via connector-level per-view gating. Tier-1 + Tier-2 is a clean, defensible contribution.
- TENT-style entropy TTA does NOT transfer (SmolVLA has no BatchNorm, flow-matching head, no softmax).

## Key sources
- LIBERO-Plus (wrist-cam finding, leaderboard) — arXiv 2510.13626
- StableVLA IB-Adapter (sigmoid Gram gate, +42pp contrast) — arXiv 2605.18287
- BFA per-view score net — arXiv 2502.11161 · view/modality dropout — 2002.09107, 2410.03010
- ISW/RobustNet selective whitening — 2103.15597 · IBN-Net — 1807.09441
- SmolVLA (2 views, 64 tok/view) — arXiv 2506.01844 · TTN — 2205.10210 · synthetic shadow — 2010.04767
