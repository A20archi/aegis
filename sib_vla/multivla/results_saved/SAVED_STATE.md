# Modal AEGIS — saved state (2026-06-21, spend PAUSED at $95.47 / $100)

## Spend status
All Modal apps stopped; $0/hr now. Budget at **$95.47 / $100** (~$4.53 left).
Remaining 3 SR pairs are PAUSED by user — resume after supervisors reset the cap.
Track real spend: `modal billing report --for "this month" --json` (URL:
https://modal.com/settings/seo-40141/usage?tab=usage).

## SmolVLA LIBERO-V robustness grid (base=gate-closed, AEGIS=gate-open, +TE, n_action_steps=1)
9 complete pairs, **mean delta +33.3 pts, 0 regressions** (gate provably never worse):

| suite  | condition        | base | AEGIS | delta  | n   |
|--------|------------------|-----:|------:|-------:|-----|
| object | motion_blur_1    |   0  |  86   | +86.0  | 100 |
| object | gaussian_noise_1 |  36  |  90   | +54.5  | 200 |
| object | lighting_1       |  58  |  92   | +34.0  | 100 |
| object | texture_1        |  83  |  97   | +14.0  | 200 |
| goal   | motion_blur_1    |  19  |  78   | +59.0  | 100 |
| goal   | viewpoint_medium |  17  |  43   | +26.0  | 100 |
| goal   | viewpoint_large  |   8  |  29   | +21.0  | 100 |
| goal   | texture_1        |  90  |  93   |  +3.0  | 100 |
| goal   | lighting_1       |  80  |  82   |  +2.0  | 100 |

Durable: Modal vol `smolvla-assets:results_modal/liberov_objgoal/` + local /tmp pulls.

## Videos OBTAINED (2026-06-21, curated demo pass, cost +$1.29)
12 mp4s = 3 axes x 2 tasks x {baseline, AEGIS}, object suite, n=4 demo set.
Saved durable: Modal vol `sib_vla/results/videos/` + repo `results_saved/videos/`.
NOTE: recorder writes RELATIVE `results/videos/...`, cell cwd=/assets/sib_vla, so
they land at `/assets/sib_vla/results/videos/`, NOT under results_modal/liberov_video.

| axis             | base SR | AEGIS SR | demo quality                          |
|------------------|--------:|---------:|---------------------------------------|
| gaussian_noise_1 |   25    |   100    | BEST — full 3-7MB rollouts, 25->100   |
| motion_blur_1    |    0    |   100    | STRONG 0->100, short clips            |
| viewpoint_medium |    0    |     0    | DROP — both fail at n=4, no contrast  |

Use gaussian_noise + motion_blur pairs for the deck (clear base-fail/AEGIS-succeed).

## Remaining 3 SR pairs to close the 12-cell grid (PAUSED)
- object / viewpoint_medium  (base partial n=50, no AEGIS)
- object / viewpoint_large   (neither arm)
- goal   / gaussian_noise_1  (neither arm)
Est ~$3-4. Resume-skip protects all 9 finished pairs; relaunch
`modal run smolvla_modal.py::main --stage stage1` after reset.

## GR00T N1.5 + AEGIS — wiring DONE (verified on real 3B via L4 smoke), no eval yet
RIB @ backbone.eagle_model.mlp1 (1152->2048, identity@init, 2.46M, fp32);
RASF @ (H=16, d=32) identity@init. Sim-eval pipeline pending. See [[project_modal_pipeline]].

## LIBERO-Plus — image validated (wand + perturbed-bddl import OK), smoke/stage1 NOT wired.

## To resume when budget is reset
1. `modal billing report` to confirm reset, then relaunch stage1 (resume-skip handles the 9 done).
2. Finish the 3 paused pairs first (cheapest path to a full 12-cell grid).
3. Optional: full-length viewpoint video demo; LIBERO-Plus smoke; GR00T N1.5 reproduce+AEGIS.
