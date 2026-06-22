# Modal AEGIS — saved state (2026-06-21, spend STOPPED at $98.31 / $100)

## Spend status
All Modal apps stopped + local CLI clients killed; $0/hr now. Budget **$98.31 / $100**
(~$1.69 left). Resume after supervisors reset the cap.
Track real spend: `modal billing report --for "this month" --json` (URL:
https://modal.com/settings/seo-40141/usage?tab=usage).

## !! MANDATORY: launch Modal ONLY via safe_modal_run.sh (NEVER bare `modal run`).
Two preventable runaways once ate 93% of a $100 budget: (1) an UNCAPPED 24-cell burst
($40.54), (2) an ORPHANED `modal run` client that survived ~11h after apps were
"stopped" and silently re-spent ($50.82). `modal app stop` does NOT kill the local
client; the app also showed under a non-"running" state label so `app list | grep
running` missed it.

FIX — both failure modes are now structurally impossible:
  sib_vla/multivla/safe_modal_run.sh   # ONLY sanctioned launcher:
     - PREFLIGHT kills stray clients + stops running apps (clean slate)
     - TRAP kills the client + stops its app on ANY exit -> can't orphan
     - WATCHDOG polls real billing; auto-aborts if this run adds > BUDGET_GUARD
       ($15 default) or month-to-date crosses HARD_TOTAL ($95 default)
  sib_vla/multivla/modal_killall.sh    # emergency teardown (kill all clients+apps)
Usage: BUDGET_GUARD=10 HARD_TOTAL=95 ./safe_modal_run.sh modal run <app>::main --stage X
Before declaring spend halted, run modal_killall.sh (or: pgrep -af "modal run";
modal app list --json checking ALL states).

## SmolVLA LIBERO-V robustness grid (base=gate-closed, AEGIS=gate-open, +TE, n_action_steps=1)
10 complete pairs, **mean delta +29.9 pts, 0 regressions** (gate provably never worse):

| suite  | condition        | base | AEGIS | delta  | n   |
|--------|------------------|-----:|------:|-------:|-----|
| object | motion_blur_1    |   0  |  86   | +86.0  | 100 |
| object | gaussian_noise_1 |  36  |  90   | +54.5  | 200 |
| object | lighting_1       |  58  |  92   | +34.0  | 100 |
| object | texture_1        |  83  |  97   | +14.0  | 200 |
| object | viewpoint_medium |   0  |   0   |  +0.0  | 100 |
| goal   | motion_blur_1    |  19  |  78   | +59.0  | 100 |
| goal   | viewpoint_medium |  17  |  43   | +26.0  | 100 |
| goal   | viewpoint_large  |   8  |  29   | +21.0  | 100 |
| goal   | texture_1        |  90  |  93   |  +3.0  | 100 |
| goal   | lighting_1       |  80  |  82   |  +2.0  | 100 |

object/viewpoint_medium is an honest +0 wash (both arms fail; not a regression).
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

## Remaining 2 SR pairs to close the 12-cell grid (STOPPED, resume after reset)
- object / viewpoint_large   (base partial n=20, no AEGIS)
- goal   / gaussian_noise_1  (neither arm)
Est ~$2-3. Resume-skip protects all 10 finished pairs. Relaunch via the guarded
launcher (NEVER bare `modal run`):
  cd sib_vla/multivla && BUDGET_GUARD=8 HARD_TOTAL=<reset+budget-5> \
    ./safe_modal_run.sh modal run smolvla_modal/smolvla_modal.py::main --stage stage1

## GR00T N1.5 + AEGIS — wiring DONE (verified on real 3B via L4 smoke), no eval yet
RIB @ backbone.eagle_model.mlp1 (1152->2048, identity@init, 2.46M, fp32);
RASF @ (H=16, d=32) identity@init. Sim-eval pipeline pending. See [[project_modal_pipeline]].

## LIBERO-Plus — image validated (wand + perturbed-bddl import OK), smoke/stage1 NOT wired.

## To resume when budget is reset
1. `modal billing report` to confirm reset, then relaunch stage1 (resume-skip handles the 9 done).
2. Finish the 3 paused pairs first (cheapest path to a full 12-cell grid).
3. Optional: full-length viewpoint video demo; LIBERO-Plus smoke; GR00T N1.5 reproduce+AEGIS.
