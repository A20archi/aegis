# A6000 RUN QUEUE — how to execute everything (restartable)

All work is queued so it runs unattended and **resumes cleanly if interrupted**.
Every stage checks for its own output and skips if already done.

## Order of operations on the A6000
1. Finish `NEW_MACHINE_CHECKLIST.md` (paths identical, envs rebuilt, data restored).
2. **SmolVLA full pipeline** (one command, resumable):
   ```
   conda activate smolvla
   cd <repo>/sib_vla
   bash ../migration/a6000_queue.sh            # stage0->stage4, end to end
   ```
   Stages:
   - **stage0** resume base **12k → 30k** (batch auto-ladder 160→128→96; lr stays on the
     resumed cosine schedule; loss already plateaued at 0.062 so this just completes the
     checkpoint ladder for peak-pick).
   - **stage1** eval base+TE clean on ckpts **12k–30k** → pick the **SR peak** (overfit guard).
   - **stage2** retrain RIB(12k)+RASF(8k) **on the peak ckpt**.
   - **stage3** AEGIS clean n=200.
   - **stage4** robustness table (4 axes, videos) + noise sweep σ0.05→1.0, both arms.
   Run a single stage with `bash ../migration/a6000_queue.sh stage2` etc.
3. **NanoVLA** (cross-arch #1) — injection already built (`nanovla/aegis_nanovla.py`):
   ```
   python scripts/finetune_nanovla.py --steps 100000 --batch-size 64 --out results/nanovla_s
   # then RIB+RASF train on the NanoVLA base, then robustness eval (mirror run_v2_pipeline)
   ```
4. **GR00T N1.5** (cross-arch #2):
   ```
   conda activate gr00t && cd /home/user/vla/Isaac-GR00T && git checkout n1.5-release
   # write gr00t/aegis_groot.py (RIB@Eagle visual output, RASF@H=16 DiT chunk) mirroring aegis_nanovla.py
   # baseline finetune on LIBERO (add --no-tune_diffusion_model or LoRA if 48GB tight) -> module train -> eval
   ```

## FAST PATH for the "beat 87.3 average" goal — `stage5` (no retrain)
The 86 base is trained on **all 4 LIBERO suites** (40 tasks), so the other 3 suites need
only EVALUATION, not training — same base+TE+AEGIS recipe that gave 86 on Spatial:
```
conda activate smolvla && cd <repo>/sib_vla
bash ../migration/a6000_queue.sh stage5        # Spatial/Object/Goal/Long clean, base+TE & AEGIS
EP=10 bash ../migration/a6000_queue.sh stage5  # faster (n=100/suite) if A6000 time is tight
```
Uses `smolvla_spatial_repro/020000` + `rib_on86.pt`/`rasf_on86.pt`. ~10h @ n=200 on A6000;
prints a 4-suite table + average vs the 87.3 target. Restartable (skips done suites/arms).
Honest expectation: avg ~84-85 (Long caps it at ~71; Object/Goal are unmeasured wildcards).

## The ONE thing to verify first (resume + changed batch)
lerobot resume restores weights+optimizer+step from the 12k checkpoint. We override
`--batch_size` to fit 48GB. Most lerobot 0.4.x builds apply CLI overrides on top of the
resumed config — but confirm on the first launch: stage0 detects an OOM (ladders down) and
also detects a resume/config rejection (prints a fallback). **If it rejects the batch
override**, use the weight-init fallback:
```
# continue from 12k weights as a fresh short run at the A6000 batch (loses optimizer state,
# negligible since converged):
cp -r outputs/smolvla_spatial_v2/checkpoints/012000 outputs/smolvla_spatial_v2/checkpoints/_init12k
bash run_base_retrain.sh 18000 160 5e-5     # 18k more steps @ batch160; edit script to load _init12k as start
```
(Only needed if true resume won't take the batch change — try the queue first.)

## Why it's "restartable as it is"
- stage0 uses lerobot's native resume → re-running continues from the newest checkpoint.
- stages 1–4 skip any step whose output JSON / .pt already exists.
- progress + the chosen peak ckpt are written to `migration/a6000_run.log`,
  `migration/peak_pick.txt`, `migration/peak_ckpt.txt`.
So if the box reboots or a stage dies, just re-run the same command.

## VRAM fallbacks (48GB)
- base resume OOM → handled automatically (batch ladder).
- GR00T finetune OOM → add `--no-tune_diffusion_model` or `--lora_rank 64 --lora_alpha 128`.
- any eval OOM → drop `--episodes 20` to `10` (n=200→100, wider CI).
