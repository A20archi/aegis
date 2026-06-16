# A6000 SETUP CHECKLIST — run top-to-bottom on the new box

Goal: identical paths, recreated envs, restored data, resumed Claude, then unpause tasks.

## 0. Prereqs on the A6000
- [ ] Same username + home so paths match: `/home/user/...` (CRITICAL — configs + Claude encode absolute paths).
- [ ] NVIDIA driver + CUDA present: `nvidia-smi` (note CUDA version), 48GB visible.
- [ ] conda/miniconda installed at `/home/user/miniconda3` (and anaconda3 if you keep that layout).
- [ ] Tailscale up: `tailscale ip -4`.

## 1. Pull the data (run migrate.sh FROM the A100 box)
- [ ] On A100: edit `migration/migrate.sh` DEST_HOST, then `bash migrate.sh --dry` → `bash migrate.sh`.
- [ ] Verify on A6000: `du -sh ~/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi` (~21G), repo `outputs/` present.

## 2. Recreate conda envs (do NOT copy env dirs — rebuild from spec)
SmolVLA / NanoVLA env (lerobot 0.4.3, torch 2.6.0+cu124):
```
conda create -n smolvla python=3.10 -y && conda activate smolvla
pip install -r migration/pip_smolvla_base.txt        # exact pins
# if a torch/cuda pin mismatches the A6000 driver, install matching torch first, then -r
python -c "import lerobot, torch; print(lerobot.__version__, torch.cuda.is_available())"
```
(NOTE: training ran from anaconda3 *base*; a clean named env `smolvla` is cleaner. `env_smolvla_base.yml` is the conda-level export if you prefer `conda env create -f`.)

GR00T env (torch 2.5.1+cu124, flash-attn must be REBUILT for this box's CUDA):
```
conda create -n gr00t python=3.10 -y && conda activate gr00t
cd /home/user/vla/Isaac-GR00T && git checkout n1.5-release
pip install --upgrade setuptools && pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4   # rebuilds for local CUDA
python -c "import gr00t, torch; print('gr00t ok', torch.cuda.is_available())"
```

## 3. HF cache
- [ ] migrate.sh copied the critical subset to `~/.cache/huggingface/hub`. Anything missing
      auto-downloads on first use (incl. bert-base-uncased for NanoVLA, ~440M).

## 4. Resume Claude context
- [ ] `cd /home/user/Desktop/SAPTARSHI_ALT/steer_information_Saptarshi`
- [ ] `claude --resume`  → pick session `dfc04c18-...` (this conversation).
      If it doesn't list, the `memory/` still auto-loads (knowledge preserved) — and
      `migration/STATE_HANDOFF.md` has the full plan as a fallback. Just start `claude` and
      say "read migration/STATE_HANDOFF.md and continue".

## 5. Smoke-verify before real runs
- [ ] SmolVLA env: `cd sib_vla && python -c "import sys;sys.path.insert(0,'.');import sib;print('sib ok')"`
- [ ] NanoVLA injection: `cd sib_vla && CUDA_VISIBLE_DEVICES="" python scripts/smoke_aegis_nanovla.py`  (expect ALL CHECKS PASSED)
- [ ] A checkpoint loads: eval one SmolVLA ckpt at small n to confirm rollouts work on the A6000.
- [ ] GR00T: `cd /home/user/vla/Isaac-GR00T && python -c "from gr00t.model.gr00t_n1 import GR00T_N1_5; print('ok')"`

## 6. Unpause the task queue (see STATE_HANDOFF.md "Pending task queue")
Order on A6000: finish SmolVLA (if not done on A100) → NanoVLA → GR00T → assemble Jun-30 deck.
Watch VRAM: if GR00T finetune OOMs at 48GB, add `--no-tune_diffusion_model` or use LoRA
(`--lora_rank 64 --lora_alpha 128`).
