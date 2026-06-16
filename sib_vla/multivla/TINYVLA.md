# TinyVLA — Reproduction & Plug-in Integration Brief

> Purpose: reproduce the TinyVLA baseline (paper-grade) on a single A100 80GB, then port our two-part VLA plug-in onto it (IB-Adapter on the visual connector "front"; SIB on the action-chunk "back"). Every claim below is cited to a primary source; where a primary source could not confirm something it is flagged **[UNVERIFIED]**.

---

## 1. Identity

- **Paper title:** *TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation*
- **arXiv id:** [2409.12514](https://arxiv.org/abs/2409.12514) (v5 PDF: https://arxiv.org/pdf/2409.12514v5). Published 2024-09-19. Accepted to **IEEE Robotics and Automation Letters (RA-L) 2025** (accepted 2025-02-06).
- **Authors:** Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, Jian Tang (East China Normal University; Midea Group AI Lab; Syracuse Univ.; Beijing Innovation Center of Humanoid Robotics; Shanghai University).
- **Project page:** https://tiny-vla.github.io/
- **Repo:** https://github.com/liyaxuanliyaxuan/TinyVLA (default branch `main`)
- **HF paper page:** https://huggingface.co/papers/2409.12514

**Architecture summary (one paragraph).** TinyVLA is a compact VLA built by taking a *pre-trained, lightweight multimodal VLM* (a LLaVA-style stack with a **CLIP ViT-L/14-336 vision tower → a linear/MLP projector → a Pythia LLM backbone**, 70M–1.4B total params) and attaching a **diffusion-policy action decoder** during robot-data fine-tuning. Language is fused into the visual backbone via **FiLM** (following RT-1/YAY). Unlike OpenVLA-style autoregressive token VLAs, TinyVLA does **not** require large-scale robot pre-training; it fine-tunes with **LoRA** (only ~5% of params trainable) and replaces autoregressive action-token generation with a **Conditional U-Net 1D diffusion head** conditioned on the LLM hidden states (FiLM conditioning). This yields ~14 ms/action inference (vs OpenVLA ~292 ms) and strong data efficiency.

---

## 2. Architecture breakdown (with file paths & class names)

All paths are repo-relative to https://github.com/liyaxuanliyaxuan/TinyVLA .

| Component | What it is | Where in code |
|---|---|---|
| **Vision encoder** | **OpenAI CLIP ViT-L/14-336** (`openai/clip-vit-large-patch14-336`); SigLIP path also supported. Patch features, `mm_vision_select_layer=-2`, `mm_vision_select_feature="patch"`. | `llava-pythia/llava_pythia/model/multimodal_encoder/clip_encoder.py` (`CLIPVisionTower`); `.../siglip_encoder.py` (`SiglipVisionTower`); selected in `.../model/llava_arch.py` via `build_vision_tower`. |
| **Visual connector / projector** *(IB "front" point)* | Maps vision features → LLM token space. Default in source is `linear` (single `nn.Linear`); **the released smallest checkpoint uses `mlp2x_gelu`** = `Linear(mm_hidden→hidden) → GELU → Linear(hidden→hidden)`. | `llava-pythia/llava_pythia/model/multimodal_projector/builder.py` → `build_vision_projector(config)`. Wired as `self.mm_projector` in `.../model/llava_arch.py` (`LlavaMetaModel`); applied in `encode_images()`. |
| **Language backbone** | **Pythia** (GPT-NeoX family). Sizes ~70M–1.4B across the three variants. | `llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py` → `LLavaPythiaModel` (extends `GPTNeoXModel`) and `LlavaPythiaForCausalLM` (extends `GPTNeoXPreTrainedModel`, `LlavaMetaForCausalLM`). |
| **Action head / decoder** *(SIB "back" point)* | **Diffusion policy head = `ConditionalUnet1D`** (1-D temporal U-Net, FiLM-conditioned on LLM hidden states), DDIM scheduler, `prediction_type='epsilon'`. ACT and FC heads also exist as alternatives. | `policy_heads/models/droid_unet_diffusion.py` → `ConditionalUnet1D`. Built & dispatched in `llava_pythia.py` (`self.embed_out`, `forward_diffusion_head`). |
| **VLM ↔ head linkage** ("linear projection" from the paper) | `self.proj_to_action`: **`nn.Identity()` for the diffusion head** (LLM `hidden_size` feeds the U-Net's `global_cond` directly); an MLP for the ACT head. | `llava_pythia.py` (`__init__` head selection on `config.action_head_type`; `forward()` does `hidden_states = outputs[0]; hidden_states = self.proj_to_action(hidden_states)`). |

**Data flow (forward pass).** images → `CLIPVisionTower` → patch features `[B, N_patch, mm_hidden]` → `mm_projector` → `[B, N_patch, hidden]` → spliced into the text-embedding sequence at `IMAGE_TOKEN_INDEX` (`prepare_inputs_labels_for_multimodal` in `llava_arch.py`) → GPT-NeoX/Pythia → `hidden_states = outputs[0]` → `proj_to_action` (Identity) → `ConditionalUnet1D` produces the action chunk.

---

## 3. Model variants + smallest checkpoint

Three sizes, classified by the multimodal-model scale ([paper §IV-A](https://arxiv.org/abs/2409.12514)):

| Variant | Total params | VLM backbone checkpoint (HF) | Notes |
|---|---|---|---|
| **TinyVLA-S (Small)** ← **SMALLEST** | **~400M** | **`lesjie/Llava-Pythia-400M`** → https://huggingface.co/lesjie/Llava-Pythia-400M | MIT license, **public / not gated**. HF lists "0.4B params", BF16 safetensors. Verified `config.json`: `hidden_size=512`, `num_hidden_layers=6`, `num_attention_heads=8`, `vocab_size=50304`, `model_type="llava_pythia"`, `mm_hidden_size=1024`, vision=`openai/clip-vit-large-patch14-336`, `mm_projector_type="mlp2x_gelu"`. |
| TinyVLA-B (Base) | ~700M | `lesjie/Llava-Pythia-700M` → https://huggingface.co/lesjie/Llava-Pythia-700M | Public, MIT. |
| TinyVLA-H (Huge) | ~1.3B | `lesjie/Llava-Pythia-1.3B` → https://huggingface.co/lesjie/Llava-Pythia-1.3B | Public, MIT. (HF UI reports "1B params" — a rounding/display quirk; README calls it ~1.3B.) |

**Critical caveat — these are VLM *initialization* checkpoints, not trained policies.** The HF repos under `lesjie/` provide only the **pre-trained vision-language model** used to *initialize* TinyVLA. **No released checkpoint contains a trained diffusion action head / fine-tuned VLA policy** (confirmed: README "Pretrained Weights" section links only the three Llava-Pythia VLMs; `lesjie`'s other repos are `scale_dp_l`, `scale_dp_h` (DexVLA-related ScaleDP heads), and dataset `dexvla_example_data`). **To get a working policy you must fine-tune yourself on robot data.** **[Implication: there is no plug-and-play "paper-grade" policy checkpoint to download.]**

The smallest Pythia (`hidden_size=512`, 6 layers) corresponds to roughly Pythia-70M-class LLM weights; "400M" is total including the CLIP-L vision tower (~300M) + projector.

---

## 4. Baseline benchmark + paper-grade target numbers

**The paper's only *simulation* benchmark is MetaWorld — NOT LIBERO, NOT Meta-World-MT50-RL.** (Verified: 0 occurrences of "LIBERO" in the paper.) Everything else is real-robot.

### 4a. Simulation — MetaWorld (50 tasks, multi-task imitation, 50 demos/task, 3 seeds × 5 iters)
Paper **only reports TinyVLA-H** here (Table I). **The smallest model (TinyVLA-S) is NOT evaluated on MetaWorld in the paper** — there is no paper-grade sim number to target for the 400M model. **[UNVERIFIED for TinyVLA-S/B.]**

| Model | Easy (28) | Medium (11) | Hard (6) | Very Hard (5) | **Avg.** |
|---|---|---|---|---|---|
| Diffusion Policy (baseline) | 23.1 | 10.7 | 1.9 | 6.1 | **10.5** |
| **TinyVLA-H** | 77.6 | 21.5 | 11.4 | 15.8 | **31.6** |

### 4b. Real robot — single-arm Franka Panda 7-DoF, 5 tasks (Table II)
Metric = avg success rate over trials; 2× ZED-2 stereo cameras. Diffusion head ablation row (Table V uses TinyVLA-H) gives the per-task best numbers:

| Model | PlaceTennis | FlipMug | StackCubes | CloseDrawer | OpenBox | **Avg.** |
|---|---|---|---|---|---|---|
| Diffusion Policy (111M) | 16.7 | 30 | 3.3 | 73.3 | — | — |
| **TinyVLA-H (diffusion head)** | 90 ±0.2 | 98.3 ±0.1 | 98.3 ±0.1 | 96.7 ±0.3 | 86.7 ±0.1 | **94.0** |
| TinyVLA + ACT head | 13.3 | 8.3 | 8.3 | 13.3 | 23.3 | — |
| TinyVLA + MLP head | 0 | 0 | 0 | 0 | 0 | — |

### 4c. Real robot — bimanual UR5, 3 tasks (Table III; 10 trials)
Tasks: PlaceBread/TransferBread, StackCubes, PlaceTennisBag. DP=111M baseline; OpenVLA (195M trainable) scores **0** on bimanual (no bimanual pre-training data). TinyVLA-H wins. (Per-task numbers partially in source; full table in PDF.)

### 4d. Inference latency (A6000, Table IV)
OpenVLA-7B 292 ms → OpenVLA-1B 140 ms → **TinyVLA-1B 14 ms** (~10–20× faster).

**Bottom line for reproduction targets:** For the **smallest** model there is **no published success-rate target** — the paper's sim/real tables are for TinyVLA-**H**. If you must hit a paper-grade number, the only fully-specified, reproducible-in-sim target is **TinyVLA-H on MetaWorld-50 = 31.6% avg** (and the per-level breakdown above). For the 400M model you will be establishing a *new* baseline, not matching a paper figure.

---

## 5. Official reproduction recipe

### Install (from README)
```bash
git clone https://github.com/liyaxuanliyaxuan/TinyVLA
conda create -n tinyvla python=3.10 -y
conda activate tinyvla
pip install --upgrade pip
pip install -r requirements.txt
cd policy_heads   && pip install -e .   # action heads (ConditionalUnet1D, ACT)
cd ../llava-pythia && pip install -e .   # VLM backbone (LlavaPythia)
```
Key deps: Python 3.10, PyTorch, HuggingFace `transformers`, `deepspeed` (train uses ZeRO-2), `diffusers` (DDIMScheduler), `h5py`. **[Exact pinned versions: check `requirements.txt` in the repo — not fully enumerated here; UNVERIFIED which transformers version.]** Two editable local packages (`policy_heads`, `llava_pythia`).

### Datasets
- **Format:** HDF5 per episode. Verified structure (README):
  ```
  root
   ├─ action               (100, 10)        # 10 = action_dim
   ├─ language_raw         (1,)
   └─ observations
        ├─ images          # multi-view
        │    ├─ left        (100, 480, 640, 3)
        │    ├─ right       (100, 480, 640, 3)
        │    └─ wrist       (100, 480, 640, 3)
        ├─ joint_positions (100, 7)
        ├─ qpos            (100, 7)          # state, state_dim=7
        └─ qvel            (100, 7)
  ```
- **Conversion:** `data_utils/` (README mentions `rlds_to_h5py.py` to go RLDS → HDF5). Register each task in `aloha_scripts/constants.py` with `dataset_dir`, `episode_len`, `camera_names`.
- **No dataset download link is provided in the README** for the paper's MetaWorld or real-robot data. **[UNVERIFIED — the actual training datasets used in the paper are not released here.]** `lesjie/dexvla_example_data` on HF is an *example-format* dataset (DexVLA), useful to validate the pipeline shape but not the paper's data.

### Train (`scripts/train.sh`)
8-GPU DeepSpeed ZeRO-2 by default; LoRA on `vit llm`, r=64, α=256:
```bash
deepspeed --master_port 29600 --num_gpus=8 --num_nodes=1 ./train_tinyvla.py \
  --deepspeed scripts/zero2.json \
  --lora_enable True --lora_module 'vit llm' --load_pretrain False \
  --pretrain_image_size 320 --lora_r 64 --lora_alpha 256 --non_lora_lr 2e-5 \
  --task_name "example_task_config" \
  --model_name_or_path /path/to/Llava-Pythia-400M \
  --version v0 --tune_mm_mlp_adapter True --freeze_vision_tower True \
  --freeze_backbone True --image_aspect_ratio pad --bf16 True \
  --output_dir $OUTPUT --max_steps 10000 --per_device_train_batch_size 32 \
  --save_steps 1000 --learning_rate 2e-4 --lr_scheduler_type cosine \
  --model_max_length 2048 --gradient_checkpointing True \
  --action_head_type droid_diffusion --use_state True --concat "token_cat" \
  --window_size 6 --report_to tensorboard
```
> `$OUTPUT` path must contain the substring `llava_pythia`. Note `train.sh` does **not** pass `--chunk_size/--action_dim/--state_dim`, so the **`train_tinyvla.py` dataclass defaults apply: `chunk_size=16`, `action_dim=10`, `state_dim=7`.**

### Post-process checkpoints (`scripts/process_ckpts.sh`)
Merges DeepSpeed ZeRO shards → `non_lora_trainables.bin` via `zero_to_fp32.py`; rsyncs out `global_step*`. Edit `LLM_MODEL_SIZE` (e.g. `410M`), `source_dir`, `target_dir`.

### Eval
Reference script: **`eval_real_franka.py`** (real Franka robot). **There is NO simulation/MetaWorld eval script in the public repo** — only real-robot eval. **[UNVERIFIED / MISSING: the MetaWorld eval harness used for Table I is not in the repo.]**

### GPU requirements
README states none explicitly. Defaults assume **8 GPUs** (DeepSpeed, `per_device_train_batch_size=32`). On a single A100 80GB you must reduce `--num_gpus=1` and lower batch size / use grad accumulation. **[UNVERIFIED exact single-GPU memory; the 400M model + LoRA should fit comfortably on 80GB.]**

---

## 6. IB / SIB integration points (concrete, with tensor shapes)

### Front — IB-Adapter on the visual connector
- **Hook target:** `build_vision_projector()` output / the `self.mm_projector` module.
  - File: `llava-pythia/llava_pythia/model/multimodal_projector/builder.py`
  - Applied in: `llava-pythia/llava_pythia/model/llava_arch.py`, method **`encode_images()`** (`image_features = self.get_model().mm_projector(image_features)`).
- **Where to inject the IB:** wrap or replace `mm_projector`, or insert your IB-Adapter immediately *after* it (on the projected tokens, before they enter the LLM via `prepare_inputs_labels_for_multimodal`). This is the exact analogue of your SmolVLA "connector/projector" front point.
- **Tensor shapes (smallest model, `Llava-Pythia-400M`):**
  - Projector **input** `[B, N_patch, mm_hidden_size=1024]` (CLIP ViT-L/14-336 patch features; for 336px input, `N_patch=576` (24×24) when select_feature="patch"). **[N_patch UNVERIFIED for the repo's `pretrain_image_size=320` setting — confirm at runtime.]**
  - Projector **output / feature dim D** `[B, N_patch, hidden_size=512]`. **The connector feature dim D you bottleneck = 512 for TinyVLA-S** (= LLM `hidden_size`). For TinyVLA-H, D = the 1.3B Pythia hidden size (read its `config.json`; ~2048).
  - The smallest model's projector is `mlp2x_gelu`: `Linear(1024→512) → GELU → Linear(512→512)`. Insert the IB on the 512-d output, or between the two linears.

### Back — SIB on the action chunk
- **Hook target:** the action chunk produced by the **diffusion head**.
  - Head class: `ConditionalUnet1D` in `policy_heads/models/droid_unet_diffusion.py`.
  - Driven by `forward_diffusion_head(...)` in `llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py`.
- **Action-chunk tensor shape:** `[B, H, action_dim] = [B, chunk_size, action_dim] = [B, 16, 10]` (defaults: **horizon H = chunk_size = 16**, **action_dim = 10**, `state_dim = 7`). The U-Net `input_dim = action_dim`; internally it permutes to `[B, action_dim, H]` for 1-D conv.
- **IMPORTANT — diffusion head changes where SIB attaches.** Because the head is a **denoising diffusion** model (`prediction_type='epsilon'`, DDIM, 100 train / 10 inference steps), the U-Net's direct output is **predicted noise ε**, *not* the action. SIB on "the predicted action chunk before execution" must therefore attach to **the final denoised action**, i.e.:
  - **At inference:** after the iterative DDIM denoising loop completes (in the model's `forward(..., eval=True)` / sampling path), you get the clean action chunk `x_0 ∈ [B, 16, 10]` — apply SIB **there**, before the action is dispatched to the robot. This is the cleanest, most faithful analogue of your SmolVLA SIB point.
  - **At training:** there is no single "predicted action chunk" tensor (the loss is an MSE on ε at random timesteps). Options: (a) apply SIB to the predicted-`x_0` reconstruction `x_0_hat = (x_t − sqrt(1−ᾱ)·ε_pred)/sqrt(ᾱ)` computed from the U-Net output; or (b) apply SIB only at inference. Decide based on whether your SIB is a train-time regularizer or an inference-time filter. **[Design decision — the SmolVLA action head is likely flow-matching/regression with a direct action output; TinyVLA's diffusion head does not expose one at train time, so this is the single biggest porting difference.]**
- **Conditioning note (for completeness):** the LLM `hidden_states [B, seq, hidden_size]` enter the U-Net as `global_cond` (FiLM). If you ever wanted an *information-bottleneck on the conditioning* instead of the output, that is a separate, third option at `proj_to_action` (which is `nn.Identity` for the diffusion head — a convenient place to drop a module).

---

## 7. Risks / gotchas

1. **No trained policy checkpoint exists.** HF `lesjie/*` repos are VLM *initializers* only; you must fine-tune to get any policy. No "download-and-eval" path. (Verified.)
2. **No paper-grade target for the smallest model.** Tables I–V report **TinyVLA-H**, not TinyVLA-S/B. The only fully-reproducible sim target is **TinyVLA-H MetaWorld-50 = 31.6%**. For 400M you set a new baseline. (Verified.)
3. **No MetaWorld eval harness in the repo.** Only `eval_real_franka.py` (real robot). Reproducing Table I requires building/locating a MetaWorld multi-task eval yourself (MetaWorld env + 50-task config + the 3-seed×5-iter protocol). **[MISSING in repo.]**
4. **No training data released.** Neither the MetaWorld demo dataset nor the real-robot HDF5 data is linked. You must generate/collect demos and convert to the HDF5 schema. (`dexvla_example_data` only shows the format.) **[MISSING.]**
5. **8-GPU DeepSpeed assumed.** `train.sh` uses `--num_gpus=8`, ZeRO-2, batch 32. Single-A100 requires editing num_gpus/batch/grad-accum. (Verified from script.)
6. **Real-robot-only original eval.** The flagship 90–98% numbers are physical Franka/UR5 — not reproducible without the hardware. Your reproducible surface is sim (MetaWorld) only.
7. **Config drift between source defaults and checkpoints.** Repo source defaults (`mm_hidden_size=768`, `hidden_size=2560`, `mm_projector_type="linear"`, CLIP image_size 224/patch 32) are **placeholders** overridden by each checkpoint's `config.json` (smallest = `mm_hidden_size=1024`, `hidden_size=512`, `mlp2x_gelu`, CLIP ViT-L/14-**336**). Always read the actual `config.json`, not the dataclass defaults. (Verified by diffing source vs HF config.json.)
8. **Dependency pinning unknown.** `requirements.txt` versions (esp. `transformers`, `diffusers`, `deepspeed`, `peft`) not enumerated here; LLaVA + GPT-NeoX + DeepSpeed stacks are version-sensitive. **[UNVERIFIED — pin before training.]**
9. **Diffusion head ⇒ SIB attach point differs from SmolVLA** (see §6). Train-time has no clean action tensor.
10. **`action_dim=10` but real data shows `action (100,10)`** while `state_dim=7` — confirm the 10-dim action layout (likely 2× (6-DoF + gripper) for bimanual, or 9-DoF+gripper) matches your robot before training. **[Layout UNVERIFIED.]**

---

## 8. Recommended repro plan — single A100 80GB

**Goal:** get the *smallest* TinyVLA training+eval loop running end-to-end, then attach IB/SIB. Accept that there is no 400M paper number to match; target a *self-consistent* MetaWorld baseline (and, if you want a paper-grade anchor, also run TinyVLA-H → MetaWorld 31.6%).

**Phase 0 — Environment (½ day).**
1. `conda create -n tinyvla python=3.10`; clone; `pip install -r requirements.txt`; `pip install -e policy_heads`; `pip install -e llava-pythia`. **Pin** `transformers`, `diffusers`, `deepspeed`, `peft` exactly from `requirements.txt`; snapshot the working env.
2. Download `lesjie/Llava-Pythia-400M` (public, MIT) and read its `config.json` to lock D=512, mm_hidden=1024, CLIP-L/14-336.

**Phase 1 — Data path validation (1 day).**
3. Pull `lesjie/dexvla_example_data` (or convert a handful of MetaWorld demos with `data_utils/rlds_to_h5py.py`) into the HDF5 schema from §5. Register a task in `aloha_scripts/constants.py`.
4. Confirm shapes: `action (T,10)`, `qpos (T,7)`, images `(T,480,640,3)`. Smoke-test the dataloader.

**Phase 2 — Single-GPU training smoke test (1 day).**
5. Edit `scripts/train.sh`: `--num_gpus=1`, drop `per_device_train_batch_size` to 4–8 + `--gradient_accumulation_steps` to keep effective batch ~32, keep `--action_head_type droid_diffusion`, LoRA r=64. Run a short `--max_steps 200` to confirm the diffusion loss decreases and checkpoints write.
6. Run `scripts/process_ckpts.sh` (set `LLM_MODEL_SIZE=410M`) to merge ZeRO shards → `non_lora_trainables.bin`.

**Phase 3 — Establish a sim baseline (2–4 days).**
7. Stand up a MetaWorld eval (env install + 50-task config; replicate the 3-seed × 5-iter, 50-demo multi-task protocol from §4a). Train 400M to convergence (`--max_steps 10000`), measure avg success. This is your *new* TinyVLA-S baseline.
8. (Optional anchor) Repeat with `Llava-Pythia-1.3B` to reproduce **TinyVLA-H = 31.6%** and validate your harness against the paper.

**Phase 4 — Plug-in port.**
9. **IB front:** subclass/wrap `mm_projector` in `multimodal_projector/builder.py` (or insert after it in `llava_arch.encode_images`); bottleneck the `D=512` projected tokens.
10. **SIB back:** hook the inference denoising output (clean `x_0 ∈ [B,16,10]`) in `forward_diffusion_head`/sampling path in `llava_pythia.py`; if you need train-time SIB, compute `x_0_hat` from the predicted ε (see §6). Re-run Phase 3 to measure delta vs your baseline.

---

### Source index
- Paper: https://arxiv.org/abs/2409.12514 · PDF https://arxiv.org/pdf/2409.12514v5 · HF https://huggingface.co/papers/2409.12514
- Project page: https://tiny-vla.github.io/
- Repo: https://github.com/liyaxuanliyaxuan/TinyVLA
  - `llava-pythia/llava_pythia/model/multimodal_projector/builder.py` (projector)
  - `llava-pythia/llava_pythia/model/llava_arch.py` (vision tower + projector wiring, `encode_images`)
  - `llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py` (backbone + head linkage, `forward_diffusion_head`)
  - `llava-pythia/llava_pythia/model/language_model/pythia/configuration_llava_pythia.py` (configs)
  - `policy_heads/models/droid_unet_diffusion.py` (`ConditionalUnet1D`)
  - `train_tinyvla.py` (ActionArguments defaults: chunk_size=16, action_dim=10, state_dim=7)
  - `scripts/train.sh`, `scripts/process_ckpts.sh`, `eval_real_franka.py`
- Checkpoints: https://huggingface.co/lesjie/Llava-Pythia-400M · /Llava-Pythia-700M · /Llava-Pythia-1.3B
  - Verified `Llava-Pythia-400M/config.json`: hidden_size=512, layers=6, heads=8, mm_hidden_size=1024, vision=openai/clip-vit-large-patch14-336, mm_projector_type=mlp2x_gelu
