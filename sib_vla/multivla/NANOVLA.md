# NanoVLA — Reproduction & Plug-in Porting Brief

> Research brief for porting our **IB-Adapter (front)** + **SIB (back)** plug-in onto NanoVLA's smallest model.
> Compiled 2026-06-13 from primary sources. Every non-trivial claim is cited to the arXiv paper (full text), HF paper page, or OpenReview.
> **Headline caveat:** there is currently **NO official public code repo and NO HuggingFace checkpoint** for NanoVLA, and the OpenReview submission has been **withdrawn**. See §3 and §7.

---

## 1. Identity

- **Confirmed title:** *NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies*. Confirmed via arXiv metadata and HF paper page. ([arXiv abstract](https://arxiv.org/abs/2510.25122), [HF paper page](https://huggingface.co/papers/2510.25122))
- **arXiv id:** `2510.25122` (v1). Category `cs.RO`. Submitted **2025-10-29**. PDF: https://arxiv.org/pdf/2510.25122v1 ; HTML: https://arxiv.org/html/2510.25122v1
- **Authors:** Jiahong Chen* (University of British Columbia), Jing Wang (University of Alberta), Long Chen (Xiaomi EV), Chwei Cai (Northeastern University), Jinghui Lu† (Xiaomi EV, corresponding `lujinghui@xiaomi.com`). (* equal contribution, † corresponding.) Source: paper header.
- **Year:** 2025 (preprint). Venue attempt: **ICLR 2026 — submission WITHDRAWN** per OpenReview. ([OpenReview forum yeHBrNVZoV](https://openreview.net/forum?id=yeHBrNVZoV))
- **Repo URL:** **None published.** No GitHub link in the paper, HF page, abstract, alphaXiv, or OpenReview; GitHub repo search returns nothing. The paper references LeRobot ([github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)) only as the framework it builds on / baselines from, not as a NanoVLA code release.

**One-paragraph architecture summary.** NanoVLA is *not* a SmolVLA/OpenVLA-style decoder-only LLM with a vision projector. It is a **lightweight ACT/DETR-style transformer encoder–decoder with a CVAE latent** (explicitly "following traditional action chunking process Zhao et al. 2023" = ACT). Each modality is encoded **independently and frozen**: a frozen vision backbone (ResNet18) gives a `C×H×W` map → 1×1 conv → flattened image tokens; a frozen language encoder (BERT-base for `-S`, Qwen2.5-0.5B for `-L`) gives one language token; proprioception/state/env are MLP-projected to 1D tokens; a per-chunk VAE latent `z` is another 1D token. These are concatenated and run through a **lightweight transformer encoder (4 layers)**; **late fusion happens only once** in the **transformer decoder via cross-attention** (decoder action-query slots attend to the encoder memory). A final linear **action head** maps each decoder slot to an action vector, producing an `[H × action_dim]` chunk. Two efficiency add-ons: **long-short action chunking** (train on long `H_train`, execute only first `h≪H_train` then replan) and a **dynamic MCB router** that picks the small (`-S`) or large (`-L`) language backbone per task. Source: paper §3 + Appendix A.1.

---

## 2. Architecture breakdown

All from paper §3.1, §4, and **Appendix A.1 "Transformer Architecture and Late-Fusion Details"** (the most load-bearing section for us).

| Component | What NanoVLA uses | Notes / source |
|---|---|---|
| **Vision encoder** | **ResNet18**, frozen. All three variants use ResNet18. | §4: "All models were using ResNet18 as the image encoder". Outputs `C×H×W` feature map. |
| **Language backbone** | **BERT-base** (NanoVLA-S) or **Qwen2.5-0.5B** (NanoVLA-L), frozen. | §4. Encoder-only BERT for `-S`; decoder LLM Qwen2.5-0.5B for `-L`. Both frozen during training (§3.1). |
| **Other modalities** | proprio/state, env, VAE latent `z` → linear/MLP to dim `D`, each a single 1D token. | A.1 "Input Projections and Token Layout". |
| **Connector / projector (our IB FRONT point)** | Vision: ResNet map → **1×1 conv `W_img`** → per-location dim `D` → flatten to `X_img ∈ ℝ^{N_img × D}`. Language: linear `x_lang = l`. These tokens then enter the **transformer encoder**. | A.1 "Visual features … projected by a 1×1 conv `W_img` to `D` per spatial location, then flattened to tokens". This is the analog of SmolVLA's visual connector. |
| **Fusion** | **Late fusion via cross-attention in the transformer decoder** (decoder slots = queries; encoder memory = keys/values). Encoder = 4 pre-norm layers; decoder = `L_dec` layers + final LayerNorm. | A.1 "Transformer Decoder (Late Fusion via Cross-Attention)". |
| **Action head / decoder type** | **Regression** (deterministic linear head on decoder slots), **CVAE-regularized** (ACT-style). NOT diffusion, NOT flow-matching. `A = Y_final · W_act + b_act ∈ ℝ^{H × d_act}`. | A.1 "Action Head". Loss = L1/L2 chunk regression + β·KL (eq. in A.1 "Variational Training Objective"). |
| **Action production (our SIB BACK point)** | Decoder produces `H` slots `Y ∈ ℝ^{H × D}` → linear head → action chunk `A ∈ ℝ^{H × d_act}`. At inference `z=0` (deterministic). | A.1. `H` is "chunk size in code". |
| **Routing** | MCB (Monte-Carlo-Beta) Bayesian win-probability router picks `-S` vs `-L` LLM at inference. Separate text-conditioned binary classifier. | §3.3. Irrelevant to our two attachment points. |

**Key tensor-shape facts (from A.1, verbatim notation):**
- `D` = latent/model dim (the connector output dim and the encoder/decoder working dim). **Exact value not stated in paper** — must be read from config when code/weights appear (see §7).
- Image tokens: `X_img ∈ ℝ^{N_img × D}` where `N_img` = flattened ResNet18 final-stage spatial map (e.g., 7×7=49 for 224² input — *infer, not stated*).
- Encoder sequence: `X = concat([x_lat, x_state, x_env, x_lang, X_img]) ∈ ℝ^{N × D}`, `N = 4 + N_img`.
- Encoder output (the fusion memory) `H ∈ ℝ^{N × D}`. Encoder depth `L_enc = 4`.
- Decoder action slots `Y ∈ ℝ^{H × D}`, `H` = chunk size.
- **Action chunk output `A ∈ ℝ^{H × d_act}`.** `H_train = 100` AC steps, inference executes `h = 10` AC steps (§4: "trained with 100 AC steps with 10 AC steps during inference"). `d_act` not numerically stated; for LIBERO it is 7 (6-DoF + gripper) by convention — *infer*.
- CVAE latent `z`: low-dim per-chunk; posterior `q_φ(z|state,actions)` from a CLS-token head; prior `N(0,I)`; β-weighted KL. (This KL is already an information bottleneck on `z` — relevant context for our SIB, see §6.)

**Module/class names + file paths:** **Not available** — no code released. The paper's own wording ("chunk size in code", "as in code", "matching implementation") implies an internal codebase derived from **ACT / LeRobot**. Closest reference implementations to mirror: LeRobot's ACT policy (`lerobot/common/policies/act/modeling_act.py` — `ACT`, `ACTEncoder`, `ACTDecoder`, action head) and SmolVLA policy in the same repo. Treat these as the structural template until NanoVLA code appears.

---

## 3. Model variants + smallest checkpoint

From **Table 1** (LIBERO) of the paper:

| Variant | Language backbone | **Total params** | Trainable params | Smallest? |
|---|---|---|---|---|
| **NanoVLA-S** | BERT-base | **161M** | 52M | **✅ SMALLEST (by total params)** |
| NanoVLA-L | Qwen2.5-0.5B | 520M | 52M | |
| NanoVLA-R | router over S+L | 296M* (251M avg elsewhere) | 52M | *avg of L/S invocation |

- **Smallest model = NanoVLA-S = 161M total / 52M trainable** (BERT-base + ResNet18 + light enc/dec). Note all three share the same 52M trainable action stack; size differs only by frozen LLM.
- **Exact HuggingFace checkpoint id(s) / URL(s):** **NONE EXIST.** HF model search filtered by `arxiv:2510.25122` returns **0 models** ([HF models?arxiv:2510.25122](https://huggingface.co/models?other=arxiv:2510.25122)). No checkpoints linked from paper/HF/OpenReview. **Gated? No — simply not released.** This is the single biggest blocker (see §7/§8).
- Trainable backbones we'd reuse (public, not NanoVLA weights): `google-bert/bert-base-uncased`, `Qwen/Qwen2.5-0.5B`, torchvision `ResNet18_Weights.IMAGENET1K_V1`.

---

## 4. Baseline benchmark + paper-grade target numbers

**Benchmarks:** (a) **LIBERO** (simulation, primary reproducible target), (b) **LIBERO-90**, (c) real-world LeRobot SO-ARM-101 (12 tasks — not reproducible without their robot/data), (d) Jetson Orin Nano latency.

**Protocol (paper §4.1):** follows OpenVLA's LIBERO setup; **SR averaged over 50 trials per suite**. For π0/SmolVLA they re-ran LeRobot defaults; OpenVLA/TraceVLA/SpatialVLA/Octo are quoted from original papers.

### LIBERO four-suite success rate (%) — Table 1 (reproduce these)

| Policy | Total | Trainable | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|---|---|
| **NanoVLA-S (smallest)** | 161M | 52M | **81.6** | **93.6** | **89.6** | **49.8** | **78.7** |
| NanoVLA-L | 520M | 52M | 87.2 | 89.8 | 90.0 | 55.2 | 80.4 |
| NanoVLA-R | 296M* | 52M | 89.8 | 96.2 | 93.0 | 57.4 | 84.1 |
| SmolVLA (ref) | 450M | 100M | 72.8 | 69.8 | 84.0 | 52.6 | 78.6 |
| OpenVLA | 7.5B | 279M | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |

→ **Paper-grade target for the smallest model (NanoVLA-S): Spatial 81.6 / Object 93.6 / Goal 89.6 / Long 49.8 / Avg 78.7.**

### LIBERO-90 (%) — Table 2

| Model | SmolVLA | OpenVLA | **NanoVLA-S** | NanoVLA-L | NanoVLA-R |
|---|---|---|---|---|---|
| LIBERO-90 | 68.9 | 62.0 | **55.1** | 83.3 | 81.6 |

→ NanoVLA-S is **deliberately weak on LIBERO-90 (55.1%)** — paper attributes this to BERT's encoder-only language being weak at large-scale instruction following. Expect this and don't over-tune.

**Latency (LIBERO-Goal, Jetson Orin Nano 8GB):** 52× FPS vs OpenVLA at +13.8% SR; vs SmolVLA at matched 50 AC steps, NanoVLA SR ~87.2% (−minor) with +43.8% FPS. Caching saves 62% inference time (Qwen-0.5B backbone) / 35% (BERT). (§4.3, §4.4.) — context, not a repro target for us.

---

## 5. Official reproduction recipe

**There is no official recipe (no code).** The paper gives only these concrete, citable knobs. Everything else must be reconstructed on a LeRobot/ACT base.

- **Framework:** LeRobot ([github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)) — paper re-runs π0/SmolVLA "from LeRobot code base with default setting" (§4.1) and builds real-robot stack on LeRobot. **No version pinned.**
- **Datasets:**
  - **LIBERO** (Liu et al. 2023a) — Spatial/Object/Goal/Long + LIBERO-90. Obtain via the official LIBERO repo / OpenVLA's processed LIBERO RLDS (the paper "follows OpenVLA to use tasks from" these suites). Standard public download.
  - Real-world LeRobot data: 50 demos/task on SO-ARM-101, Intel RealSense D435i third-person RGB, 30 Hz, 1280×720 → **not publicly released; not reproducible.** (Appendix A.2.)
- **Training hyperparameters (the only stated numbers):**
  - Action chunking: **`H_train = 100` AC steps; inference executes `h = 10`** (§4).
  - Encoder depth **4 layers**; decoder + CVAE per ACT; loss = L1/L2 chunk regression **+ β·KL(q_φ(z)‖N(0,I))** (A.1).
  - At inference set **`z = 0`** (deterministic, no latency cost).
  - Vision + language encoders **frozen**; only the ~52M enc/dec/heads train.
  - "Finetuning on LIBERO usually takes **~1M steps** to converge" (Appendix A.5).
- **Train/eval commands:** none given. Use LeRobot's `lerobot/scripts/train.py` + LIBERO eval harness (OpenVLA-style 50-trial rollouts) as the substrate.
- **GPU requirements:** training profiled on **1× H20**, batch 1–32; also runs on **RTX 3060** and **Jetson Orin Nano**; "~30–40 GPU-hours" on edge devices for LIBERO finetune (Appendix A.5). → **Comfortably fits one A100 80GB** (it trained on a single H20 and even a 3060).

---

## 6. IB/SIB integration points

This is the decision-critical section. **NanoVLA's architecture differs from SmolVLA** (encoder-decoder CVAE vs decoder-LLM + projector), so map carefully.

### 6.1 IB-Adapter (FRONT) — visual connector
- **Where:** the **visual projection that maps ResNet18 features into the encoder token space**, i.e. the `1×1 conv W_img` (+ flatten + positional add) that produces `X_img ∈ ℝ^{N_img × D}`, *before* these tokens enter the 4-layer transformer encoder. This is the exact analog of SmolVLA's visual connector/projector where we already attach IB.
- **Hook:** wrap the output of `W_img` (the conv) — insert the IB-Adapter on `X_img` tokens. (If you prefer post-encoder fusion memory, you could alternatively hook `H` after the encoder, but that mixes language/state; the **clean front analog is on `X_img` only**.)
- **Tensor shape at hook:** `X_img ∈ ℝ^{B × N_img × D}` (the connector feature dim is **`D`**). `N_img` = flattened ResNet18 final-stage grid (≈49 for 224² input — *verify in config*). **`D` value is not in the paper — read it from the released config (none yet) or set it yourself when you re-implement.**

### 6.2 SIB (BACK) — action-chunk output
- **Where:** the **action head output `A = Y_final·W_act + b_act ∈ ℝ^{H × d_act}`** — the predicted action chunk before execution. Identical conceptual point to where SIB sits on SmolVLA's action chunk.
- **Hook:** apply SIB on `A` (or equivalently on `Y_final` decoder slots just before `W_act`, depending on whether you want SIB in action space or feature space — your SmolVLA SIB is on the **action chunk tensor**, so hook `A`).
- **Tensor shape at hook:** action chunk **`A ∈ ℝ^{B × H × d_act}`**. **Horizon `H = 100` (training chunk)**; at inference only first `h=10` rows are executed — decide whether SIB regularizes the full 100 or the executed 10 (full chunk is the supervised tensor, matching the chunk loss). **`d_act` not numerically stated** (7 for LIBERO by convention — verify).
- **Action-head type note (important for SIB):** NanoVLA's head is **deterministic regression with a CVAE latent**, **NOT diffusion/flow-matching.** So SIB attaches directly to a single forward-pass action tensor `A` — **no iterative denoising/integration loop to thread through** (unlike π0 flow-matching or DDPM heads). This is the *easy* case for SIB and matches your SmolVLA setup.
- **Interaction caveat with the existing CVAE-KL:** NanoVLA *already* contains an information bottleneck — the **β·KL on the per-chunk latent `z`** (A.1 explicitly calls this "an information bottleneck on z"). Our SIB (spectral IB on the action chunk) is a *different* bottleneck (output-space/spectral vs latent-space/KL). They are complementary but **both regularize the same chunk**; budget for tuning β (CVAE) vs the SIB weight jointly, and consider an ablation isolating each.

---

## 7. Risks / gotchas

1. **No code, no weights, withdrawn submission (BIGGEST RISK).** No GitHub, no HF checkpoint, OpenReview withdrawn. Reproduction = **re-implementation from the paper on a LeRobot/ACT base**, not a clone-and-run. Porting our plug-in therefore means porting onto *our* re-implementation, and "paper-grade" numbers are a moving target.
2. **Underspecified hyperparameters.** Model dim `D`, `N_img`, `d_act`, decoder depth `L_dec`, heads `h`, FFN dim, β, optimizer/LR/schedule, batch size, and the exact LIBERO preprocessing are **not in the paper.** Matching 78.7% avg without these is non-trivial.
3. **NanoVLA-S is genuinely weak on long-horizon / LIBERO-90** (Long 49.8, LIBERO-90 55.1). Don't mistake that for a repro bug — it's expected for the BERT variant.
4. **50-trial SR has high variance** (your own MEMORY notes the n-size sensitivity on LIBERO-Long; paper uses n=50, some baselines text says n=50 / table says "average over 50 test trials" while §4.1 also says "average of 50 independent trials"). Use n=50 to compare apples-to-apples.
5. **CVAE + frozen encoders interaction.** Both vision and language frozen; only ~52M trains. Freezing is load-bearing for their environmental-robustness claims (§4.4) — keep encoders frozen when porting IB (the IB-Adapter is the *trainable* path into a frozen vision encoder, which fits this design well).
6. **Two bottlenecks stacking** (CVAE-KL on `z` + our SIB on `A`) may interact; needs an ablation (see §6.2).
7. **Routing (`-R`) is out of scope** for the smallest-model port and adds a separate classifier + dual backbones — ignore for the baseline.
8. **Author affiliation churn** (Xiaomi EV / UBC / UAlberta) and a withdrawn submission lower the odds of an imminent official release — don't block on waiting for code.

---

## 8. Recommended reproduction plan (single A100 80GB)

Compute is **not** the constraint (it trained on one H20 / even RTX 3060). The constraint is **no code**. Plan:

1. **Confirm no release one more time before building** (cheap): re-check HF (`models?other=arxiv:2510.25122`), the authors' GitHub (Jiahong Chen / Jinghui Lu), and OpenReview supplementary zip for any dropped code. If anything appears, switch to clone-and-run.
2. **Adopt LeRobot ACT as the substrate.** NanoVLA = ACT + (frozen ResNet18 vision) + (frozen BERT-base language token) + (late-fusion cross-attn decoder) + (long-short chunking) + (CVAE-KL). Start from LeRobot's ACT policy and swap: vision backbone → frozen ResNet18 with `1×1 conv` projector; add a frozen BERT-base language token; set `chunk_size H_train=100`, execute `h=10`.
3. **Pin dims explicitly** since the paper omits them: pick `D` (e.g., 256–512, ACT default 512), `N_img`=49 (224² ResNet18), `d_act`=7 (LIBERO), `L_enc=4`, choose `L_dec` (ACT default ~7). Document these as our re-impl config — they ARE the IB `D` and SIB `H`/`d_act`.
4. **Get LIBERO data** (OpenVLA-style processed LIBERO Spatial/Object/Goal/Long) and wire the OpenVLA/LeRobot 50-trial eval harness.
5. **Train NanoVLA-S baseline** (BERT-base + ResNet18, both frozen; ~52M trainable). Target the paper's per-suite SR (78.7 avg). Budget for the stated ~1M-step convergence; on A100 80GB use a large batch (it OOM'd OpenVLA at bs32 on H20, but our trainable stack is tiny → push batch high).
6. **Validate baseline against Table 1** per suite (esp. Object 93.6 / Goal 89.6 which are the strong cells; accept Long ≈49.8 and LIBERO-90 ≈55.1 as expected-weak).
7. **Then port the plug-in:** (a) IB-Adapter on `X_img` after `W_img` (dim `D`); (b) SIB on action chunk `A ∈ ℝ^{H×d_act}` (`H=100`). Keep CVAE-KL on; run the joint β-vs-SIB-weight ablation from §6.2. Because the head is deterministic regression (not diffusion), SIB attaches in one shot exactly as on SmolVLA.

---

## Source list
- arXiv abstract & metadata: https://arxiv.org/abs/2510.25122 — title, authors, date, category.
- arXiv full text (HTML, used for §2/§4/§5/§6): https://arxiv.org/html/2510.25122v1 ; PDF https://arxiv.org/pdf/2510.25122v1
- HF paper page (no repo/ckpt links present): https://huggingface.co/papers/2510.25122
- HF model search by arXiv id → **0 results**: https://huggingface.co/models?other=arxiv:2510.25122
- OpenReview (**withdrawn**): https://openreview.net/forum?id=yeHBrNVZoV
- alphaXiv (no code links; 403 to bot): https://www.alphaxiv.org/abs/2510.25122v1
- Moonlight review (confirms no code/ckpt links): https://www.themoonlight.io/en/review/nanovla-routing-decoupled-vision-language-understanding-for-nano-sized-generalist-robotic-policies
- LeRobot (framework substrate, ACT/SmolVLA reference impl): https://github.com/huggingface/lerobot

*Items I could NOT confirm from primary sources (flagged in-text): exact model dim `D`, `N_img`, `d_act`, decoder depth `L_dec`, β value, optimizer/LR/batch, official train/eval commands, and any official module/class names or file paths (no code released).*
