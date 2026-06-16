"""NanoVLA-S = language-conditioned ACT.

Reuses LeRobot's ACT nn.Module unmodified. The frozen BERT-base instruction
embedding is fed through ACT's `env_state` 1D-token slot (FeatureType.ENV),
which is exactly NanoVLA's "one language token concatenated into the encoder
sequence". IB-Adapter attaches at `act.encoder_img_feat_input_proj` (the 1×1
conv `W_img`, the FRONT point); SIB attaches at the `act.action_head` output
(the action chunk `A ∈ [B,H,d_act]`, the BACK point).

Validated: ACT(config) with a 768-d env_state token yields (B, chunk_size,
action_dim) + CVAE (mu,logvar); params ~84M (+frozen BERT ~110M).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def make_nanovla_config(
    state_dim: int = 8,
    action_dim: int = 7,
    image_keys: tuple[str, ...] = ("observation.images.image",),
    image_shape: tuple[int, int, int] = (3, 224, 224),
    lang_dim: int = 768,            # BERT-base hidden size = the env_state/language token dim
    chunk_size: int = 100,          # NanoVLA H_train
    n_action_steps: int = 10,       # NanoVLA executes h=10 then replans (long-short chunking)
    dim_model: int = 512,
    n_encoder_layers: int = 4,      # paper
    n_decoder_layers: int = 7,      # paper (ACT default is 1 due to an upstream bug; NanoVLA uses ~7)
    dim_feedforward: int = 3200,
    n_heads: int = 8,
    latent_dim: int = 32,
    kl_weight: float = 10.0,
    vision_backbone: str = "resnet18",
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
) -> ACTConfig:
    """Build the ACT config for NanoVLA-S. The language token is carried by the
    env_state feature (shape = lang_dim)."""
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(lang_dim,)),  # language token
    }
    for k in image_keys:
        input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=image_shape)
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    # We do our own (de)normalization in the train/eval loop, so use IDENTITY here.
    norm = {ft: NormalizationMode.IDENTITY for ft in
            (FeatureType.STATE, FeatureType.ENV, FeatureType.VISUAL, FeatureType.ACTION)}
    return ACTConfig(
        chunk_size=chunk_size, n_action_steps=n_action_steps, dim_model=dim_model,
        n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
        dim_feedforward=dim_feedforward, n_heads=n_heads, use_vae=True,
        latent_dim=latent_dim, kl_weight=kl_weight, vision_backbone=vision_backbone,
        pretrained_backbone_weights=pretrained_backbone_weights,
        temporal_ensemble_coeff=None,
        input_features=input_features, output_features=output_features,
        normalization_mapping=norm,
    )


class NanoVLAS(nn.Module):
    """Language-conditioned ACT. Holds the ACT module + a frozen BERT instruction
    encoder. forward(batch) accepts either a precomputed env_state language token
    or raw instruction text in batch['task'] (list[str])."""

    def __init__(self, config: ACTConfig, lang_model: str = "google-bert/bert-base-uncased"):
        super().__init__()
        self.config = config
        self.act = ACT(config)
        self.lang_dim = config.env_state_feature.shape[0]
        # frozen language encoder
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        self.bert = AutoModel.from_pretrained(lang_model)
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad_(False)
        # freeze the vision backbone (NanoVLA keeps vision frozen)
        if hasattr(self.act, "backbone"):
            for p in self.act.backbone.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, texts: list[str], device) -> Tensor:
        toks = self.tokenizer(list(texts), padding=True, truncation=True,
                              max_length=32, return_tensors="pt").to(device)
        out = self.bert(**toks)
        # pooled CLS embedding -> (B, lang_dim)
        emb = getattr(out, "pooler_output", None)
        if emb is None:
            emb = out.last_hidden_state[:, 0]
        return emb.to(dtype=next(self.act.parameters()).dtype)

    def _ensure_lang_token(self, batch: dict) -> dict:
        if OBS_ENV_STATE not in batch and "task" in batch:
            dev = batch[OBS_IMAGES][0].device if OBS_IMAGES in batch else batch[OBS_STATE].device
            batch = dict(batch)
            batch[OBS_ENV_STATE] = self.encode_text(batch["task"], dev)
        return batch

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        """Training forward: returns (loss, parts)."""
        batch = self._ensure_lang_token(batch)
        actions_hat, (mu, logvar) = self.act(batch)
        target = batch[ACTION]
        l1 = F.l1_loss(target, actions_hat, reduction="none")
        if "action_is_pad" in batch:
            l1 = l1 * (~batch["action_is_pad"]).unsqueeze(-1)
        l1 = l1.mean()
        parts = {"l1": float(l1.detach())}
        loss = l1
        if self.config.use_vae and mu is not None:
            kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
            loss = l1 + self.config.kl_weight * kld
            parts["kld"] = float(kld.detach())
        parts["loss"] = float(loss.detach())
        return loss, parts

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> Tensor:
        """Eval: deterministic (z=0) action chunk (B, chunk_size, action_dim)."""
        batch = self._ensure_lang_token(batch)
        self.act.eval()
        actions, _ = self.act(batch)     # use_vae & not training -> latent=0 inside ACT
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict) -> Tensor:
        """First action of the chunk (caller handles receding-horizon queueing)."""
        return self.predict_action_chunk(batch)[:, 0]
