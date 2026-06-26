"""
baselines.py — cheap no-train robustness baselines to compete against AEGIS on LIBERO-Plus.

Both wrap the FROZEN base ACT (no RIB), provide reset()/select_action() like the policy, so the
eval drivers can swap them in transparently. Δ-over-base is the comparison metric.

  TTAWrapper  — Test-Time Augmentation: K augmented views of the obs per replan, predict a chunk
                for each, average the chunks. Standard no-train robustness control.
  BNAdaptWrapper — Test-time BatchNorm adaptation on the ResNet-18 backbone: blends frozen training
                BN stats with the current input's batch stats (Schneider et al. 2020 'prior' rule),
                so the feature normalization tracks the corrupted test distribution.
"""
from __future__ import annotations
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- TTA
def _augment(img: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """img (1,C,H,W) in normalized space -> (k,C,H,W) augmented views (view 0 = identity)."""
    views = [img]
    C, H, W = img.shape[1:]
    for _ in range(k - 1):
        x = img
        # small random resized crop (approx viewpoint/scale jitter)
        s = 0.85 + 0.25 * torch.rand(1, generator=gen, device=gen.device).item()
        nh, nw = max(8, int(H * s)), max(8, int(W * s))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        # mild photometric jitter
        x = x * (0.9 + 0.2 * torch.rand(1, generator=gen, device=gen.device).item())
        views.append(x)
    return torch.cat(views, 0)


class TTAWrapper(nn.Module):
    def __init__(self, base, n_action_steps=5, k=4, image_keys=("observation.images.image",)):
        super().__init__()
        self.base = base
        self.n_action_steps = n_action_steps
        self.k = k
        self.image_keys = image_keys
        self._q = deque()
        self._gen = torch.Generator(device=next(base.parameters()).device).manual_seed(0)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def reset(self):
        self._q.clear()
        if hasattr(self.base, "reset"):
            self.base.reset()

    @torch.no_grad()
    def _chunk(self, batch):
        # build K augmented copies of the (agentview) image; keep others tiled
        b0 = {kk: v for kk, v in batch.items()}
        aug_img = _augment(batch[self.image_keys[0]], self.k, self._gen)   # (k,C,H,W)
        chunks = []
        for i in range(self.k):
            bi = dict(b0)
            bi[self.image_keys[0]] = aug_img[i:i+1]
            chunks.append(self.base.predict_action_chunk(bi).float())
        return torch.stack(chunks, 0).mean(0)                              # (1,T,A)

    @torch.no_grad()
    def select_action(self, batch):
        if not self._q:
            chunk = self._chunk(batch)
            for a in chunk[:, : self.n_action_steps].transpose(0, 1):
                self._q.append(a)
        return self._q.popleft()


# --------------------------------------------------------------------------- BN adapt
class BNAdaptWrapper(nn.Module):
    """Test-time BN adaptation on model.backbone: y = norm(x; α·train_stats + (1-α)·batch_stats)."""
    def __init__(self, base, n_action_steps=5, alpha=0.5):
        super().__init__()
        self.base = base
        self.n_action_steps = n_action_steps
        self._q = deque()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        # install prior-blended BN on the ResNet backbone
        bb = self.base.model.backbone
        for m in bb.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_pre_hook(self._make_hook(m, alpha))

    @staticmethod
    def _make_hook(bn, alpha):
        rm, rv = bn.running_mean.clone(), bn.running_var.clone()
        def hook(mod, inp):
            x = inp[0]
            bm = x.mean(dim=(0, 2, 3)); bv = x.var(dim=(0, 2, 3), unbiased=False)
            mod.running_mean = alpha * rm + (1 - alpha) * bm
            mod.running_var = alpha * rv + (1 - alpha) * bv
            return None
        return hook

    def reset(self):
        self._q.clear()
        if hasattr(self.base, "reset"):
            self.base.reset()

    @torch.no_grad()
    def select_action(self, batch):
        if not self._q:
            chunk = self.base.predict_action_chunk(batch).float()
            for a in chunk[:, : self.n_action_steps].transpose(0, 1):
                self._q.append(a)
        return self._q.popleft()
