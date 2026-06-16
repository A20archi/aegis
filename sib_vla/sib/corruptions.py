"""Eval-time observation corruptions (Section 7).

Applied to the observation image *only at evaluation*, before it enters the
frozen backbone -- training never sees them, so this is a corruption-agnostic
robustness test.  Three families x two severities.  All ops are plain
torch/numpy on a float image in ``[0, 1]``; none are differentiable through (we
never need gradients here).

Image convention: channel-first ``(C, H, W)`` or batched ``(B, C, H, W)``,
float in ``[0, 1]``.  ``apply`` accepts uint8 ``[0, 255]`` and round-trips it.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

# Two severities per family. Tuned to be visible but not destructive.
SEVERITY: dict[str, list[dict]] = {
    "gaussian_noise": [{"std": 0.05}, {"std": 0.12}, {"std": 0.20}, {"std": 0.30},
                       {"std": 0.50}, {"std": 0.70}, {"std": 0.75}, {"std": 1.00}],
    "gaussian_blur": [{"sigma": 1.0}, {"sigma": 2.0}],
    "brightness_contrast": [{"brightness": 0.15, "contrast": 0.85},
                            {"brightness": 0.30, "contrast": 0.70}],
    # LIBERO-V sensor-noise family (image-space, paper Sec 10.1.4). 3 severities
    # each (mild / medium / strong). Deterministic given params -> reproducible.
    "motion_blur":  [{"length": 7, "angle": 30.0}, {"length": 13, "angle": 30.0},
                     {"length": 21, "angle": 30.0}],
    "zoom_blur":    [{"max_zoom": 1.08, "steps": 5}, {"max_zoom": 1.18, "steps": 7},
                     {"max_zoom": 1.30, "steps": 9}],
    "fog":          [{"density": 0.25}, {"density": 0.45}, {"density": 0.65}],
    "glass_blur":   [{"sigma": 0.7, "warp": 1.5}, {"sigma": 1.0, "warp": 2.5},
                     {"sigma": 1.5, "warp": 4.0}],
}


def _safe_randn(shape, *, device, dtype, generator=None) -> Tensor:
    """torch.randn that tolerates a cross-device generator. A generator bound to a
    different device than the target tensor raises; in that case fall back to the
    default RNG (still seeded at run level). Single choke-point for every randn in
    this module so the device contract can't silently regress (see test_eval_paths).
    """
    g = generator if (generator is not None and generator.device == torch.device(device)) else None
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


def _ensure_bchw(img: Tensor) -> tuple[Tensor, bool]:
    if img.dim() == 3:
        return img.unsqueeze(0), True
    if img.dim() == 4:
        return img, False
    raise ValueError(f"expected (C,H,W) or (B,C,H,W), got {tuple(img.shape)}")


def gaussian_noise(img: Tensor, std: float, generator: torch.Generator | None = None) -> Tensor:
    noise = _safe_randn(img.shape, device=img.device, dtype=img.dtype, generator=generator)
    return (img + std * noise).clamp_(0.0, 1.0)


def _gaussian_kernel1d(sigma: float, device, dtype) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur with reflect padding, per channel."""
    x, squeezed = _ensure_bchw(img)
    B, C, H, W = x.shape
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    kh = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    kw = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode="reflect")
    x = torch.nn.functional.conv2d(x, kh, groups=C)
    x = torch.nn.functional.pad(x, (r, r, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, kw, groups=C)
    return x if not squeezed else x.squeeze(0)


def brightness_contrast(img: Tensor, brightness: float, contrast: float) -> Tensor:
    """``out = (img - 0.5) * contrast + 0.5 + brightness``, clamped to ``[0, 1]``."""
    return ((img - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)


def motion_blur(img: Tensor, length: int, angle: float) -> Tensor:
    """Directional (linear) motion blur: convolve with a line kernel of `length`
    pixels oriented at `angle` degrees. Emulates camera/robot motion smear."""
    x, squeezed = _ensure_bchw(img)
    B, C, H, W = x.shape
    L = max(1, int(length))
    k = torch.zeros(L, L, device=x.device, dtype=x.dtype)
    th = math.radians(angle)
    cx = (L - 1) / 2.0
    for t in range(L):
        d = t - cx
        r = int(round(cx + d * math.sin(th)))
        c = int(round(cx + d * math.cos(th)))
        if 0 <= r < L and 0 <= c < L:
            k[r, c] = 1.0
    k = k / k.sum().clamp_min(1.0)
    ker = k.view(1, 1, L, L).expand(C, 1, L, L)
    pad = L // 2
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = torch.nn.functional.conv2d(x, ker, groups=C)
    return x if not squeezed else x.squeeze(0)


def zoom_blur(img: Tensor, max_zoom: float, steps: int) -> Tensor:
    """Average a stack of progressively zoomed-in crops (resized back to HxW) to
    emulate focal-length/zoom motion."""
    x, squeezed = _ensure_bchw(img)
    B, C, H, W = x.shape
    acc = x.clone()
    n = 1
    for z in torch.linspace(1.0, float(max_zoom), int(steps), device=x.device, dtype=x.dtype)[1:]:
        zh, zw = int(round(H * float(z))), int(round(W * float(z)))
        up = torch.nn.functional.interpolate(x, size=(zh, zw), mode="bilinear",
                                             align_corners=False)
        top = (zh - H) // 2
        left = (zw - W) // 2
        acc = acc + up[:, :, top:top + H, left:left + W]
        n += 1
    out = (acc / n).clamp(0.0, 1.0)
    return out if not squeezed else out.squeeze(0)


def fog(img: Tensor, density: float) -> Tensor:
    """Atmospheric scattering: blend toward a bright veil with a smooth low-freq
    fog map; reduces contrast. Deterministic (fog map from image coords)."""
    x, squeezed = _ensure_bchw(img)
    B, C, H, W = x.shape
    yy = torch.linspace(0, math.pi, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
    xx = torch.linspace(0, math.pi, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
    # smooth plasma-ish veil in [0,1]
    veil = 0.5 + 0.5 * (torch.sin(2.0 * yy) * torch.cos(3.0 * xx)
                        + torch.sin(3.0 * yy + 1.0) * torch.cos(2.0 * xx + 0.5)) / 2.0
    t = float(density) * veil          # per-pixel fog weight
    out = x * (1.0 - t) + 1.0 * t      # blend toward white
    # mild global contrast loss with fog
    out = (out - 0.5) * (1.0 - 0.3 * float(density)) + 0.5
    out = out.clamp(0.0, 1.0)
    return out if not squeezed else out.squeeze(0)


def glass_blur(img: Tensor, sigma: float, warp: float) -> Tensor:
    """Frosted-glass: Gaussian blur + smooth local pixel displacement (grid warp).
    Deterministic warp field (sin/cos of coords) so the perturbation is reproducible."""
    x, squeezed = _ensure_bchw(img)
    B, C, H, W = x.shape
    x = gaussian_blur(x, sigma)
    ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, H, 1)
    xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, W)
    base_y, base_x = torch.broadcast_tensors(ys, xs)
    dx = torch.sin(12.0 * base_y) * torch.cos(15.0 * base_x)
    dy = torch.cos(13.0 * base_y) * torch.sin(11.0 * base_x)
    off = float(warp)
    grid_x = (base_x + (off / W) * dx).clamp(-1, 1)
    grid_y = (base_y + (off / H) * dy).clamp(-1, 1)
    grid = torch.stack([grid_x, grid_y], dim=-1).expand(B, H, W, 2)
    out = torch.nn.functional.grid_sample(x, grid, mode="bilinear",
                                          padding_mode="reflection", align_corners=True)
    return out if not squeezed else out.squeeze(0)


_FAMILIES: dict[str, Callable] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "brightness_contrast": brightness_contrast,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "fog": fog,
    "glass_blur": glass_blur,
}


def apply(img: Tensor, name: str, severity: int,
          generator: torch.Generator | None = None) -> Tensor:
    """Apply corruption ``name`` at ``severity`` (0 or 1) to an image tensor.

    Accepts uint8 ``[0,255]`` or float ``[0,1]``; returns the same dtype/range.
    """
    if name not in _FAMILIES:
        raise ValueError(f"unknown corruption {name!r}; choices {list(_FAMILIES)}")
    if not 0 <= severity < len(SEVERITY[name]):
        raise ValueError(f"severity {severity} out of range for {name}")
    is_uint8 = img.dtype == torch.uint8
    x = img.float() / 255.0 if is_uint8 else img
    params = SEVERITY[name][severity]
    if name == "gaussian_noise":
        x = _FAMILIES[name](x, generator=generator, **params)
    else:
        x = _FAMILIES[name](x, **params)
    if is_uint8:
        x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    return x


def eval_corruption_list() -> list[tuple[str, int]]:
    """The full (name, severity) grid: 3 families x 2 severities = 6 conditions."""
    return [(name, s) for name in _FAMILIES for s in range(len(SEVERITY[name]))]


# Action-space perturbation (a distinct robustness axis from image corruptions).
# Hypothesis: an information limit on the action chunk should make execution more
# robust to additive action noise. Applied at eval only, to the executed action.
ACTION_NOISE_STD = [0.05, 0.10, 0.20]


def perturb_actions(actions: Tensor, std: float,
                    generator: torch.Generator | None = None) -> Tensor:
    """Add zero-mean Gaussian noise of scale ``std`` to an action tensor.

    The action tensor is on CPU by the env-step stage, while the shared eval
    generator may live on CUDA; torch.randn rejects a cross-device generator.
    Use the generator only when its device matches the tensor, else fall back to
    the default RNG (still seeded at run level).
    """
    if std <= 0:
        return actions
    noise = _safe_randn(actions.shape, device=actions.device,
                        dtype=actions.dtype, generator=generator)
    return actions + std * noise
