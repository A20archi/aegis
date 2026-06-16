"""Frozen SmolVLA policy + trainable spectral bottleneck.

The bottleneck sits *after* the flow-matching sampler.  The sampler runs under
``no_grad`` (verified: ``SmolVLAPolicy.predict_action_chunk`` is decorated
``@torch.no_grad()`` and returns ``(B, H, d)`` in the policy's normalised action
space).  Only :class:`sib.bottleneck.SpectralActionModule` is trained.

Verified against LeRobot 0.4.3:
* ``base.predict_action_chunk(batch) -> (B, n_action_steps, action_dim)``.
* ``base.config.n_action_steps == chunk_size`` (50 for SmolVLA); ``action_dim``
  from ``config.action_feature.shape[0]`` (7 for LIBERO).
* ``select_action`` fills a queue from ``actions.transpose(0, 1)`` and pops the
  front; we reproduce that convention so the bottleneck-corrected chunk is what
  gets executed.
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .bottleneck import SpectralActionModule


class SIBPolicy(nn.Module):
    """Compose a frozen base policy with the spectral bottleneck module.

    Parameters
    ----------
    base : object
        A policy exposing ``predict_action_chunk(batch) -> (B, H, d)``.  All of
        its parameters are frozen here.
    module : SpectralActionModule
        The trainable bottleneck.
    n_action_steps : int
        How many steps of each chunk are executed before re-querying (the
        rollout queue length); defaults to the chunk horizon.
    context_fn : callable, optional
        ``(batch, A) -> (B, context_dim)`` for the context-conditioned variant.
    """

    def __init__(
        self,
        base,
        module: SpectralActionModule,
        n_action_steps: Optional[int] = None,
        context_fn: Optional[Callable[[dict, Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.module = module
        self.context_fn = context_fn
        self.n_action_steps = n_action_steps or module.H
        self._action_queue: deque = deque()
        self.freeze_base()
        self.assert_frozen()

    # ----------------------------------------------------------------- freezing
    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def assert_frozen(self) -> None:
        n_trainable = sum(p.requires_grad for p in self.base.parameters())
        assert n_trainable == 0, f"base has {n_trainable} trainable params; must be frozen"

    def train(self, mode: bool = True):
        """Keep the base in eval no matter what; only the module toggles."""
        self.training = mode
        self.module.train(mode)
        self.base.eval()
        return self

    # ----------------------------------------------------------------- forward
    def correct_chunk(self, A: Tensor, batch: Optional[dict] = None) -> Tensor:
        """Run the bottleneck on an already-sampled chunk ``A`` (B, H, d)."""
        in_dtype = A.dtype
        A32 = A.float()
        context = None
        if self.context_fn is not None and self.module.module_type in ("sib", "raw_vib"):
            context = self.context_fn(batch, A32)
        out = self.module(A32, context=context)
        return out["A_hat"].to(in_dtype)

    def predict_chunk(self, batch: dict,
                      pre_noise_std: float = 0.0,
                      generator: "torch.Generator | None" = None) -> Tensor:
        """Sample a chunk from the frozen policy, then apply the bottleneck.

        pre_noise_std: inject additive Gaussian noise on the raw chunk BEFORE
        the bottleneck, so the Wiener gain can suppress it.  This is the
        correct insertion point for testing 'does the rate bottleneck filter
        action-space noise?' -- the hypothesis that motivated this module.
        Post-bottleneck noise (actuator jitter) is handled in eval.py instead.
        """
        from sib import corruptions as corr
        dev_type = next(self.base.parameters()).device.type
        amp_ctx = (torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
                   if dev_type == 'cuda' else contextlib.nullcontext())
        with torch.no_grad(), amp_ctx:
            A = self.base.predict_action_chunk(batch).float()  # BF16 compute → FP32 out
        if pre_noise_std > 0:
            A = corr.perturb_actions(A, pre_noise_std, generator=generator)
        # Module may carry gradients in train(); under eval/no_grad it does not.
        return self.correct_chunk(A, batch)

    # ----------------------------------------------------------- rollout helper
    def reset(self) -> None:
        self._action_queue.clear()
        if hasattr(self.base, "reset"):
            self.base.reset()

    @torch.no_grad()
    def select_action(self, batch: dict,
                      pre_noise_std: float = 0.0,
                      generator: "torch.Generator | None" = None) -> Tensor:
        """Return a single action, refilling from a corrected chunk when empty.

        Mirrors ``SmolVLAPolicy.select_action``: queue holds steps along dim 0,
        batch along dim 1 (hence the transpose), popped front-first.
        pre_noise_std is passed to predict_chunk (noise before bottleneck).
        """
        if not self._action_queue:
            chunk = self.predict_chunk(batch,
                                       pre_noise_std=pre_noise_std,
                                       generator=generator)   # (B, H, d)
            steps = chunk.transpose(0, 1)[: self.n_action_steps]   # (n, B, d)
            self._action_queue.extend(steps)
        return self._action_queue.popleft()

    # ------------------------------------------------------------------- params
    def trainable_parameters(self):
        """Only the bottleneck's parameters are trainable."""
        return (p for p in self.module.parameters() if p.requires_grad)


class EMABlendPolicy(SIBPolicy):
    """SIBPolicy + EMA blending at chunk-boundary re-queries.

    At each re-query the new chunk's first ``ema_k`` actions are blended
    with the last executed action to smooth the discontinuity:
        a_exec[t] = (1 - w_t) * a_new[t] + w_t * a_prev_last
    where w_t decays linearly from ``ema_alpha`` at t=0 down to 0 at t=ema_k.
    """

    def __init__(self, base, module, n_action_steps=None, context_fn=None,
                 ema_alpha: float = 0.5, ema_k: int = 5) -> None:
        super().__init__(base, module, n_action_steps=n_action_steps,
                         context_fn=context_fn)
        self.ema_alpha = ema_alpha
        self.ema_k = ema_k
        self._last_action: Optional[Tensor] = None

    def reset(self) -> None:
        super().reset()
        self._last_action = None

    @torch.no_grad()
    def select_action(self, batch: dict,
                      pre_noise_std: float = 0.0,
                      generator: "torch.Generator | None" = None) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_chunk(batch, pre_noise_std=pre_noise_std,
                                       generator=generator)   # (B, H, d)
            steps = list(chunk.transpose(0, 1)[: self.n_action_steps])  # list[(B,d)]
            if self._last_action is not None:
                k = min(self.ema_k, len(steps))
                for t in range(k):
                    # weight decays linearly alpha → 0 over k steps
                    w = self.ema_alpha * (1.0 - t / k)
                    steps[t] = (1.0 - w) * steps[t] + w * self._last_action
            self._action_queue.extend(steps)
        action = self._action_queue.popleft()
        self._last_action = action.clone()
        return action


class ForgeActionHeadPolicy(nn.Module):
    """Forge recipe applied to the action head, SIB module placed after.

    The 5 Forge levers applied to the action head at inference:
      L2 – n_action_steps (re-query cadence)
      L3 – percentile-style soft clamp on raw predicted chunks
      L5 – ACT temporal ensembling: position-aligned blend of overlapping
           raw chunks builds a 'virtual chunk' that SIB then filters

    The SIB module sits AFTER the forge-enhanced action head output, exactly
    as the recipe specifies. Levers 1 (EMA) and 4 (augmentation) are
    training-time and cannot be applied here.

    virtual_chunk[k] = weighted average of raw_chunk_s[k+s] over all s,
    where s = steps since that chunk was predicted, giving a full H-step
    position-aligned blend. SIB filters the virtual chunk; position 0 is
    the executed action.
    """

    def __init__(
        self,
        base,
        module: "SpectralActionModule",
        n_action_steps: int = 1,
        ensemble_coeff: float = 0.01,
        lever3_clamp: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.module = module
        self.n_action_steps = n_action_steps
        self.ensemble_coeff = float(ensemble_coeff)
        self.lever3_clamp = float(lever3_clamp)
        self._chunks: list = []   # list of (chunk [B,H,d], start_t)
        self._step_t: int = 0
        self._freeze_base()

    def _freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        self.training = mode
        self.module.train(mode)
        self.base.eval()
        return self

    def reset(self) -> None:
        self._chunks.clear()
        self._step_t = 0
        if hasattr(self.base, "reset"):
            self.base.reset()

    def _build_virtual_chunk(self, current_t: int) -> Tensor:
        """Position-aligned temporal ensemble: V[k] = Σ_s w_s * chunk_s[k+s].

        Each raw chunk predicted `s` steps ago contributes its offset (k+s)
        to position k of the virtual chunk (the prediction for step t+k).
        Weights: exp(-ensemble_coeff * s), so newer predictions contribute
        more (s=0 is the most recent, highest weight).
        """
        H = self.module.H
        ref, _ = self._chunks[-1]
        B, _, d = ref.shape
        device, dtype = ref.device, ref.dtype

        V = torch.zeros(B, H, d, device=device, dtype=dtype)
        W = torch.zeros(H, device=device, dtype=dtype)

        for chunk, start in self._chunks:
            s = current_t - start      # steps since this chunk was predicted
            if not (0 <= s < H):
                continue
            n = H - s                  # number of positions this chunk covers
            w = torch.exp(
                torch.tensor(-self.ensemble_coeff * float(s),
                             device=device, dtype=dtype))
            # chunk[:, s:, :] covers V[:, 0:n, :] (position-aligned)
            V[:, :n, :].add_(w * chunk[:, s:, :])
            W[:n].add_(w)

        W.clamp_(min=1e-8)
        return V / W[None, :, None]

    @torch.no_grad()
    def select_action(
        self,
        batch: dict,
        pre_noise_std: float = 0.0,
        generator: "torch.Generator | None" = None,
    ) -> Tensor:
        import contextlib
        from sib import corruptions as corr

        # ---- L2: re-query every n_action_steps ----
        if self._step_t % self.n_action_steps == 0:
            dev_type = next(self.base.parameters()).device.type
            amp_ctx = (torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
                       if dev_type == "cuda" else contextlib.nullcontext())
            with amp_ctx:
                raw = self.base.predict_action_chunk(batch).float()
            if pre_noise_std > 0:
                raw = corr.perturb_actions(raw, pre_noise_std, generator=generator)
            # ---- L3: percentile-style soft clamp in normalised action space ----
            if self.lever3_clamp < 1.0:
                raw = raw.clamp(-self.lever3_clamp, self.lever3_clamp)
            self._chunks.append((raw, self._step_t))

        # Prune chunks that no longer overlap the current step
        H = self.module.H
        self._chunks = [(c, s) for (c, s) in self._chunks
                        if (self._step_t - s) < H]

        # ---- L5: temporal ensemble → virtual full chunk ----
        virtual = self._build_virtual_chunk(self._step_t)   # (B, H, d)

        # ---- SIB module after the forge-enhanced action head ----
        out = self.module(virtual)
        action = out["A_hat"][:, 0, :]   # execute position 0 (current step)

        self._step_t += 1
        return action

    def trainable_parameters(self):
        return (p for p in self.module.parameters() if p.requires_grad)


def make_state_context_fn(state_key: str = "observation.state"):
    """A simple ``context_fn`` for the context-conditioned channel ``sigma_k^2(o)``.

    Returns the proprioceptive state vector as the context embedding -- a cheap,
    always-available default. ``context_dim`` of the channel must equal this
    vector's width. For the *stronger* variant the planner describes (allocation
    predicted from VLM features), replace this with a forward hook that mean-pools
    the SmolVLA expert's hidden states; the rest of the pipeline is unchanged.
    """
    def context_fn(batch: dict, A: Tensor) -> Tensor:
        s = batch[state_key]
        if s.dim() > 2:                       # (B, T, D) -> last observed step
            s = s[:, -1, :]
        return s.float()
    return context_fn


def load_smolvla(checkpoint: str, device: str = "cuda"):
    """Load a frozen SmolVLA policy and its pre/post processors.

    Returns ``(policy, preprocessor, postprocessor)``.  The preprocessor
    normalises observations and tokenises the task; the postprocessor
    unnormalises the action chunk back to robot space.  Verified signature
    (LeRobot 0.4.3): ``make_pre_post_processors(policy.config, checkpoint)``.
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, checkpoint)
    action_dim = policy.config.action_feature.shape[0]
    # H is the chunk horizon the policy actually predicts (chunk_size, e.g. 50),
    # NOT n_action_steps (the receding-horizon execution length, e.g. 1).
    horizon = policy.config.chunk_size
    return policy, preprocessor, postprocessor, horizon, action_dim
