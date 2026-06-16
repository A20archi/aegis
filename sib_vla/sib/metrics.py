"""Evaluation metrics and the statistics the spec mandates.

Trajectory metrics (``rms_jerk``, ``hf_energy_fraction``) operate on the
**executed** action sequence of one episode, shape ``(T, d)`` with ``T`` the
episode length (distinct from the chunk horizon ``H``).  Inference statistics
use Wilson 95% intervals for single success rates and a two-proportion z-test
for held-out / corruption comparisons.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.fft import dct


# --------------------------------------------------------------------------- #
# Trajectory smoothness / spectral content
# --------------------------------------------------------------------------- #
def _as_2d(actions) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected (T, d) action sequence, got {a.shape}")
    return a


def rms_jerk(actions) -> float:
    """RMS of the third finite difference of an executed sequence ``(T, d)``.

    Returns ``nan`` if the episode is too short (``T < 4``).
    """
    a = _as_2d(actions)
    if a.shape[0] < 4:
        return float("nan")
    j = np.diff(a, n=3, axis=0)
    return float(np.sqrt(np.mean(j ** 2)))


def hf_energy_fraction(actions, cutoff_frac: float = 0.5) -> float:
    """Fraction of executed-action DCT energy at/above a band cutoff.

    The cutoff is a fraction of the number of bands ``T`` (``cutoff_frac=0.5``
    -> upper half of the spectrum).  Energy is pooled across action dims via the
    orthonormal DCT-II (Parseval: spectral energy == time-domain energy).
    Always report ``cutoff_frac`` alongside this number.
    """
    a = _as_2d(actions)
    T = a.shape[0]
    if T < 2:
        return float("nan")
    X = dct(a, type=2, norm="ortho", axis=0)            # (T, d)
    energy = X ** 2
    k = max(1, int(round(cutoff_frac * T)))
    total = energy.sum()
    if total <= 0:
        return 0.0
    return float(energy[k:].sum() / total)


# --------------------------------------------------------------------------- #
# Bit allocation
# --------------------------------------------------------------------------- #
def bits_per_band_summary(bits_hd: np.ndarray) -> dict:
    """Summarise an ``(H, d)`` per-band bit allocation (already in bits)."""
    b = np.asarray(bits_hd, dtype=np.float64)
    return {
        "bits_per_band_dim": b.tolist(),            # (H, d)
        "bits_per_band": b.sum(axis=1).tolist(),    # (H,) summed over dims
        "total_bits": float(b.sum()),
    }


# --------------------------------------------------------------------------- #
# Inference statistics
# --------------------------------------------------------------------------- #
def success_rate(successes: Sequence[bool]) -> float:
    s = np.asarray(successes, dtype=np.float64)
    return float(s.mean()) if s.size else float("nan")


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(p_hat, lo, hi)``.  ``(nan, nan, nan)`` if ``n_total == 0``.
    The Wilson interval is well behaved near 0/1 and for small ``n``, unlike the
    normal (Wald) interval.
    """
    if n_total <= 0:
        return (float("nan"), float("nan"), float("nan"))
    n = float(n_total)
    p = n_success / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def two_proportion_ztest(
    s1: int, n1: int, s2: int, n2: int
) -> tuple[float, float]:
    """Pooled two-proportion z-test.  Returns ``(z, two_sided_p)``.

    Tests H0: ``p1 == p2``.  ``(nan, nan)`` if a sample is empty or the pooled
    proportion is degenerate (0 or 1).
    """
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"))
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return (float("nan"), float("nan"))
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z = (p1 - p2) / se
    p_two = math.erfc(abs(z) / math.sqrt(2.0))          # 2 * (1 - Phi(|z|))
    return (z, p_two)
