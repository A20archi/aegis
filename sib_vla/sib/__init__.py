"""Spectral Information Bottleneck for a frozen flow-matching VLA action head."""

from .transforms import SpectralTransform, dct_ii_matrix, cayley_orthogonal
from .lambda_estimator import LambdaEstimator
from .bottleneck import (
    SpectralActionModule,
    GaussianChannel,
    LearnedGain,
    LowpassMask,
    build_module,
    rate_nats,
    bits_per_band,
    wiener_gain,
)
from . import losses, metrics, waterfill, corruptions, recording

__all__ = [
    "SpectralTransform",
    "dct_ii_matrix",
    "cayley_orthogonal",
    "LambdaEstimator",
    "SpectralActionModule",
    "GaussianChannel",
    "LearnedGain",
    "LowpassMask",
    "build_module",
    "rate_nats",
    "bits_per_band",
    "wiener_gain",
    "losses",
    "metrics",
    "waterfill",
    "corruptions",
    "recording",
]
