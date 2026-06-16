"""NanoVLA-S re-implementation (language-conditioned ACT) + IB/SIB plug-in port.

NanoVLA has no public code/checkpoint (arXiv 2510.25122, withdrawn). We
re-implement the smallest variant (NanoVLA-S) as LeRobot's ACT (ResNet18 +
transformer enc/dec + CVAE + action-chunk regression) conditioned on a frozen
BERT-base instruction token, injected via ACT's env_state 1D-token slot — so the
tested ACT module is reused unmodified. See multivla/NANOVLA_IMPL_PLAN.md.
"""
from .modeling_nanovla import NanoVLAS, make_nanovla_config

__all__ = ["NanoVLAS", "make_nanovla_config"]
