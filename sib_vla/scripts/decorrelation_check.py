"""Stage 1 spectral pre-check: gates A (story) and B (assumption).

Reads the cached chunks, forms the band covariance of the predicted DCT
coefficients (pooled over samples and action dims), and reports:

* **Gate A (story).**  If essentially all energy sits in the lowest ~2-3 bands,
  the per-band allocation story is weak (it collapses to "most bands are free to
  drop").  Flag it; proceed but bias the suite toward contact-rich tasks.
* **Gate B (assumption).**  If the off-diagonal energy ratio of the band
  *correlation* matrix is high, the bands are correlated and the per-band
  independent-channel model is mis-specified -> set ``rotation: pca`` and save
  the decorrelating rotation (eigenvectors of the band covariance).

    python scripts/decorrelation_check.py --config configs/sib.yaml

Outputs: ``results/stage1.json`` (+ ``results/pca_rotation.pt`` if Gate B fires).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from sib.data import ChunkCache
from sib.transforms import SpectralTransform
from sib.utils import load_config, save_json


def band_covariance(X: torch.Tensor) -> torch.Tensor:
    """``(H, H)`` covariance of bands, pooling samples and action dims.

    Each ``(sample, dim)`` is one observation of the length-``H`` band vector.
    """
    N, H, d = X.shape
    obs = X.permute(0, 2, 1).reshape(N * d, H)                  # (N*d, H)
    obs = obs - obs.mean(dim=0, keepdim=True)
    return (obs.T @ obs) / (obs.shape[0] - 1)                   # (H, H)


def offdiag_energy_ratio(cov: torch.Tensor) -> float:
    """Fraction of correlation-matrix energy off the diagonal (in [0, 1])."""
    d = torch.sqrt(torch.diag(cov).clamp(min=1e-12))
    corr = cov / (d[:, None] * d[None, :])
    off = corr - torch.diag(torch.diag(corr))
    return float((off ** 2).sum() / (corr ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])

    cache = ChunkCache.load(out / "chunk_cache.pt")
    H = cache.A.shape[1]
    X = SpectralTransform(H).analyze(cache.A)                   # (N, H, d)

    cov = band_covariance(X)
    band_energy = torch.diag(cov)
    frac = (band_energy / band_energy.sum())
    low3 = float(frac[:3].sum())
    offdiag = offdiag_energy_ratio(cov)

    thr = cfg.get("stage1", {})
    gate_a_weak = low3 >= thr.get("low_band_energy_threshold", 0.95)
    gate_b_correlated = offdiag >= thr.get("offdiag_threshold", 0.20)

    report = {
        "band_energy_fraction": frac.tolist(),
        "cumulative_low3_band_energy": low3,
        "offdiag_energy_ratio": offdiag,
        "gate_a_allocation_story_weak": bool(gate_a_weak),
        "gate_b_bands_correlated": bool(gate_b_correlated),
        "recommended_rotation": "pca" if gate_b_correlated else "none",
    }

    if gate_b_correlated:
        # PCA rotation = eigenvectors (rows) so that R @ X decorrelates the bands.
        evals, evecs = torch.linalg.eigh(cov)
        R = evecs.flip(1).T.contiguous()                       # (H, H), orthogonal
        torch.save(R, out / "pca_rotation.pt")
        report["pca_rotation_saved"] = str(out / "pca_rotation.pt")

    save_json(report, out / "stage1.json")
    print(f"[stage1] low-3 energy={low3:.3f} (weak story={gate_a_weak}); "
          f"offdiag={offdiag:.3f} (correlated={gate_b_correlated}); "
          f"rotation -> {report['recommended_rotation']}")


if __name__ == "__main__":
    main()
