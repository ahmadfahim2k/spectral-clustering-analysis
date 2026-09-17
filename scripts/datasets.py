"""
Synthetic dataset generation.

Three classic non-convex / convex shapes used to stress-test spectral methods
against the k-means baseline (scikit-learn generators, fixed seed):

    rings   two concentric circles  — non-convex, k-means fails, SC should win
    moons   two interleaving moons  — non-convex, k-means fails, SC should win
    blobs   three isotropic gaussians — convex, k-means and SC both do well

Every generator is seeded with SEED (42) so the datasets are identical on every
run and across the notebook and the mirrored script.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

from . import SEED


def _scale(X: np.ndarray) -> np.ndarray:
    """Standardise features so the RBF bandwidth behaves consistently."""
    return StandardScaler().fit_transform(X)


def make_synthetic(n_samples: int = 300, seed: int = SEED) -> dict:
    """
    Build the three synthetic datasets.

    Returns a dict keyed by dataset name; each value has:
        X       (n, 2) standardised feature matrix
        y       (n,)   ground-truth cluster labels
        k_true  int    true number of clusters
    """
    rng = seed

    Xr, yr = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=rng)
    Xm, ym = make_moons(n_samples=n_samples, noise=0.07, random_state=rng)
    Xb, yb = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.60, random_state=rng)

    return {
        "rings": {"X": _scale(Xr), "y": yr, "k_true": 2,
                  "label": "Concentric rings"},
        "moons": {"X": _scale(Xm), "y": ym, "k_true": 2,
                  "label": "Two moons"},
        "blobs": {"X": _scale(Xb), "y": yb, "k_true": 3,
                  "label": "Three blobs"},
    }
