"""
Clustering algorithms benchmarked in the dissertation.

    kmeans_baseline           standard k-means on the raw features
    unnormalised_spectral     L = D - W                 (von Luxburg, 2007)
    spectral_shi_malik        L_rw = D^-1 L             (Shi & Malik, 2000)
    spectral_ng_jordan_weiss  L_sym = D^-1/2 L D^-1/2   (Ng, Jordan & Weiss, 2001)
    self_tuning_spectral      local-scaling + NJW       (Zelnik-Manor & Perona, 2004)

All spectral methods follow the same three-stage recipe: (1) build a Laplacian
from the affinity matrix W, (2) embed points using its k smallest eigenvectors,
(3) run k-means in the embedding. They differ only in how the Laplacian is
normalised and how the embedding is post-processed.

Reproducibility: every k-means call uses n_init=10 and random_state=SEED (42).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from . import SEED
from .graphs import self_tuning_affinity


def _kmeans(embedding: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """k-means with the fixed, reproducible settings used throughout the project."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(embedding)


def _degree_matrix(W: np.ndarray) -> np.ndarray:
    """Diagonal degree matrix D from affinity matrix W."""
    return np.diag(W.sum(axis=1))


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def kmeans_baseline(X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Plain k-means on the raw feature matrix (the non-spectral baseline)."""
    return _kmeans(X, k, seed)


# --------------------------------------------------------------------------- #
# Unnormalised spectral clustering  —  von Luxburg (2007)
# --------------------------------------------------------------------------- #
def unnormalised_spectral(W: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """
    Cluster with the unnormalised Laplacian L = D - W.

    Uses the k eigenvectors of L with the smallest eigenvalues as the embedding,
    then applies k-means.
    """
    D = _degree_matrix(W)
    L = D - W
    # symmetric -> use eigh; take the k smallest eigenvectors
    _, vecs = eigh(L, subset_by_index=[0, k - 1])
    return _kmeans(vecs, k, seed)


# --------------------------------------------------------------------------- #
# Normalised spectral clustering  —  Shi & Malik (2000)
# --------------------------------------------------------------------------- #
def spectral_shi_malik(W: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """
    Random-walk normalised cut (Shi & Malik, 2000).

    Solves the generalised eigenproblem  L u = lambda D u, which is equivalent to
    using the random-walk Laplacian L_rw = D^-1 L. The k eigenvectors with the
    smallest eigenvalues form the embedding, followed by k-means.
    """
    D = _degree_matrix(W)
    L = D - W
    # generalised symmetric-definite eigenproblem L u = lambda D u
    _, vecs = eigh(L, D, subset_by_index=[0, k - 1])
    return _kmeans(vecs, k, seed)


# --------------------------------------------------------------------------- #
# Normalised spectral clustering  —  Ng, Jordan & Weiss (2001)
# --------------------------------------------------------------------------- #
def _njw_embedding(W: np.ndarray, k: int) -> np.ndarray:
    """
    Symmetric-normalised embedding of Ng, Jordan & Weiss (2001).

    Build L_sym = D^-1/2 L D^-1/2, take its k smallest eigenvectors, then
    row-normalise the resulting matrix to unit length before clustering.
    """
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    # L_sym = I - D^-1/2 W D^-1/2, built by elementwise scaling rather than dense
    # diagonal matmuls (D^-1/2 W D^-1/2)_ij = d_i^-1/2 W_ij d_j^-1/2. This avoids
    # allocating full n x n diagonal matrices and two matmuls — much lighter and
    # faster on large graphs, mathematically identical.
    L_sym = -W * np.outer(d_inv_sqrt, d_inv_sqrt)
    L_sym[np.diag_indices_from(L_sym)] += 1.0
    L_sym = (L_sym + L_sym.T) / 2.0  # enforce numerical symmetry
    _, vecs = eigh(L_sym, subset_by_index=[0, k - 1])
    # row-normalise to unit norm (the key NJW step)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return vecs / norms


def spectral_ng_jordan_weiss(W: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Symmetric-normalised spectral clustering (Ng, Jordan & Weiss, 2001)."""
    U = _njw_embedding(W, k)
    return _kmeans(U, k, seed)


# --------------------------------------------------------------------------- #
# Self-tuning spectral clustering  —  Zelnik-Manor & Perona (2004)
# --------------------------------------------------------------------------- #
def self_tuning_spectral(
    X: np.ndarray, k: int, n_neighbors: int = 7, seed: int = SEED
) -> np.ndarray:
    """
    Self-tuning spectral clustering (Zelnik-Manor & Perona, 2004).

    Replaces the single global RBF bandwidth with a per-point *local scale*
    (distance to the n_neighbors-th neighbour), then clusters the resulting
    affinity with the NJW symmetric-normalised recipe. Operates on the raw
    points X because the affinity is built internally.
    """
    A = self_tuning_affinity(X, n_neighbors=n_neighbors)
    U = _njw_embedding(A, k)
    return _kmeans(U, k, seed)


# Registry so notebooks can iterate over algorithms uniformly.
# 'input' flags whether the algorithm consumes the affinity matrix W or the
# raw feature matrix X.
ALGORITHMS = {
    "kmeans":            {"fn": kmeans_baseline,          "input": "X", "label": "K-Means (baseline)"},
    "unnormalised":      {"fn": unnormalised_spectral,    "input": "W", "label": "Unnormalised SC"},
    "shi_malik":         {"fn": spectral_shi_malik,       "input": "W", "label": "Normalised SC — Shi & Malik (2000)"},
    "ng_jordan_weiss":   {"fn": spectral_ng_jordan_weiss, "input": "W", "label": "Normalised SC — Ng, Jordan & Weiss (2001)"},
    "self_tuning":       {"fn": self_tuning_spectral,     "input": "X", "label": "Self-Tuning SC — Zelnik-Manor & Perona (2004)"},
}
