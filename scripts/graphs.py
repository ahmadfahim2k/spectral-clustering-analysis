"""
Similarity-graph construction.

For the synthetic datasets we build a fully-connected Gaussian (RBF) affinity
graph whose bandwidth sigma is chosen by the *median heuristic*. A self-tuning
(local-scaling) affinity is also provided for the Zelnik-Manor & Perona method.

For the network datasets (airport / PPI / facebook) the affinity matrix comes
directly from the observed edge weights, so those pipelines skip this module and
feed their adjacency matrix straight into `scripts.spectral`.

References
----------
von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and
    Computing, 17(4), 395-416.
Zelnik-Manor, L., & Perona, P. (2004). Self-tuning spectral clustering. NIPS 17.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Full matrix of squared Euclidean distances between rows of X."""
    d = squareform(pdist(X, metric="euclidean"))
    return d ** 2


def median_heuristic(X: np.ndarray) -> float:
    """
    Bandwidth sigma for the RBF kernel via the median heuristic.

    sigma is set to the median of the *non-zero* pairwise Euclidean distances.
    This is a standard, data-driven, parameter-free choice that adapts the kernel
    scale to the spread of the data (von Luxburg, 2007).
    """
    dists = pdist(X, metric="euclidean")
    dists = dists[dists > 0]
    if dists.size == 0:
        raise ValueError("Cannot compute median heuristic: all points are identical.")
    return float(np.median(dists))


def rbf_affinity(X: np.ndarray, sigma: float | None = None) -> tuple[np.ndarray, float]:
    """
    Fully-connected Gaussian (RBF) affinity matrix.

        W_ij = exp( -||x_i - x_j||^2 / (2 * sigma^2) )

    Parameters
    ----------
    X : (n, d) array of points.
    sigma : kernel bandwidth. If None, the median heuristic is used.

    Returns
    -------
    W : (n, n) symmetric affinity matrix with a zero diagonal.
    sigma : the bandwidth actually used (useful for logging / reproducibility).
    """
    if sigma is None:
        sigma = median_heuristic(X)
    sq = pairwise_sq_dists(X)
    W = np.exp(-sq / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)  # no self-similarity in the graph
    return W, float(sigma)


def knn_rbf_affinity(
    X: np.ndarray, n_neighbors: int = 10, sigma: float | None = None
) -> tuple[np.ndarray, float]:
    """
    Symmetric k-nearest-neighbour RBF affinity graph.

    A sparser alternative to the fully-connected `rbf_affinity`. Each point is
    connected only to its `n_neighbors` nearest neighbours (edges symmetrised
    with the OR rule), with Gaussian weights on the surviving edges. kNN graphs
    usually give cleaner eigengaps and stronger spectral results on non-convex
    data than a dense global-bandwidth graph, so this is provided as a drop-in
    option for the notebooks. Not the specified default.
    """
    if sigma is None:
        sigma = median_heuristic(X)
    sq = pairwise_sq_dists(X)
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    W = np.exp(-sq / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)

    # keep only the k largest weights (nearest neighbours) per row
    mask = np.zeros_like(W, dtype=bool)
    nn_idx = np.argsort(-W, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    mask[rows, nn_idx.ravel()] = True
    mask = mask | mask.T  # symmetrise (OR rule)
    W = W * mask
    return W, float(sigma)


def self_tuning_affinity(X: np.ndarray, n_neighbors: int = 7) -> np.ndarray:
    """
    Local-scaling affinity of Zelnik-Manor & Perona (2004).

        A_ij = exp( -d(x_i, x_j)^2 / (sigma_i * sigma_j) )

    where sigma_i is the distance from x_i to its `n_neighbors`-th nearest
    neighbour. Using a *local* scale per point lets the graph adapt to regions
    of differing density, which a single global sigma cannot do.

    n_neighbors=7 is the value recommended in the original paper.
    """
    sq = pairwise_sq_dists(X)
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    # distance to the k-th nearest neighbour (row-wise), excluding self (col 0)
    sorted_d = np.sort(np.sqrt(sq), axis=1)
    sigma = sorted_d[:, k]                       # (n,)
    sigma = np.where(sigma <= 0, 1e-12, sigma)   # guard against duplicates
    scale = np.outer(sigma, sigma)               # sigma_i * sigma_j
    A = np.exp(-sq / scale)
    np.fill_diagonal(A, 0.0)
    return A
