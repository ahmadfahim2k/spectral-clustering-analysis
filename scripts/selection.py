"""
k-selection heuristics.

    kmeans_elbow          inertia-vs-k curve + automatic elbow (k-means baseline)
    eigengap              spectrum of the graph Laplacian (diagnostic curve)
    suggest_k_candidates  top-N plausible k values from the eigenvalue gaps
    pick_best_k           disambiguate candidates with a downstream quality metric

Spectral k-selection is a *two-step* procedure (see `suggest_k_candidates` /
`pick_best_k`): first narrow to a few plausible k from the eigenvalue spectrum,
then choose among them with a downstream clustering-quality metric. This avoids
the well-known failure mode of the single-largest-gap heuristic, which is
systematically biased toward k=2 — every connected graph's Laplacian has an
eigenvalue at (or near) zero, so the first gap away from zero tends to dominate
regardless of the true number of clusters (von Luxburg, 2007, §8.3).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import SEED


def kmeans_elbow(
    X: np.ndarray, k_range: range | list[int] = range(1, 11), seed: int = SEED
) -> dict:
    """
    Elbow method for k-means.

    Computes within-cluster inertia (sum of squared distances to centroids) for
    each k in `k_range`, then locates the elbow as the point of maximum distance
    from the straight line joining the first and last (k, inertia) points — a
    standard geometric formalisation of the visual "elbow".

    Returns a dict with the k values, inertias, and the suggested k.
    """
    ks = list(k_range)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        inertias.append(float(km.inertia_))

    suggested = _elbow_point(ks, inertias)
    return {"k": ks, "inertia": inertias, "suggested_k": suggested}


def _elbow_point(ks: list[int], inertias: list[float]) -> int:
    """Largest perpendicular distance to the line through the curve's endpoints."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    if len(x) < 3:
        return int(x[0])
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    dists = []
    for xi, yi in zip(x, y):
        vec = np.array([xi, yi]) - p1
        proj = np.dot(vec, line) * line
        dists.append(np.linalg.norm(vec - proj))
    return int(x[int(np.argmax(dists))])


def eigengap(
    W: np.ndarray, laplacian: str = "sym", k_max: int = 10
) -> dict:
    """
    Eigengap heuristic for choosing the number of spectral clusters.

    The eigenvalues of the graph Laplacian are sorted in ascending order; the
    number of clusters is read off as the index just before the first large gap
    between consecutive eigenvalues (von Luxburg, 2007). We normalise the
    Laplacian the same way NJW does (`sym`) by default so the estimate matches
    the algorithm it feeds.

    Returns the leading eigenvalues, the gaps between them, and the suggested k.
    """
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)
    n = W.shape[0]

    if laplacian == "sym":
        # L_sym = I - D^-1/2 W D^-1/2 via elementwise scaling (no dense diagonal
        # matmuls) — light and fast on large graphs, identical result.
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        M = -W * np.outer(d_inv_sqrt, d_inv_sqrt)
        M[np.diag_indices_from(M)] += 1.0
        M = (M + M.T) / 2.0
        n_eig = min(k_max + 1, n)
        vals = eigh(M, eigvals_only=True, subset_by_index=[0, n_eig - 1])
    elif laplacian == "unnormalised":
        L = np.diag(d) - W
        n_eig = min(k_max + 1, n)
        vals = eigh(L, eigvals_only=True, subset_by_index=[0, n_eig - 1])
    else:
        raise ValueError("laplacian must be 'sym' or 'unnormalised'")

    vals = np.sort(vals)
    gaps = np.diff(vals)
    # NAIVE single-largest-gap estimate, kept only as a diagnostic to contrast
    # against the two-step selection. It is biased toward k=2 (see module
    # docstring), so DO NOT use it directly for clustering — prefer
    # suggest_k_candidates + pick_best_k.
    naive = int(np.argmax(gaps[:k_max]) + 1)
    naive = max(naive, 2)
    return {
        "eigenvalues": vals.tolist(),
        "gaps": gaps.tolist(),
        "naive_suggested_k": naive,
        # backward-compatible alias (still the naive value; do not rely on it)
        "suggested_k": naive,
    }


def suggest_k_candidates(
    eigenvalues: Sequence[float],
    top_n: int = 3,
    k_min: int = 2,
    k_max: int = 10,
) -> list[int]:
    """
    Step 1 of two-step k-selection: shortlist plausible cluster counts.

    Rather than committing to the single largest eigenvalue gap, we return the
    `top_n` values of k whose gap g_k = lambda_{k+1} - lambda_k is largest,
    restricted to a sensible range [k_min, k_max]. A gap at sorted-eigenvalue
    index i corresponds to k = i + 1 clusters (the k eigenvalues below the gap).

    Parameters
    ----------
    eigenvalues : ascending Laplacian eigenvalues (e.g. from `eigengap`).
    top_n       : how many candidate k values to return.
    k_min, k_max: inclusive bounds on candidate k (clamped to the data size by
                  the caller supplying enough eigenvalues).

    Returns
    -------
    A list of candidate k values, ordered by decreasing gap size (best first).
    """
    vals = np.sort(np.asarray(eigenvalues, dtype=float))
    if vals.size < 2:
        return [max(k_min, 1)]
    gaps = np.diff(vals)                 # gaps[i] corresponds to k = i + 1
    ks = np.arange(1, gaps.size + 1)     # candidate k for each gap
    in_range = (ks >= k_min) & (ks <= k_max)
    ks, gaps = ks[in_range], gaps[in_range]
    if ks.size == 0:
        return [k_min]
    order = np.argsort(-gaps)            # largest gap first
    candidates = ks[order][:top_n]
    return [int(k) for k in candidates]


def pick_best_k(
    X: np.ndarray,
    candidates: Sequence[int],
    cluster_fn: Callable[[int], np.ndarray],
    ground_truth: np.ndarray | None = None,
) -> dict:
    """
    Step 2 of two-step k-selection: disambiguate candidates by clustering quality.

    For each candidate k we cluster with `cluster_fn(k)` and score the result:

      * ground truth available  -> Adjusted Rand Index (higher is better);
      * no ground truth         -> Silhouette Score on X (higher is better).

    The k with the best score wins. This grounds the final choice in the actual
    partition quality instead of the eigenvalue spectrum alone.

    Parameters
    ----------
    X          : feature matrix (or embedding) used for the Silhouette metric.
    candidates : shortlist of k values from `suggest_k_candidates`.
    cluster_fn : callable mapping k -> integer label array. Typically a closure
                 over the affinity/feature matrix, e.g.
                 `lambda k: spectral_ng_jordan_weiss(W, k)`.
    ground_truth : optional true labels; when given, ARI is used instead of
                 Silhouette.

    Returns
    -------
    dict with the chosen `best_k`, the `criterion` used, and the per-candidate
    `scores` (for plotting / reporting).
    """
    if not candidates:
        raise ValueError("candidates must be a non-empty sequence of k values.")

    criterion = "ARI" if ground_truth is not None else "silhouette"
    scores: dict[int, float] = {}

    for k in candidates:
        labels = np.asarray(cluster_fn(int(k)))
        n_found = len(np.unique(labels))
        if ground_truth is not None:
            score = float(adjusted_rand_score(ground_truth, labels))
        elif n_found < 2 or n_found >= len(labels):
            score = float("-inf")        # silhouette undefined for a single cluster
        else:
            score = float(silhouette_score(X, labels))
        scores[int(k)] = score

    best_k = max(scores, key=scores.get)
    return {"best_k": int(best_k), "criterion": criterion, "scores": scores}


def select_spectral_k(
    W: np.ndarray,
    X_eval: np.ndarray,
    reference_fn: Callable[[int], np.ndarray],
    ground_truth: np.ndarray | None = None,
    top_n: int = 4,
    k_min: int = 2,
    k_max: int = 10,
    laplacian: str = "sym",
) -> dict:
    """
    Convenience wrapper composing the full two-step spectral k-selection, so
    every dataset script/notebook selects k the same way.

    Step 1 — shortlist candidate k from the eigenvalue spectrum of `W`
             (`suggest_k_candidates`).
    Step 2 — disambiguate by clustering quality (`pick_best_k`): ARI against
             `ground_truth` when available, otherwise Silhouette on `X_eval`.

    `reference_fn(k) -> labels` is the clustering used to score each candidate.
    Passing the most robust method available (e.g. self-tuning spectral
    clustering) as the reference makes the chosen k reflect the intrinsic cluster
    structure; the resulting k is then applied uniformly to every algorithm for a
    fair comparison. `X_eval` is the feature matrix or spectral embedding used
    only for the Silhouette metric (unlabelled network datasets).

    Returns a single dict carrying both the diagnostics (eigenvalues, gaps, the
    naive single-gap estimate, the candidate shortlist) and the final choice
    (`best_k`, `criterion`, per-candidate `scores`) — ready to serialise.
    """
    gap = eigengap(W, laplacian=laplacian, k_max=max(k_max, top_n + 1))
    candidates = suggest_k_candidates(
        gap["eigenvalues"], top_n=top_n, k_min=k_min, k_max=k_max
    )
    chosen = pick_best_k(X_eval, candidates, reference_fn, ground_truth=ground_truth)
    return {
        "eigenvalues": gap["eigenvalues"],
        "gaps": gap["gaps"],
        "naive_suggested_k": gap["naive_suggested_k"],
        "candidates": candidates,
        "best_k": chosen["best_k"],
        "criterion": chosen["criterion"],
        "scores": chosen["scores"],
        "top_n": top_n,
    }
