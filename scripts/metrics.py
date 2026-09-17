"""
Evaluation metrics.

Supervised (ground truth available — the synthetic datasets):
    ARI   Adjusted Rand Index          [-0.5, 1], higher is better
    NMI   Normalised Mutual Information [0, 1],    higher is better

Unsupervised (no labels — the three network datasets):
    Silhouette       [-1, 1], higher is better
    Davies-Bouldin   [0, inf), LOWER is better

`evaluate` dispatches on whether ground-truth labels are supplied so a notebook
can call one function regardless of dataset type.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)


def supervised_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    """ARI and NMI against ground-truth labels."""
    return {
        "ARI": float(adjusted_rand_score(labels_true, labels_pred)),
        "NMI": float(normalized_mutual_info_score(labels_true, labels_pred)),
    }


def unsupervised_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Silhouette score and Davies-Bouldin index from the feature/embedding matrix.

    Both are undefined when a partition collapses to a single cluster; in that
    degenerate case we return NaN so the failure is visible rather than crashing.
    """
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def evaluate(
    labels_pred: np.ndarray,
    X: np.ndarray | None = None,
    labels_true: np.ndarray | None = None,
) -> dict:
    """
    Unified evaluation.

    If `labels_true` is provided, returns ARI + NMI (supervised). Otherwise
    returns silhouette + Davies-Bouldin computed on `X` (unsupervised). For the
    synthetic datasets we report the supervised metrics as the primary result.
    """
    if labels_true is not None:
        return supervised_metrics(labels_true, labels_pred)
    if X is None:
        raise ValueError("Unsupervised evaluation requires the feature matrix X.")
    return unsupervised_metrics(X, labels_pred)
