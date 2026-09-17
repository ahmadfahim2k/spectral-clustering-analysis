"""
Multi-embedding internal evaluation for the network datasets.

For a graph and a set of algorithm cluster-assignments, compute Silhouette Score
and Davies-Bouldin Index under three embeddings (unnormalised Laplacian,
symmetric normalised Laplacian, shortest-path MDS) — see `scripts.embeddings` —
and analyse whether the algorithm *ranking* is stable across embeddings.

Rationale (supervisor point 1): a single fixed embedding might happen to favour
some algorithms. Reporting all three, and checking ranking consistency, makes the
comparison robust and any embedding-dependence explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, davies_bouldin_score

from .embeddings import (
    laplacian_embedding, shortest_path_embedding,
    shortest_path_embedding_from_similarity,
)

# embedding key -> human label
EMBEDDINGS = {
    "unnorm": "Unnormalised Laplacian",
    "sym": "Symmetric normalised Laplacian",
    "shortest_path": "Shortest-path (MDS)",
}
# Silhouette: higher is better. Davies-Bouldin: lower is better.
METRIC_HIGHER_BETTER = {"silhouette": True, "davies_bouldin": False}


def build_embeddings(adjacency, n_components: int = 10, similarity_graph: bool = False):
    """
    Construct all three neutral embeddings for one dataset.

    `similarity_graph=False` (default, used for the sparse real networks) computes
    the shortest-path embedding from unweighted hop distance. `similarity_graph=
    True` (for the synthetic RBF affinity, which is fully connected) instead
    converts the affinity to edge lengths via -log and runs weighted shortest
    paths — see `embeddings.shortest_path_embedding_from_similarity`. The two
    Laplacian embeddings are identical in both cases.

    Returns (embeddings_dict, timings_dict).
    """
    embeddings = {
        "unnorm": laplacian_embedding(adjacency, n_components, normed=False),
        "sym": laplacian_embedding(adjacency, n_components, normed=True),
    }
    if similarity_graph:
        sp_coords, sp_seconds = shortest_path_embedding_from_similarity(adjacency, n_components)
    else:
        sp_coords, sp_seconds = shortest_path_embedding(adjacency, n_components)
    embeddings["shortest_path"] = sp_coords
    return embeddings, {"shortest_path_seconds": sp_seconds}


def _safe_metrics(coords: np.ndarray, labels) -> tuple[float, float]:
    labels = np.asarray(labels)
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan"), float("nan")
    return (
        float(silhouette_score(coords, labels)),
        float(davies_bouldin_score(coords, labels)),
    )


def score_under_embeddings(embeddings: dict, labels_by_algorithm: dict) -> pd.DataFrame:
    """
    Score clusterings against pre-built embeddings (one row per algorithm,
    silhouette_/davies_bouldin_ columns per embedding). Separated from
    `build_embeddings` so callers can build the embeddings once and reuse them
    (important for the larger graphs, where each embedding is memory-heavy).
    """
    rows = []
    for name, labels in labels_by_algorithm.items():
        row = {"algorithm": name}
        for key in EMBEDDINGS:
            sil, dbi = _safe_metrics(embeddings[key], labels)
            row[f"silhouette_{key}"] = sil
            row[f"davies_bouldin_{key}"] = dbi
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_multi_embedding(adjacency, labels_by_algorithm: dict, n_components: int = 10,
                             similarity_graph: bool = False):
    """
    Convenience wrapper: build all three embeddings then score every algorithm.
    Returns (dataframe, timings). Set `similarity_graph=True` for a fully-connected
    similarity graph (e.g. the synthetic RBF affinity). For large graphs, prefer
    building the embeddings once with `build_embeddings` and calling
    `score_under_embeddings`.
    """
    embeddings, timings = build_embeddings(adjacency, n_components,
                                           similarity_graph=similarity_graph)
    return score_under_embeddings(embeddings, labels_by_algorithm), timings


def rank_within_embedding(df: pd.DataFrame, metric: str, embedding: str) -> list[str]:
    """Algorithm names ordered best -> worst for one metric under one embedding."""
    col = f"{metric}_{embedding}"
    s = df.set_index("algorithm")[col]
    return s.sort_values(ascending=not METRIC_HIGHER_BETTER[metric]).index.tolist()


def ranking_consistency(df: pd.DataFrame, metric: str = "silhouette") -> dict:
    """
    Analyse whether the best->worst algorithm ranking is stable across the three
    embeddings for a given metric.

    Returns a dict with: the per-embedding ordering, the winner and top-2 set per
    embedding, whether the winner / top-2 agree across all embeddings, and the
    pairwise Spearman rank correlations between embeddings.
    """
    orders = {e: rank_within_embedding(df, metric, e) for e in EMBEDDINGS}
    winners = {e: orders[e][0] for e in EMBEDDINGS}
    top2 = {e: set(orders[e][:2]) for e in EMBEDDINGS}

    winner_agree = len(set(winners.values())) == 1
    top2_sets = list(top2.values())
    top2_agree = all(s == top2_sets[0] for s in top2_sets)

    # pairwise Spearman correlation of the full orderings
    algos = df["algorithm"].tolist()
    rankvec = {e: [orders[e].index(a) for a in algos] for e in EMBEDDINGS}
    keys = list(EMBEDDINGS)
    spearman = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rho, _ = spearmanr(rankvec[keys[i]], rankvec[keys[j]])
            spearman[f"{keys[i]}~{keys[j]}"] = float(rho)

    return {
        "metric": metric,
        "orders": orders,
        "winners": winners,
        "winner_agree": winner_agree,
        "top2": {e: sorted(top2[e]) for e in EMBEDDINGS},
        "top2_agree": top2_agree,
        "spearman": spearman,
    }


def consistency_verdict(cons: dict, dataset: str) -> str:
    """One-line human summary of a `ranking_consistency` result."""
    metric = cons["metric"]
    if cons["winner_agree"] and cons["top2_agree"]:
        w = next(iter(cons["winners"].values()))
        return (f"On {dataset}, the {metric} ranking is stable across all three "
                f"embeddings — {w} ranks first and the top-2 set is identical.")
    if cons["top2_agree"]:
        return (f"On {dataset}, the top-2 algorithms by {metric} are the same across "
                f"all three embeddings, but the winner differs: {cons['winners']}.")
    lines = ", ".join(f"{e}: {cons['orders'][e][0]}" for e in EMBEDDINGS)
    return (f"On {dataset}, the {metric} ranking is embedding-dependent — the "
            f"top algorithm changes with the embedding ({lines}).")


def format_comparison_table(df: pd.DataFrame, metric: str = "silhouette") -> pd.DataFrame:
    """Tidy per-embedding view of one metric (columns = embeddings)."""
    cols = ["algorithm"] + [f"{metric}_{e}" for e in EMBEDDINGS]
    out = df[cols].copy()
    out.columns = ["algorithm"] + [f"{metric}_{e}" for e in EMBEDDINGS]
    return out.round(4)
