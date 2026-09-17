"""
Benchmark for the network-dataset evaluation (supervisor points 1 and 2), run on
the THREE REAL datasets (US Airports, Yeast PPI, Facebook).

Point 2 — times `scipy.sparse.csgraph.shortest_path` (all-pairs, hop distance)
on each real graph and prints the wall-clock seconds.

Point 1 — runs the five algorithms on each real graph, scores every algorithm
under all three embeddings (unnormalised Laplacian, symmetric normalised
Laplacian, shortest-path MDS), and reports whether the algorithm ranking is
consistent across embeddings.

k is chosen per dataset by the project's two-step selection with Silhouette
disambiguation (no ground truth on these datasets).

Run:  python -m scripts.benchmark_network_eval
Writes results/network_eval_benchmark.json.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import SEED
from .embeddings import shortest_path_distance
from .network_datasets import REAL_LOADERS
from .network_eval import (
    EMBEDDINGS, build_embeddings, score_under_embeddings, ranking_consistency,
    consistency_verdict, format_comparison_table,
)
from .selection import select_spectral_k
from .spectral import (
    kmeans_baseline, unnormalised_spectral, spectral_shi_malik,
    spectral_ng_jordan_weiss, self_tuning_spectral,
)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "network_eval_benchmark.json"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
N_COMPONENTS = 10


def load_real_datasets() -> dict:
    data = {}
    for name, loader in REAL_LOADERS.items():
        A, nodes, G = loader()
        data[name] = {"A": A, "nodes": nodes, "n": A.shape[0], "m": int(A.nnz // 2)}
    return data


# --------------------------------------------------------------------------- #
# Point 2 — shortest-path timing
# --------------------------------------------------------------------------- #
def time_shortest_paths(data: dict) -> list[dict]:
    print("=" * 78)
    print("POINT 2 — all-pairs shortest-path timing on the REAL graphs (hop distance)")
    print("=" * 78)
    print(f"{'dataset':10s}{'nodes':>8s}{'edges':>9s}{'seconds':>12s}")
    rows = []
    for name, d in data.items():
        _, seconds = shortest_path_distance(d["A"])
        rows.append({"dataset": name, "nodes": d["n"], "edges": d["m"],
                     "shortest_path_seconds": round(seconds, 3)})
        print(f"{name:10s}{d['n']:>8d}{d['m']:>9d}{seconds:>12.3f}")
    return rows


# --------------------------------------------------------------------------- #
# Point 1 — multi-embedding ranking consistency on each real dataset
# --------------------------------------------------------------------------- #
def run_algorithms(A_dense, X_embed, k: int) -> dict:
    """
    Five clusterings of a network. The three graph-native spectral methods take
    the adjacency directly; the K-Means baseline and Self-Tuning SC operate on the
    unnormalised-Laplacian node embedding (a graph has no raw feature space).
    """
    return {
        "K-Means (embedding)": kmeans_baseline(X_embed, k),
        "Unnormalised SC": unnormalised_spectral(A_dense, k),
        "Shi-Malik SC": spectral_shi_malik(A_dense, k),
        "Ng-Jordan-Weiss SC": spectral_ng_jordan_weiss(A_dense, k),
        "Self-Tuning SC (embedding)": self_tuning_spectral(X_embed, k),
    }


def evaluate_dataset(name: str, d: dict) -> dict:
    print("\n" + "=" * 78)
    print(f"POINT 1 — {name}  (n={d['n']}, m={d['m']})")
    print("=" * 78)

    A = d["A"]
    # Build the three embeddings ONCE and reuse them for k-selection, the
    # algorithm inputs, and scoring. Rebuilding them (as the convenience wrapper
    # does) roughly triples peak memory, which OOMs on the 4039-node graph.
    embeddings, timings = build_embeddings(A, N_COMPONENTS)
    X_embed = embeddings["unnorm"]
    A_dense = A.toarray()

    # k via two-step selection, Silhouette disambiguation (no ground truth)
    sel = select_spectral_k(
        A_dense, X_embed,
        reference_fn=lambda k: spectral_ng_jordan_weiss(A_dense, k),
        ground_truth=None, top_n=3, k_min=2, k_max=8,
    )
    k = sel["best_k"]
    print(f"k-selection (two-step, Silhouette): candidates={sel['candidates']} -> k={k}")

    labels = run_algorithms(A_dense, X_embed, k)
    del A_dense
    gc.collect()
    df = score_under_embeddings(embeddings, labels)
    del embeddings
    gc.collect()

    print("\nSilhouette Score by embedding (higher is better):")
    print(format_comparison_table(df, "silhouette").to_string(index=False))
    print("\nDavies-Bouldin Index by embedding (lower is better):")
    print(format_comparison_table(df, "davies_bouldin").to_string(index=False))

    out = {"n": d["n"], "m": d["m"], "k": k, "candidates": sel["candidates"],
           "table": df.round(4).to_dict(orient="records"), "consistency": {}}

    print("\nRanking consistency:")
    for metric in ("silhouette", "davies_bouldin"):
        cons = ranking_consistency(df, metric=metric)
        verdict = consistency_verdict(cons, dataset=name)
        print(f"  - {verdict}")
        print(f"      Spearman(rank) between embeddings: "
              f"{ {k2: round(v, 3) for k2, v in cons['spearman'].items()} }")
        out["consistency"][metric] = {
            "winners": cons["winners"], "winner_agree": cons["winner_agree"],
            "top2_agree": cons["top2_agree"], "spearman": cons["spearman"],
            "verdict": verdict,
        }
    return out


def _part_path(name: str) -> Path:
    return RESULTS.parent / f"network_eval_{name}.json"


def main():
    """
    Usage:
        python -m scripts.benchmark_network_eval            # all three datasets
        python -m scripts.benchmark_network_eval facebook   # one (or more) by name

    Each dataset's results are written to results/network_eval_<name>.json as soon
    as it finishes (so a long run can be done dataset-by-dataset and is robust to
    interruption); the combined results/network_eval_benchmark.json is rebuilt
    from those parts at the end.
    """
    import sys
    np.random.seed(SEED)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    which = [a for a in sys.argv[1:] if a in REAL_LOADERS] or list(REAL_LOADERS)
    data = {name: None for name in which}
    for name in which:
        A, nodes, G = REAL_LOADERS[name]()
        data[name] = {"A": A, "nodes": nodes, "n": A.shape[0], "m": int(A.nnz // 2)}

    timing = time_shortest_paths(data)
    timing_by = {r["dataset"]: r for r in timing}

    for name, d in data.items():
        out = evaluate_dataset(name, d)
        out["shortest_path_seconds"] = timing_by[name]["shortest_path_seconds"]
        with open(_part_path(name), "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> wrote {_part_path(name).name}")

    # combine whatever per-dataset parts exist into the final file
    combined = {}
    for name in REAL_LOADERS:
        p = _part_path(name)
        if p.exists():
            combined[name] = json.load(open(p))
    with open(RESULTS, "w") as f:
        json.dump({"seed": SEED, "source": "real datasets", "datasets": combined}, f, indent=2)
    print(f"\nWrote {RESULTS} ({len(combined)} datasets)")


if __name__ == "__main__":
    main()
