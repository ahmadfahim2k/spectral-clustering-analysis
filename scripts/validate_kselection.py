"""
Diagnostic: is Silhouette-based k-selection trustworthy?

On the synthetic datasets we know the true k, so we can check whether swapping the
two-step disambiguation criterion from ARI (supervised) to Silhouette
(unsupervised) still recovers the correct number of clusters. This tells us how
much to trust Silhouette-based selection on the real-world networks (Airports,
Yeast PPI, Facebook), where no ground truth is available.

This is a *diagnostic only*. ARI remains the primary disambiguation criterion in
the notebook and in `scripts.run_synthetic`; nothing here changes that.

Silhouette is computed on the standardised feature matrix, exactly as
`select_spectral_k` does when `ground_truth=None`.
"""

from __future__ import annotations

import numpy as np

from . import SEED
from .datasets import make_synthetic
from .graphs import rbf_affinity
from .selection import select_spectral_k
from .spectral import self_tuning_spectral


def silhouette_vs_ari(
    n_samples: int = 300,
    seed: int = SEED,
    top_n: int = 4,
    k_min: int = 2,
    k_max: int = 8,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the two-step k-selection twice per synthetic dataset — once with ARI and
    once with Silhouette disambiguation — and compare the chosen k.

    Returns a list of per-dataset dicts. When `verbose`, prints a side-by-side
    table plus a plain-language verdict.
    """
    np.random.seed(seed)
    data = make_synthetic(n_samples=n_samples, seed=seed)

    results: list[dict] = []
    for name, ds in data.items():
        X, y, k_true = ds["X"], ds["y"], ds["k_true"]
        W, _ = rbf_affinity(X)
        # identical selection except for the disambiguation criterion
        ref = lambda k, X=X: self_tuning_spectral(X, k)
        via_ari = select_spectral_k(
            W, X, reference_fn=ref, ground_truth=y,
            top_n=top_n, k_min=k_min, k_max=k_max,
        )
        via_sil = select_spectral_k(
            W, X, reference_fn=ref, ground_truth=None,
            top_n=top_n, k_min=k_min, k_max=k_max,
        )
        results.append({
            "dataset": name,
            "true_k": k_true,
            "k_via_ARI": via_ari["best_k"],
            "k_via_Silhouette": via_sil["best_k"],
            "candidates": via_sil["candidates"],
            "silhouette_scores": {k: round(v, 3) for k, v in via_sil["scores"].items()},
            "match_true": via_sil["best_k"] == k_true,
            "match_ari": via_sil["best_k"] == via_ari["best_k"],
        })

    if verbose:
        _print_table(results)
    return results


def _print_table(results: list[dict]) -> None:
    header = f"{'dataset':<9}{'true_k':<8}{'k_via_ARI':<11}{'k_via_Silhouette':<18}{'match?'}"
    print(header)
    print("-" * len(header))
    for r in results:
        mark = "yes" if r["match_true"] else "NO"
        print(f"{r['dataset']:<9}{r['true_k']:<8}{r['k_via_ARI']:<11}"
              f"{r['k_via_Silhouette']:<18}{mark}")

    agree = [r["dataset"] for r in results if r["match_true"]]
    disagree = [r["dataset"] for r in results if not r["match_true"]]

    print()
    if not disagree:
        print("VERDICT: Silhouette disambiguation recovers the correct k on ALL three "
              "datasets.\n         This validates using Silhouette for k-selection on the "
              "real-world\n         datasets, where no ground truth is available.")
    else:
        print(f"VERDICT: Silhouette matches the true k on {agree or 'none'} but DISAGREES "
              f"on {disagree}.\n         Silhouette rewards compact/convex clusters, so it "
              "is a biased criterion\n         for non-convex structure. On the unlabelled "
              "real datasets, treat\n         Silhouette-selected k as a guide only and "
              "cross-check it against the\n         eigengap candidate shortlist (and, where "
              "possible, cluster stability).")


if __name__ == "__main__":
    silhouette_vs_ari()
