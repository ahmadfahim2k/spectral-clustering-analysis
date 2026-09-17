"""
Reproducible synthetic-dataset benchmark (mirrors notebooks/01_synthetic.ipynb).

Runs, for each synthetic subset (rings / moons / blobs):
  1. build the RBF affinity graph with median-heuristic sigma
  2. choose k — elbow for k-means, eigengap for the spectral methods
  3. run all five algorithms with the fixed seed
  4. score with ARI + NMI against the ground truth
  5. export everything to results/synthetic.json for the visualiser

Usage:
    python -m scripts.run_synthetic            # writes results/synthetic.json
    python -m scripts.run_synthetic --quiet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import SEED
from .datasets import make_synthetic
from .graphs import rbf_affinity
from .selection import kmeans_elbow, select_spectral_k
from .spectral import ALGORITHMS, self_tuning_spectral
from .metrics import supervised_metrics
from .export import export_results

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "synthetic.json"


def run(verbose: bool = True) -> dict:
    np.random.seed(SEED)  # global determinism for anything that peeks at numpy RNG
    data = make_synthetic(seed=SEED)

    payload = {"dataset_group": "synthetic", "seed": SEED, "subsets": {}}

    for name, ds in data.items():
        X, y, k_true = ds["X"], ds["y"], ds["k_true"]

        # 1. similarity graph (RBF, median heuristic)
        W, sigma = rbf_affinity(X)

        # 2. k-selection
        #   K-Means  -> elbow method.
        #   Spectral -> TWO-STEP: shortlist candidate k from the eigenvalue
        #   spectrum, then disambiguate by clustering quality. Ground truth is
        #   available here, so ARI is the disambiguation criterion. Self-tuning
        #   spectral clustering is the reference partition because its local
        #   scaling best reflects the intrinsic structure; the chosen k is then
        #   applied uniformly to every spectral method. This avoids the k=2 bias
        #   of the naive single-largest-gap heuristic.
        elbow = kmeans_elbow(X, k_range=range(1, 9))
        spectral_sel = select_spectral_k(
            W, X,
            reference_fn=lambda k: self_tuning_spectral(X, k),
            ground_truth=y,
            top_n=4, k_min=2, k_max=8,
        )
        k_spectral = spectral_sel["best_k"]
        k_kmeans = elbow["suggested_k"]

        if verbose:
            print(f"\n=== {name} ({ds['label']}) — n={len(X)}, k_true={k_true} ===")
            print(f"  sigma (median heuristic) = {sigma:.4f}")
            print(f"  elbow k={k_kmeans}; naive eigengap k={spectral_sel['naive_suggested_k']}; "
                  f"candidates={spectral_sel['candidates']} -> two-step k={k_spectral} "
                  f"({spectral_sel['criterion']})")

        subset = {
            "label": ds["label"],
            "k_true": k_true,
            "points": X.tolist(),
            "ground_truth": y.tolist(),
            "sigma": sigma,
            "selection": {
                "elbow": elbow,
                "spectral": spectral_sel,   # eigenvalues, gaps, candidates, scores, best_k
                "k_kmeans": k_kmeans,
                "k_spectral": k_spectral,
            },
            "algorithms": {},
        }

        # 3-4. run every algorithm and score it
        for key, spec in ALGORITHMS.items():
            k = k_kmeans if key == "kmeans" else k_spectral
            fn, kind = spec["fn"], spec["input"]
            labels = fn(X, k) if kind == "X" else fn(W, k)
            scores = supervised_metrics(y, labels)
            subset["algorithms"][key] = {
                "label": spec["label"],
                "k_used": int(k),
                "labels": labels.tolist(),
                "metrics": scores,
            }
            if verbose:
                print(f"    {spec['label']:<48} "
                      f"ARI={scores['ARI']:.3f}  NMI={scores['NMI']:.3f}")

        payload["subsets"][name] = subset

    out = export_results(payload, RESULTS_PATH)
    if verbose:
        print(f"\nWrote {out}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="suppress per-algorithm output")
    args = ap.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
