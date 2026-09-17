"""
COMP702 dissertation — shared spectral-clustering pipeline.

Reusable, reproducible building blocks imported by every dataset notebook:

    graphs      similarity-graph construction (RBF median heuristic, self-tuning)
    spectral    the four spectral clustering algorithms + a KMeans baseline
    selection   k-selection heuristics (KMeans elbow, spectral eigengap)
    metrics     supervised (ARI/NMI) and unsupervised (silhouette/DBI) scoring
    datasets    synthetic dataset generation
    export      JSON serialisation of results for the Next.js visualiser

A single global seed (SEED = 42) is used everywhere for reproducibility.
"""

SEED = 42

__all__ = ["graphs", "spectral", "selection", "metrics", "datasets", "export", "SEED"]
