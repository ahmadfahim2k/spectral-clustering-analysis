"""
ADDITIONAL ANALYSIS (does not change the existing pipeline).

Compares the existing single-reference k-selection (Self-Tuning eigengap shortlist
+ ARI/Silhouette disambiguation, via `select_spectral_k`) against a five-algorithm
VOTING scheme:

  * each of the four spectral methods proposes a k from ITS OWN Laplacian's
    eigengaps (Unnormalised -> L; Shi-Malik & Ng-Jordan-Weiss -> L_sym, which share
    a spectrum; Self-Tuning -> the sym-Laplacian of its local-scaling affinity);
  * K-Means proposes a k from its own elbow (within-cluster SSE vs k);
  * a simple majority vote picks k; ties are broken with the same downstream
    quality score used elsewhere (ARI where ground truth exists, else Silhouette).

Reports, per dataset: each algorithm's proposal, the voted k, the single-reference
k, whether they differ, wall-clock timings for both, and — where the voted k
differs — the downstream clustering-quality change.

Run:  python -m scripts.k_voting            # all datasets
      python -m scripts.k_voting facebook   # one or more by name
Writes results/k_voting_comparison.json (merged across runs).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from . import SEED
from .datasets import make_synthetic
from .graphs import rbf_affinity, self_tuning_affinity
from .network_datasets import load_airports, load_yeast, load_facebook
from .network_eval import build_embeddings
from .selection import eigengap, suggest_k_candidates, pick_best_k, kmeans_elbow, select_spectral_k
from .spectral import (kmeans_baseline, unnormalised_spectral, spectral_shi_malik,
                       spectral_ng_jordan_weiss, self_tuning_spectral)
from .metrics import supervised_metrics
import networkx as nx
from networkx.algorithms.community import modularity

RESULTS = Path(__file__).resolve().parent.parent / "results" / "k_voting_comparison.json"
KMIN, KMAX = 2, 8


def _spectral_proposal(W, laplacian):
    ev = eigengap(W, laplacian=laplacian, k_max=max(KMAX, 8))["eigenvalues"]
    return suggest_k_candidates(ev, top_n=1, k_min=KMIN, k_max=KMAX)[0]


def collect_proposals(W_spectral, X_embed):
    """Each of the five algorithms' own top-k proposal."""
    sym_k = _spectral_proposal(W_spectral, "sym")            # Shi-Malik & NJW share L_sym spectrum
    A_st = self_tuning_affinity(X_embed, n_neighbors=7)
    elbow_k = int(kmeans_elbow(X_embed, range(1, KMAX + 1))["suggested_k"])
    return {
        "Unnormalised SC": _spectral_proposal(W_spectral, "unnormalised"),
        "Shi-Malik SC": sym_k,
        "Ng-Jordan-Weiss SC": sym_k,
        "Self-Tuning SC": _spectral_proposal(A_st, "sym"),
        "K-Means": min(KMAX, max(KMIN, elbow_k)),
    }


def vote(proposals, X_eval, reference_fn, ground_truth):
    counts = Counter(proposals.values())
    top = max(counts.values())
    winners = sorted(k for k, c in counts.items() if c == top)
    if len(winners) == 1:
        return winners[0], dict(counts), False
    chosen = pick_best_k(X_eval, winners, reference_fn, ground_truth=ground_truth)["best_k"]
    return int(chosen), dict(counts), True


def quality_at_k(W_spectral, X_embed, k, kind, y=None, G=None, weight=None):
    """Downstream quality of all five algorithms at a given k."""
    labs = {
        "K-Means": np.asarray(kmeans_baseline(X_embed, k)),
        "Unnormalised SC": np.asarray(unnormalised_spectral(W_spectral, k)),
        "Shi-Malik SC": np.asarray(spectral_shi_malik(W_spectral, k)),
        "Ng-Jordan-Weiss SC": np.asarray(spectral_ng_jordan_weiss(W_spectral, k)),
        "Self-Tuning SC": np.asarray(self_tuning_spectral(X_embed, k)),
    }
    out = {}
    for nm, lab in labs.items():
        if kind == "synthetic":
            out[nm] = round(supervised_metrics(y, lab)["ARI"], 3)
        else:
            comms = {}
            for i, c in enumerate(lab):
                comms.setdefault(int(c), set()).add(i)
            out[nm] = round(modularity(G, list(comms.values()), weight=weight), 3)
    return out  # ARI (synthetic) or modularity (network)


def process(name, kind, loader=None, weighted_mod=False):
    t0 = perf_counter()
    if kind == "synthetic":
        X, y = loader["X"], loader["y"]
        W_spectral = rbf_affinity(X)[0]
        X_embed = X
        reference_fn = lambda k: self_tuning_spectral(X, k)
        gt = y
        top_n_single = 4
        G, weight = None, None
    else:
        A, nodes, G = loader()
        W_spectral = A.toarray()
        X_embed = build_embeddings(A, 10)[0]["unnorm"]
        reference_fn = lambda k: spectral_ng_jordan_weiss(W_spectral, k)
        gt = None
        y = None
        top_n_single = 3
        weight = "weight" if weighted_mod else None
    t_build = perf_counter() - t0

    # existing single-reference method
    t1 = perf_counter()
    sel = select_spectral_k(W_spectral, X_embed, reference_fn=reference_fn,
                            ground_truth=gt, top_n=top_n_single, k_min=KMIN, k_max=KMAX)
    t_single = perf_counter() - t1
    k_single = int(sel["best_k"])

    # five-algorithm voting
    t2 = perf_counter()
    proposals = collect_proposals(W_spectral, X_embed)
    k_vote, counts, was_tie = vote(proposals, X_embed, reference_fn, gt)
    t_vote = perf_counter() - t2

    rec = {
        "dataset": name, "kind": kind,
        "proposals": proposals, "voteCounts": counts, "tieBroken": was_tie,
        "k_voted": k_vote, "k_single_reference": k_single, "differs": k_vote != k_single,
        "timings": {
            "graph_build_s": round(t_build, 3),
            "single_reference_k_selection_s": round(t_single, 3),
            "voting_s": round(t_vote, 3),
            "single_reference_total_s": round(t_build + t_single, 3),
            "voting_total_s": round(t_build + t_vote, 3),
        },
    }
    if k_vote != k_single:
        rec["quality_single_k"] = quality_at_k(W_spectral, X_embed, k_single, kind, y, G, weight)
        rec["quality_voted_k"] = quality_at_k(W_spectral, X_embed, k_vote, kind, y, G, weight)
    return rec


def _print(rec):
    print(f"\n=== {rec['dataset']} ===", flush=True)
    print(f"  proposals: {rec['proposals']}")
    print(f"  vote counts: {rec['voteCounts']}"
          f"{'  (tie broken downstream)' if rec['tieBroken'] else ''}")
    print(f"  voted k = {rec['k_voted']}   single-reference k = {rec['k_single_reference']}"
          f"   -> {'DIFFERS' if rec['differs'] else 'same'}")
    t = rec["timings"]
    print(f"  time: graph build {t['graph_build_s']}s | "
          f"single-ref k-sel {t['single_reference_k_selection_s']}s | voting {t['voting_s']}s")
    print(f"        full: single-ref {t['single_reference_total_s']}s  vs  voting {t['voting_total_s']}s")
    if rec["differs"]:
        print(f"  downstream ARI/modularity @single k={rec['k_single_reference']}: {rec['quality_single_k']}")
        print(f"  downstream ARI/modularity @voted  k={rec['k_voted']}:  {rec['quality_voted_k']}")


def main():
    np.random.seed(SEED)
    which = [a for a in sys.argv[1:]] or ["synthetic", "airports", "yeast", "facebook"]

    jobs = []
    if "synthetic" in which:
        data = make_synthetic(seed=SEED)
        for shape, ds in data.items():
            jobs.append((f"synthetic_{shape}", "synthetic", ds, False))
    if "airports" in which:
        jobs.append(("airports", "network", load_airports, True))
    if "yeast" in which:
        jobs.append(("yeast", "network", load_yeast, False))
    if "facebook" in which:
        jobs.append(("facebook", "network", load_facebook, False))

    existing = {}
    if RESULTS.exists():
        existing = {r["dataset"]: r for r in json.load(open(RESULTS)).get("records", [])}

    for name, kind, loader, wmod in jobs:
        rec = process(name, kind, loader=loader, weighted_mod=wmod)
        _print(rec)
        existing[name] = rec
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        order = ["synthetic_rings", "synthetic_moons", "synthetic_blobs",
                 "airports", "yeast", "facebook"]
        recs = [existing[k] for k in order if k in existing]
        json.dump({"seed": SEED, "k_min": KMIN, "k_max": KMAX, "records": recs},
                  open(RESULTS, "w"), indent=2)
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
