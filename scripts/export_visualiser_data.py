"""
Consolidate all notebook results into static JSON for the Next.js visualiser.

Produces (under visualiser/public/data/, and a copy of the dashboard file at
results/visualiser_data.json):

    dashboard.json            datasets + algorithms + flat results list
                              (metrics, cluster sizes, degeneracy flags)
    explorer/<id>.json        per-dataset geometry for the D3 views
                              (synthetic: 2-D points; networks: node positions +
                              sampled edges), plus per-algorithm labels/metrics

Metric sources:
    synthetic ARI/NMI              results/synthetic.json
    silhouette / Davies-Bouldin    results/network_eval_*.json (3 embeddings)
    modularity                     recomputed here (weighted for Airports,
                                   unweighted for Yeast/Facebook), or read from
                                   results/facebook_cache.json for Facebook
Degenerate/near-degenerate `flag`s are derived from the cluster-size profile,
matching the notebooks' diagnostic sections (e.g. Airports TVI, Facebook
self-tuning micro-clusters).

Run:  python -m scripts.export_visualiser_data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity

from . import SEED
from .network_datasets import load_airports, load_yeast, load_facebook, graph_to_adjacency
from .network_eval import build_embeddings
from .selection import select_spectral_k
from .spectral import (kmeans_baseline, unnormalised_spectral, spectral_shi_malik,
                       spectral_ng_jordan_weiss, self_tuning_spectral)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_DATA = REPO / "visualiser" / "public" / "data"
OUT_EXPLORER = OUT_DATA / "explorer"

ALGORITHMS = ["K-Means", "Unnormalised SC", "Shi-Malik SC",
              "Ng-Jordan-Weiss SC", "Self-Tuning SC"]

# map every label variant used across the notebooks to the canonical name
CANON = {
    "kmeans": "K-Means", "unnormalised": "Unnormalised SC", "shi_malik": "Shi-Malik SC",
    "ng_jordan_weiss": "Ng-Jordan-Weiss SC", "self_tuning": "Self-Tuning SC",
    "K-Means (baseline)": "K-Means", "K-Means (embedding)": "K-Means",
    "Unnormalised SC": "Unnormalised SC",
    "Normalised SC — Shi & Malik (2000)": "Shi-Malik SC", "Shi-Malik SC": "Shi-Malik SC",
    "Normalised SC — Ng, Jordan & Weiss (2001)": "Ng-Jordan-Weiss SC",
    "Ng-Jordan-Weiss SC": "Ng-Jordan-Weiss SC",
    "Self-Tuning SC — Zelnik-Manor & Perona (2004)": "Self-Tuning SC",
    "Self-Tuning SC (embedding)": "Self-Tuning SC",
}


def sizes_of(labels) -> list[int]:
    return sorted(np.bincount(np.asarray(labels)).tolist(), reverse=True)


def make_flag(dataset_id: str, algo: str, sizes: list[int], n: int):
    """Degeneracy flag from the cluster-size profile (see notebook diagnostics)."""
    if not sizes or len(sizes) < 2:
        return None
    largest, smallest = max(sizes), min(sizes)
    if smallest == 1:
        if dataset_id == "airports":
            return "degenerate — isolates a single low-degree node (TVI, a pendant leaf)"
        return "degenerate — isolates a single node"
    if largest / n >= 0.9:
        if dataset_id == "facebook" and algo == "Self-Tuning SC":
            return ("near-degenerate — local-scale micro-clusters split tight "
                    "friend-groups off a single giant community")
        if dataset_id.startswith("yeast"):
            return "near-degenerate — one giant cluster plus tiny low-degree fragments"
        return "near-degenerate — one dominant cluster with small fragments"
    return None


def silhouette_db_table(path: Path) -> dict:
    """Read a network_eval_*.json 'table' keyed by canonical algorithm name."""
    tab = json.load(open(path))["table"]
    out = {}
    for row in tab:
        out[CANON[row["algorithm"]]] = {
            "silhouette_unnorm": row["silhouette_unnorm"],
            "silhouette_sym": row["silhouette_sym"],
            "silhouette_shortest_path": row["silhouette_shortest_path"],
            "davies_bouldin_unnorm": row["davies_bouldin_unnorm"],
            "davies_bouldin_sym": row["davies_bouldin_sym"],
            "davies_bouldin_shortest_path": row["davies_bouldin_shortest_path"],
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic datasets (rings / moons / blobs)
# --------------------------------------------------------------------------- #
def build_synthetic(datasets, results, explorer):
    syn = json.load(open(RESULTS / "synthetic.json"))["subsets"]
    multi = json.load(open(RESULTS / "network_eval_synthetic.json"))["datasets"]
    from .graphs import rbf_affinity
    for name, sub in syn.items():
        did = f"synthetic_{name}"
        pretty = {"rings": "Rings", "moons": "Moons", "blobs": "Blobs"}[name]
        n = len(sub["points"])
        sil_tab = {CANON[r["algorithm"]]: r for r in multi[name]["table"]}
        # weighted modularity on the same RBF affinity graph (a proximity graph,
        # so Q here measures compactness — see notebook Section 10)
        W, _ = rbf_affinity(np.array(sub["points"]))
        Gsyn = nx.from_numpy_array(W)
        datasets.append({"id": did, "label": f"Synthetic — {pretty}", "n": n,
                         "type": "scatter", "group": "synthetic", "hasGroundTruth": True,
                         "k_true": sub["k_true"]})
        algo_geo = {}
        for key, a in sub["algorithms"].items():
            algo = CANON[key]
            sizes = sizes_of(a["labels"])
            s = sil_tab.get(algo, {})
            comms = {}
            for i, c in enumerate(a["labels"]):
                comms.setdefault(int(c), set()).add(i)
            Q = modularity(Gsyn, list(comms.values()), weight="weight")
            metrics = {
                "ARI": a["metrics"]["ARI"], "NMI": a["metrics"]["NMI"],
                "silhouette_unnorm": s.get("silhouette_unnorm"),
                "silhouette_sym": s.get("silhouette_sym"),
                "silhouette_shortest_path": s.get("silhouette_shortest_path"),
                "davies_bouldin_unnorm": s.get("davies_bouldin_unnorm"),
                "davies_bouldin_sym": s.get("davies_bouldin_sym"),
                "davies_bouldin_shortest_path": s.get("davies_bouldin_shortest_path"),
                "modularity": round(float(Q), 4),
                "modularityNote": "proximity graph — measures compactness",
            }
            flag = make_flag(did, algo, sizes, n)
            results.append({"dataset": did, "algorithm": algo, "k": a["k_used"],
                            "metrics": metrics, "clusterSizes": sizes, "flag": flag})
            algo_geo[algo] = {"labels": a["labels"], "k": a["k_used"],
                              "clusterSizes": sizes, "flag": flag, "metrics": metrics}
        explorer[did] = {"id": did, "label": f"Synthetic — {pretty}", "type": "scatter",
                         "n": n, "hasGroundTruth": True,
                         "points": sub["points"], "groundTruth": sub["ground_truth"],
                         "algorithms": algo_geo}


# --------------------------------------------------------------------------- #
# Network datasets
# --------------------------------------------------------------------------- #
def run_network(loader, weighted_mod):
    A, nodes, G = loader()
    emb, _ = build_embeddings(A, 10)
    Xe = emb["unnorm"]; Ad = A.toarray()
    sel = select_spectral_k(Ad, Xe, reference_fn=lambda k: spectral_ng_jordan_weiss(Ad, k),
                            ground_truth=None, top_n=3, k_min=2, k_max=8)
    k = sel["best_k"]
    labels = {
        "K-Means": np.asarray(kmeans_baseline(Xe, k)),
        "Unnormalised SC": np.asarray(unnormalised_spectral(Ad, k)),
        "Shi-Malik SC": np.asarray(spectral_shi_malik(Ad, k)),
        "Ng-Jordan-Weiss SC": np.asarray(spectral_ng_jordan_weiss(Ad, k)),
        "Self-Tuning SC": np.asarray(self_tuning_spectral(Xe, k)),
    }
    pos = nx.spring_layout(G, seed=SEED, iterations=50)
    P = np.array([pos[nodes[i]] for i in range(len(nodes))])
    weight = "weight" if weighted_mod else None
    idx = {nd: i for i, nd in enumerate(nodes)}
    Q = {}
    for algo, lab in labels.items():
        comms = {}
        for nd in G.nodes():
            comms.setdefault(int(lab[idx[nd]]), set()).add(nd)
        Q[algo] = modularity(G, list(comms.values()), weight=weight)
    return A, nodes, G, k, labels, P, Q


def facebook_from_cache():
    A, nodes, G = load_facebook()
    cj = json.load(open(RESULTS / "facebook_cache.json"))
    npz = np.load(RESULTS / "facebook_cache.npz")
    algos = cj["algos"]
    labels = {CANON[a]: npz[f"lab_{i}"] for i, a in enumerate(algos)}
    P = npz["pos"]
    Q = {CANON[a]: q for a, q in cj["Q"].items()}
    return A, nodes, G, int(cj["k"]), labels, P, Q


def edges_index(G, nodes, sample=None):
    idx = {nd: i for i, nd in enumerate(nodes)}
    E = [(idx[u], idx[v]) for u, v in G.edges()]
    if sample and len(E) > sample:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(E), size=sample, replace=False)
        E = [E[i] for i in sel]
    return E


def build_network(did, label, loader, weighted_mod, datasets, results, explorer,
                  note=None, edge_sample=None, use_cache=False, coords_map=None):
    if use_cache:
        A, nodes, G, k, labels, P, Q = facebook_from_cache()
    else:
        A, nodes, G, k, labels, P, Q = run_network(loader, weighted_mod)
    n = A.shape[0]
    sil_tab = silhouette_db_table(RESULTS / f"network_eval_{did}.json")

    entry = {"id": did, "label": label, "n": n, "type": "network", "group": "network",
             "hasGroundTruth": False, "k": k, "edges": int(A.nnz // 2)}
    if note:
        entry["note"] = note
    datasets.append(entry)

    # round positions to keep the JSON small
    Pr = np.round(P, 4)
    algo_geo = {}
    for algo, lab in labels.items():
        sizes = sizes_of(lab)
        s = sil_tab.get(algo, {})
        metrics = {
            "ARI": None, "NMI": None,
            "silhouette_unnorm": s.get("silhouette_unnorm"),
            "silhouette_sym": s.get("silhouette_sym"),
            "silhouette_shortest_path": s.get("silhouette_shortest_path"),
            "davies_bouldin_unnorm": s.get("davies_bouldin_unnorm"),
            "davies_bouldin_sym": s.get("davies_bouldin_sym"),
            "davies_bouldin_shortest_path": s.get("davies_bouldin_shortest_path"),
            "modularity": round(float(Q[algo]), 4),
        }
        flag = make_flag(did, algo, sizes, n)
        results.append({"dataset": did, "algorithm": algo, "k": int(k),
                        "metrics": metrics, "clusterSizes": sizes, "flag": flag})
        algo_geo[algo] = {"labels": [int(x) for x in lab], "k": int(k),
                          "clusterSizes": sizes, "flag": flag, "metrics": metrics}

    geo = {"id": did, "label": label, "type": "network", "n": n,
           "hasGroundTruth": False, "note": note,
           "positions": Pr.tolist(),
           "edges": edges_index(G, nodes, sample=edge_sample),
           "modularityWeighted": weighted_mod,
           "algorithms": algo_geo}

    # per-node [lat, lon] for the geographic map (airports); null where unmatched
    if coords_map is not None:
        latlon = []
        mapped = 0
        for nd in nodes:
            c = coords_map.get(str(nd))
            if c is None:
                latlon.append(None)
            else:
                latlon.append([round(float(c[0]), 4), round(float(c[1]), 4)])
                mapped += 1
        geo["latlon"] = latlon
        geo["mappedCount"] = mapped
        print(f"    geographic coords matched for {mapped}/{n} airports", flush=True)

    explorer[did] = geo


def main():
    OUT_EXPLORER.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    datasets, results, explorer = [], [], {}

    print("synthetic...", flush=True)
    build_synthetic(datasets, results, explorer)
    print("airports...", flush=True)
    from .network_datasets import load_airport_coords
    _coords_df = load_airport_coords()
    airport_coords = {str(r.airport): (r.lat, r.long) for r in _coords_df.itertuples(index=False)}
    build_network("airports", "US Airports", load_airports, True, datasets, results, explorer,
                  note="Weighted by flight counts. One airport (TVI) is an isolated pendant "
                       "leaf that causes degenerate splits under unnormalised methods.",
                  edge_sample=12000, coords_map=airport_coords)
    print("yeast...", flush=True)
    build_network("yeast", "Yeast PPI", load_yeast, False, datasets, results, explorer,
                  note="50% of proteins are degree-1; unnormalised methods fragment.",
                  edge_sample=None)
    print("facebook...", flush=True)
    build_network("facebook", "Facebook Social Circles", load_facebook, False,
                  datasets, results, explorer,
                  note="Unweighted friendships; lowest leaf fraction (1.9%) — unnormalised "
                       "methods succeed here.",
                  edge_sample=12000, use_cache=True)

    dashboard = {"datasets": datasets, "algorithms": ALGORITHMS, "results": results}
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    json.dump(dashboard, open(OUT_DATA / "dashboard.json", "w"), indent=2)
    json.dump(dashboard, open(RESULTS / "visualiser_data.json", "w"), indent=2)
    for did, geo in explorer.items():
        json.dump(geo, open(OUT_EXPLORER / f"{did}.json", "w"))
    print(f"\nWrote dashboard.json ({len(datasets)} datasets, {len(results)} results) "
          f"+ {len(explorer)} explorer files.")
    print("datasets:", [d["id"] for d in datasets])


if __name__ == "__main__":
    main()
