"""
Network dataset loaders (for milestones 2-4) and size-matched proxy generators.

Loaders read an edge list you have downloaded locally and return a scipy sparse
adjacency plus the node list. Download sources:

    US Airports        Kaggle (airports as nodes, flights as edges)
    Yeast PPI          NetworkRepository — bio-yeast
    Facebook circles   Stanford SNAP — ego-Facebook (facebook_combined.txt)

Until those files are in place, `proxy_network` builds graphs with the SAME node
and edge counts so the shortest-path timing and the multi-embedding evaluation
can be exercised end-to-end. Because all-pairs shortest-path wall-clock time is
governed by graph size (n nodes, m edges), a size-matched proxy gives a valid
timing estimate for the real dataset.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

# repo-root data folder (works regardless of the current working directory)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Known / representative sizes. Facebook and yeast are exact; the airport count
# depends on the exact Kaggle export, so it is an estimate to refine once the
# real file is loaded.
DATASET_SIZES = {
    "airports": {"n": 727, "m": 22955, "exact": True,
                 "note": "aggregated from the raw Kaggle flight CSV; largest component"},
    "yeast":    {"n": 1458, "m": 1948, "exact": True,
                 "note": "NetworkRepository bio-yeast (connected)"},
    "facebook": {"n": 4039, "m": 88234, "exact": True,
                 "note": "SNAP ego-Facebook (facebook_combined)"},
}


def graph_to_adjacency(G: nx.Graph):
    """Return (csr_adjacency, node_list) for a networkx graph."""
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr", dtype=float)
    return A, nodes


def load_edgelist(
    path: str | Path,
    weighted: bool = False,
    delimiter: str | None = None,
    comments: str = "#",
    largest_component: bool = True,
):
    """
    Load an undirected graph from an edge-list file.

    Parameters
    ----------
    weighted          : if True, expects a third column of edge weights.
    delimiter         : column separator (None -> any whitespace).
    largest_component : restrict to the largest connected component (recommended,
                        so shortest paths are finite and embeddings are stable).

    Returns (csr_adjacency, node_list, networkx_graph).
    """
    path = Path(path)
    if weighted:
        G = nx.read_weighted_edgelist(path, comments=comments, delimiter=delimiter)
    else:
        G = nx.read_edgelist(path, comments=comments, delimiter=delimiter)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if largest_component and G.number_of_nodes() and not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()
    A, nodes = graph_to_adjacency(G)
    return A, nodes, G


def _finalize(G: nx.Graph, largest_component: bool = True):
    """Drop self-loops, optionally restrict to the largest connected component,
    and return (csr_adjacency, node_list, graph)."""
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if largest_component and G.number_of_nodes() and not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()
    A, nodes = graph_to_adjacency(G)
    return A, nodes, G


# --------------------------------------------------------------------------- #
# Facebook — SNAP ego-Facebook (edge list)
# --------------------------------------------------------------------------- #
def load_facebook(path=None, largest_component: bool = True):
    """SNAP ego-Facebook: whitespace-separated 'u v' edge list, undirected."""
    path = Path(path) if path else DATA_DIR / "facebook_combined.txt"
    return load_edgelist(path, weighted=False, largest_component=largest_component)


# --------------------------------------------------------------------------- #
# Yeast PPI — NetworkRepository bio-yeast (Matrix Market)
# --------------------------------------------------------------------------- #
def load_yeast(path=None, largest_component: bool = True):
    """
    NetworkRepository bio-yeast in Matrix Market format
    (`%%MatrixMarket matrix coordinate pattern symmetric`). Read with
    `scipy.io.mmread`, binarise (pattern = unweighted), and return the same
    (adjacency, nodes, graph) structure as the other loaders.
    """
    if path is None:
        for cand in (DATA_DIR / "bio-yeast" / "bio-yeast.mtx", DATA_DIR / "bio-yeast.mtx"):
            if cand.exists():
                path = cand
                break
        else:
            raise FileNotFoundError("bio-yeast.mtx not found under data/")
    M = sio.mmread(str(path)).tocsr().astype(float)
    M.data[:] = 1.0                      # 'pattern' -> unweighted
    M = M.maximum(M.T)                   # ensure symmetric
    G = nx.from_scipy_sparse_array(M)
    return _finalize(G, largest_component)


# --------------------------------------------------------------------------- #
# US Airports — raw Kaggle flight-level CSV -> aggregated weighted graph
# --------------------------------------------------------------------------- #
def build_airport_graph(
    raw_path=None,
    compact_path=None,
    chunksize: int = 2_000_000,
) -> Path:
    """
    Aggregate the ~500MB flight-level CSV into a compact airport-to-airport edge
    list without loading the whole file into memory.

    Nodes = unique airport codes; each undirected pair's weight = total number of
    flights between them (sum of the `Flights` column across both directions and
    all dates). Only three columns are read, in chunks. The result is written to
    `compact_path` (source,target,weight) so the raw file is processed only once.

    Returns the compact file path.
    """
    raw_path = Path(raw_path) if raw_path else DATA_DIR / "Airports2.csv"
    compact_path = Path(compact_path) if compact_path else DATA_DIR / "airports_graph.csv"

    usecols = ["Origin_airport", "Destination_airport", "Flights"]
    acc: dict[tuple[str, str], float] = defaultdict(float)

    reader = pd.read_csv(raw_path, usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk.dropna(subset=usecols)
        chunk["Flights"] = pd.to_numeric(chunk["Flights"], errors="coerce")
        chunk = chunk.dropna(subset=["Flights"])
        o = chunk["Origin_airport"].astype(str).str.strip()
        d = chunk["Destination_airport"].astype(str).str.strip()
        # canonical undirected pair (sorted so A-B and B-A merge)
        pair = np.sort(np.stack([o.values, d.values], axis=1), axis=1)
        a, b = pair[:, 0], pair[:, 1]
        keep = a != b                    # drop self-loops
        grp = (pd.DataFrame({"a": a[keep], "b": b[keep], "w": chunk["Flights"].values[keep]})
               .groupby(["a", "b"], sort=False)["w"].sum())
        for (x, y), w in grp.items():
            acc[(x, y)] += float(w)

    out = pd.DataFrame(((x, y, w) for (x, y), w in acc.items()),
                       columns=["source", "target", "weight"])
    out.to_csv(compact_path, index=False)
    return compact_path


def load_airports(compact_path=None, raw_path=None,
                  largest_component: bool = True, rebuild: bool = False):
    """
    Load the aggregated US airport network.

    Reads the compact `data/airports_graph.csv` (source,target,weight); if it does
    not exist (or `rebuild=True`), it is built once from the raw Kaggle CSV via
    `build_airport_graph`. Edge weight = number of flights between the pair.
    """
    compact_path = Path(compact_path) if compact_path else DATA_DIR / "airports_graph.csv"
    if rebuild or not compact_path.exists():
        build_airport_graph(raw_path=raw_path, compact_path=compact_path)
    df = pd.read_csv(compact_path)
    df = df[df["weight"] > 0]            # drop 0-flight artifact edges
    G = nx.from_pandas_edgelist(df, "source", "target", edge_attr="weight")
    return _finalize(G, largest_component)


def build_airport_coords(raw_path=None, coords_path=None, chunksize: int = 2_000_000) -> Path:
    """
    Extract an airport-code -> (latitude, longitude) table from the raw Kaggle
    flight CSV, streamed in chunks. Coordinates appear in both the origin columns
    (Org_airport_lat/long) and destination columns (Dest_airport_lat/long); we
    take the first non-null pair seen per code. Cached to `coords_path` so the
    500 MB file is read only once.
    """
    raw_path = Path(raw_path) if raw_path else DATA_DIR / "Airports2.csv"
    coords_path = Path(coords_path) if coords_path else DATA_DIR / "airports_coords.csv"

    coords: dict[str, tuple[float, float]] = {}
    pairs = [("Origin_airport", "Org_airport_lat", "Org_airport_long"),
             ("Destination_airport", "Dest_airport_lat", "Dest_airport_long")]
    usecols = [c for trio in pairs for c in trio]
    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=chunksize):
        for code_col, lat_col, lon_col in pairs:
            sub = chunk[[code_col, lat_col, lon_col]].copy()
            sub[lat_col] = pd.to_numeric(sub[lat_col], errors="coerce")
            sub[lon_col] = pd.to_numeric(sub[lon_col], errors="coerce")
            sub = sub.dropna()
            for code, lat, lon in sub.itertuples(index=False):
                code = str(code).strip()
                if code not in coords:
                    coords[code] = (float(lat), float(lon))

    out = pd.DataFrame(((c, lat, lon) for c, (lat, lon) in coords.items()),
                       columns=["airport", "lat", "long"])
    out.to_csv(coords_path, index=False)
    return coords_path


def load_airport_coords(coords_path=None, raw_path=None, rebuild: bool = False) -> "pd.DataFrame":
    """Load (building/caching if needed) the airport-code -> lat/long table."""
    coords_path = Path(coords_path) if coords_path else DATA_DIR / "airports_coords.csv"
    if rebuild or not coords_path.exists():
        build_airport_coords(raw_path=raw_path, coords_path=coords_path)
    return pd.read_csv(coords_path)


# registry so the benchmark can iterate over the real datasets uniformly
REAL_LOADERS = {
    "airports": load_airports,
    "yeast": load_yeast,
    "facebook": load_facebook,
}


def proxy_network(n: int, m: int, seed: int = 42, communities: int | None = None,
                  out_ratio: float = 1 / 12):
    """
    Size-matched proxy graph with `n` nodes and ~`m` edges.

    communities=None -> Erdos-Renyi G(n, m) (for pure timing, where structure is
    irrelevant). communities=c -> a stochastic block model with c planted
    communities and the same target edge count (for exercising the clustering +
    multi-embedding evaluation with meaningful structure). `out_ratio` sets the
    between-/within-community edge-probability ratio: small = well separated,
    larger (e.g. 0.25) = noisier, more realistic community overlap.

    Returns (csr_adjacency, node_list, networkx_graph).
    """
    if communities is None:
        G = nx.gnm_random_graph(n, m, seed=seed)
    else:
        sizes = [n // communities] * communities
        sizes[-1] += n - sum(sizes)
        # in/out probabilities chosen to hit ~m edges at the requested mixing
        avg_deg = 2.0 * m / n
        p_in = min(1.0, avg_deg / (n / communities * (1 + (communities - 1) * out_ratio)))
        p_out = p_in * out_ratio
        probs = np.full((communities, communities), p_out)
        np.fill_diagonal(probs, p_in)
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()
    A, nodes = graph_to_adjacency(G)
    return A, nodes, G
