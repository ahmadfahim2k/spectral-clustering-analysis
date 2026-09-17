"""
Embeddings for internal cluster-quality evaluation on the network datasets.

The network datasets (US Airports, Yeast PPI, Facebook) have no ground truth, so
Silhouette Score and Davies-Bouldin Index are computed in a fixed *embedding
space* that is the same for every algorithm (a neutral comparison space). To
avoid committing to a single embedding that might flatter some algorithms, we
provide three:

    laplacian_embedding(A, normed=False)  unnormalised Laplacian eigenvectors
    laplacian_embedding(A, normed=True)   symmetric normalised Laplacian (L_sym)
    shortest_path_embedding(A)            classical MDS of all-pairs graph distance

Each returns an (n_nodes x n_components) coordinate matrix on which both internal
indices are then computed the same way.

All-pairs shortest paths use `scipy.sparse.csgraph.shortest_path`; the wall-clock
time is returned so it can be reported (see `shortest_path_distance`).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh as dense_eigh
from scipy.sparse.csgraph import laplacian, shortest_path
from scipy.sparse.linalg import eigsh


def _as_sparse(adjacency) -> sp.csr_matrix:
    """Coerce an adjacency (dense, sparse, or networkx) to a CSR float matrix."""
    if sp.issparse(adjacency):
        return adjacency.tocsr().astype(float)
    return sp.csr_matrix(np.asarray(adjacency, dtype=float))


def _smallest_eigenvectors(M: sp.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    k eigenvectors of a symmetric PSD matrix with the smallest eigenvalues.

    Uses ARPACK (`eigsh`, which='SA') with a dense fallback for small or
    ill-conditioned problems, so it is robust across the dataset sizes here.
    """
    n = M.shape[0]
    k = int(min(k, n - 1))
    # Dense eigh restricted to the k smallest eigenpairs (subset_by_index) is fast
    # and light up to a few thousand nodes, and avoids ARPACK's slow convergence
    # on the smallest eigenvalues of a Laplacian. Computing only k vectors (not a
    # full decomposition) keeps peak memory well under control on the 4039-node
    # graph. All three network datasets fall here.
    if n <= 5000:
        Md = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)
        vals, vecs = dense_eigh(Md, subset_by_index=[0, k - 1])
        return vals, vecs
    # Very large graphs: shift-invert around 0 for the smallest eigenvalues.
    try:
        Mr = (M + 1e-6 * sp.identity(n, format="csr")).tocsc()
        vals, vecs = eigsh(Mr, k=k, sigma=0, which="LM")
        order = np.argsort(vals)
        return vals[order] - 1e-6, vecs[:, order]
    except Exception:
        vals, vecs = np.linalg.eigh(M.toarray())
        return vals[:k], vecs[:, :k]


def laplacian_embedding(adjacency, n_components: int = 10, normed: bool = False) -> np.ndarray:
    """
    Spectral embedding from the graph Laplacian.

    Parameters
    ----------
    adjacency   : (n, n) graph adjacency (weighted or unweighted).
    n_components: embedding dimension (number of informative eigenvectors).
    normed      : False -> unnormalised Laplacian L = D - A;
                  True  -> symmetric normalised Laplacian L_sym = I - D^-1/2 A D^-1/2.

    The trivial (constant) eigenvector at eigenvalue ~0 is dropped, so the result
    has `n_components` informative dimensions.
    """
    A = _as_sparse(adjacency)
    L = laplacian(A, normed=normed)
    L = sp.csr_matrix(L)
    # take one extra and drop the leading (near-constant) trivial eigenvector
    vals, vecs = _smallest_eigenvectors(L, n_components + 1)
    return vecs[:, 1 : n_components + 1]


def shortest_path_distance(adjacency, method: str = "D", directed: bool = False,
                           unweighted: bool = True):
    """
    All-pairs shortest-path distance matrix via `scipy.sparse.csgraph.shortest_path`.

    Returns (distance_matrix, elapsed_seconds). Disconnected components produce
    `inf` entries; these are replaced by (max finite distance + 1) so downstream
    metrics remain defined. The timing measures only the shortest-path call.

    `unweighted=True` (default) uses hop-count graph distance. This is the right
    notion of "graph distance" for these datasets: the yeast/facebook edges are
    unweighted, and the airport edge weights are flight *counts* (a similarity,
    not a distance), so they must not be interpreted as path lengths.
    """
    A = _as_sparse(adjacency)
    t0 = time.perf_counter()
    D = shortest_path(A, method=method, directed=directed, unweighted=unweighted)
    elapsed = time.perf_counter() - t0

    finite = np.isfinite(D)
    if not finite.all():
        fill = (D[finite].max() + 1.0) if finite.any() else 1.0
        D = np.where(finite, D, fill)
    return D, elapsed


def classical_mds(distance: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Classical MDS / Principal Coordinates Analysis of a distance matrix.

    Double-centres the squared distances and returns the top `n_components`
    principal coordinates, giving a Euclidean embedding on which Silhouette and
    Davies-Bouldin can be computed like the Laplacian embeddings.
    """
    D = np.asarray(distance, dtype=float)
    n = D.shape[0]
    D2 = D ** 2
    row = D2.mean(axis=1, keepdims=True)
    col = D2.mean(axis=0, keepdims=True)
    B = -0.5 * (D2 - row - col + D2.mean())
    B = (B + B.T) / 2.0
    k = int(min(n_components, n - 1))
    if n <= 5000:
        # k largest eigenpairs only (subset_by_index counts from the smallest)
        vals, vecs = dense_eigh(B, subset_by_index=[n - k, n - 1])
        order = np.argsort(-vals)
        vals, vecs = vals[order], vecs[:, order]
    else:
        vals, vecs = eigsh(B, k=k, which="LA")
        order = np.argsort(-vals)
        vals, vecs = vals[order], vecs[:, order]
    pos = vals > 0
    return vecs[:, pos] * np.sqrt(vals[pos])


def shortest_path_embedding(adjacency, n_components: int = 10, unweighted: bool = True):
    """
    Embedding from graph distances: classical MDS of the all-pairs shortest-path
    (hop-count) matrix. Returns (coords, shortest_path_seconds).
    """
    D, elapsed = shortest_path_distance(adjacency, unweighted=unweighted)
    return classical_mds(D, n_components), elapsed


def similarity_to_distance(affinity) -> np.ndarray:
    """
    Convert a *similarity* graph to edge lengths for shortest-path use.

    length_ij = -log(W_ij). For a Gaussian RBF affinity
    W_ij = exp(-||x_i-x_j||^2 / 2σ^2) this returns ||x_i-x_j||^2 / 2σ^2, i.e. a
    monotone function of the Euclidean distance — so shortest paths on the
    resulting graph are a sensible geodesic distance. Non-edges (W_ij = 0) map to
    "absent" (0 in scipy's csgraph convention). Zero diagonal.
    """
    W = _as_sparse(affinity).toarray()
    np.fill_diagonal(W, 0.0)
    with np.errstate(divide="ignore"):
        length = -np.log(W)
    length[~np.isfinite(length)] = 0.0     # W<=0 -> no edge
    np.fill_diagonal(length, 0.0)
    return length


def shortest_path_embedding_from_similarity(affinity, n_components: int = 10):
    """
    Shortest-path embedding for a *similarity* graph (e.g. the synthetic RBF
    affinity, which is fully connected so hop distance would be a degenerate all-
    ones). Converts affinity to edge lengths via `similarity_to_distance`, runs
    weighted all-pairs shortest path, then classical MDS. Returns
    (coords, shortest_path_seconds).
    """
    length = similarity_to_distance(affinity)
    D, elapsed = shortest_path_distance(sp.csr_matrix(length), unweighted=False)
    return classical_mds(D, n_components), elapsed
