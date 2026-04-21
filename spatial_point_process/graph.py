"""Graph construction utilities."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_knn_graph(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
    """Build a symmetric deduplicated kNN edge index for points."""
    if coords.shape[0] < 2:
        raise ValueError("Need at least 2 points to build a graph")
    k = min(k_neighbors + 1, coords.shape[0])
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]

    src = np.repeat(np.arange(coords.shape[0]), idx.shape[1] - 1)
    dst = idx[:, 1:].reshape(-1)
    edges = np.stack([src, dst], axis=0)
    rev = edges[[1, 0], :]
    edges = np.concatenate([edges, rev], axis=1)
    key = edges[0] * coords.shape[0] + edges[1]
    keep = np.unique(key, return_index=True)[1]
    return edges[:, np.sort(keep)]
