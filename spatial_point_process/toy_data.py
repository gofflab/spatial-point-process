"""Toy synthetic spatial transcriptomics data generation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_toy_dataset(n_per_domain: int = 250, seed: int = 0, include_z: bool = False) -> pd.DataFrame:
    """Create a 3-domain synthetic dataset with separable spatial/gene structure."""
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i}" for i in range(12)]
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.2]], dtype=np.float32)
    gene_weights = np.array(
        [
            [8, 8, 8, 3, 2, 1, 1, 1, 1, 1, 0.5, 0.5],
            [1, 1, 1, 8, 8, 8, 3, 2, 1, 1, 0.5, 0.5],
            [1, 1, 1, 1, 1, 1, 3, 8, 8, 8, 4, 4],
        ],
        dtype=np.float64,
    )
    gene_probs = gene_weights / gene_weights.sum(axis=1, keepdims=True)

    rows = []
    for domain_id, center in enumerate(centers):
        pts = rng.normal(loc=center, scale=0.6, size=(n_per_domain, 2))
        gene_idx = rng.choice(len(genes), size=n_per_domain, p=gene_probs[domain_id])
        if include_z:
            z = rng.normal(loc=float(domain_id), scale=0.15, size=n_per_domain)
        else:
            z = [None] * n_per_domain
        for i in range(n_per_domain):
            row = {
                "x": float(pts[i, 0]),
                "y": float(pts[i, 1]),
                "gene_id": genes[gene_idx[i]],
                "true_domain": domain_id,
            }
            if include_z:
                row["z"] = float(z[i])
            rows.append(row)

    return pd.DataFrame(rows)
