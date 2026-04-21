"""Toy synthetic spatial transcriptomics data generation."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def _generate_simple_points(rng: np.random.Generator, n_per_domain: int) -> List[np.ndarray]:
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.2]], dtype=np.float32)
    return [rng.normal(loc=center, scale=0.6, size=(n_per_domain, 2)) for center in centers]


def _generate_structured_points(rng: np.random.Generator, n_per_domain: int) -> List[np.ndarray]:
    def arc_points(center_x: float, center_y: float, radius: float, theta_start: float, theta_stop: float) -> np.ndarray:
        theta = rng.uniform(theta_start, theta_stop, size=n_per_domain)
        radial_noise = rng.normal(loc=0.0, scale=0.18, size=n_per_domain)
        x = center_x + (radius + radial_noise) * np.cos(theta) + rng.normal(scale=0.10, size=n_per_domain)
        y = center_y + 0.7 * (radius + radial_noise) * np.sin(theta) + rng.normal(scale=0.14, size=n_per_domain)
        return np.column_stack([x, y])

    def wave_band() -> np.ndarray:
        x = rng.uniform(-1.5, 6.8, size=n_per_domain)
        y = 0.8 * np.sin(1.2 * x) + 0.33 * x - 3.4 + rng.normal(scale=0.22, size=n_per_domain)
        return np.column_stack([x, y])

    def ring_points(center_x: float, center_y: float, radius: float) -> np.ndarray:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_domain)
        radial_noise = rng.normal(loc=0.0, scale=0.16, size=n_per_domain)
        x = center_x + (radius + radial_noise) * np.cos(theta)
        y = center_y + (radius * 0.8 + radial_noise) * np.sin(theta)
        return np.column_stack([x, y])

    def bifurcated_column() -> np.ndarray:
        branch = rng.integers(0, 2, size=n_per_domain)
        x_center = np.where(branch == 0, -4.4, -2.3)
        y_center = np.where(branch == 0, 4.9, 7.2)
        x = x_center + rng.normal(scale=0.35, size=n_per_domain)
        y = y_center + rng.normal(scale=0.55, size=n_per_domain)
        return np.column_stack([x, y])

    return [
        arc_points(center_x=-4.2, center_y=0.6, radius=2.5, theta_start=0.15 * np.pi, theta_stop=1.18 * np.pi),
        wave_band(),
        ring_points(center_x=3.6, center_y=3.6, radius=2.1),
        bifurcated_column(),
    ]


def _make_gene_probabilities(variant: str, genes: List[str]) -> np.ndarray:
    if variant == "simple":
        gene_weights = np.array(
            [
                [8, 8, 8, 3, 2, 1, 1, 1, 1, 1, 0.5, 0.5],
                [1, 1, 1, 8, 8, 8, 3, 2, 1, 1, 0.5, 0.5],
                [1, 1, 1, 1, 1, 1, 3, 8, 8, 8, 4, 4],
            ],
            dtype=np.float64,
        )
    elif variant == "structured":
        gene_weights = np.array(
            [
                [10, 10, 7, 2, 2, 1, 1, 1, 1, 1, 0.8, 0.8, 4, 2, 1, 1],
                [1, 1, 2, 10, 10, 7, 2, 1, 1, 1, 0.8, 0.8, 1, 4, 2, 1],
                [1, 1, 1, 1, 2, 2, 10, 10, 8, 2, 0.8, 0.8, 1, 1, 4, 2],
                [1, 1, 1, 1, 1, 1, 2, 3, 2, 10, 8, 8, 2, 1, 1, 4],
            ],
            dtype=np.float64,
        )
    else:  # pragma: no cover - guarded by generate_toy_dataset validation
        raise ValueError(f"Unsupported toy dataset variant: {variant}")

    if gene_weights.shape[1] != len(genes):
        raise ValueError("Gene weight table does not match the number of genes")
    return gene_weights / gene_weights.sum(axis=1, keepdims=True)


def _make_z_values(
    rng: np.random.Generator,
    pts: np.ndarray,
    domain_id: int,
    variant: str,
) -> np.ndarray:
    if variant == "simple":
        return rng.normal(loc=float(domain_id), scale=0.15, size=pts.shape[0])
    radial = np.sqrt((pts[:, 0] - pts[:, 0].mean()) ** 2 + (pts[:, 1] - pts[:, 1].mean()) ** 2)
    return domain_id + 0.18 * radial + rng.normal(scale=0.12, size=pts.shape[0])


def generate_toy_dataset(
    n_per_domain: int = 250,
    seed: int = 0,
    include_z: bool = False,
    variant: str = "simple",
) -> pd.DataFrame:
    """Create a synthetic dataset with point-level spatial structure and marker genes."""
    rng = np.random.default_rng(seed)
    variant = str(variant).lower()
    point_generators: Dict[str, Callable[[np.random.Generator, int], List[np.ndarray]]] = {
        "simple": _generate_simple_points,
        "structured": _generate_structured_points,
    }
    if variant not in point_generators:
        raise ValueError(f"Unsupported toy dataset variant '{variant}'. Use one of: {sorted(point_generators)}")

    genes = [f"gene_{i}" for i in range(12 if variant == "simple" else 16)]
    point_sets = point_generators[variant](rng, n_per_domain)
    gene_probs = _make_gene_probabilities(variant, genes)

    rows = []
    for domain_id, pts in enumerate(point_sets):
        gene_idx = rng.choice(len(genes), size=n_per_domain, p=gene_probs[domain_id])
        if include_z:
            z_values = _make_z_values(rng, pts, domain_id, variant)
        else:
            z_values = [None] * n_per_domain
        for i in range(n_per_domain):
            row = {
                "x": float(pts[i, 0]),
                "y": float(pts[i, 1]),
                "gene_id": genes[gene_idx[i]],
                "true_domain": domain_id,
            }
            if include_z:
                row["z"] = float(z_values[i])
            rows.append(row)

    return pd.DataFrame(rows)
