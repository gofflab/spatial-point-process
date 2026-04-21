"""Core dataclasses and constants for spatial point process training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

REQUIRED_COLUMNS = {"x", "y", "gene_id"}
OPTIONAL_COLUMNS = {"z"}


@dataclass
class TrainConfig:
    """Configuration for model fitting."""

    n_programs: int = 6
    hidden_dim: int = 64
    n_layers: int = 2
    k_neighbors: int = 12
    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 1e-4
    smoothness_weight: float = 0.8
    entropy_weight: float = 0.02
    temperature: float = 1.0
    seed: int = 0
    device: str = "cpu"


@dataclass
class FitResult:
    """Summary of a completed fit."""

    n_points: int
    n_genes: int
    n_programs: int
    final_loss: float
    gene_categories: List[str]
    spatial_dims: int
    history: Dict[str, List[float]]
