"""Training and fitting routines for spatial point programs."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from spatial_point_process.graph import build_knn_graph
from spatial_point_process.io import encode_inputs
from spatial_point_process.model import PointProgramModel
from spatial_point_process.types import FitResult, TrainConfig


def set_seed(seed: int) -> None:
    """Set numpy/torch RNG seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    gene_ids: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute objective and individual loss components."""
    pred_gene_probs = outputs["pred_gene_probs"].clamp_min(1e-9)
    nll = -torch.log(pred_gene_probs[torch.arange(gene_ids.numel(), device=gene_ids.device), gene_ids]).mean()
    coord_nll = -outputs["pred_coord_logprob"].mean()

    q = outputs["assignment_probs"]
    src, dst = edge_index
    smoothness = ((q[src] - q[dst]) ** 2).sum(dim=1).mean()
    entropy = -(q * torch.log(q.clamp_min(1e-9))).sum(dim=1).mean()

    loss = nll + 0.25 * coord_nll + cfg.smoothness_weight * smoothness + cfg.entropy_weight * entropy
    metrics = {
        "loss": float(loss.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "coord_nll": float(coord_nll.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }
    return loss, metrics


def fit_model(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[PointProgramModel, FitResult, Dict[str, np.ndarray]]:
    """Train model and return fitted module, fit metadata, and output artifacts."""
    set_seed(cfg.seed)
    coords_np, gene_ids_np, gene_categories, spatial_dims = encode_inputs(df)
    edges_np = build_knn_graph(coords_np, cfg.k_neighbors)

    if str(cfg.device).lower() == "cpu":
        torch.set_num_threads(1)
    device = torch.device(cfg.device)
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    gene_ids = torch.tensor(gene_ids_np, dtype=torch.long, device=device)
    edge_index = torch.tensor(edges_np, dtype=torch.long, device=device)

    model = PointProgramModel(
        n_genes=len(gene_categories),
        spatial_dims=spatial_dims,
        n_programs=cfg.n_programs,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"loss": [], "nll": [], "coord_nll": [], "smoothness": [], "entropy": []}
    for _epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        outputs = model(coords, gene_ids, edge_index, temperature=cfg.temperature)
        loss, metrics = compute_loss(outputs, gene_ids, edge_index, cfg)
        loss.backward()
        opt.step()
        for k, v in metrics.items():
            history[k].append(v)

    model.eval()
    with torch.no_grad():
        outputs = model(coords, gene_ids, edge_index, temperature=cfg.temperature)

    artifacts = {
        "coords": coords_np,
        "gene_ids": gene_ids_np,
        "edge_index": edges_np,
        "assignment_probs": outputs["assignment_probs"].cpu().numpy(),
        "assignment_logits": outputs["assignment_logits"].cpu().numpy(),
        "coord_evidence": outputs["coord_evidence"].cpu().numpy(),
        "program_gene_probs": outputs["program_gene_probs"].cpu().numpy(),
        "program_coord_means": model.program_coord_means.detach().cpu().numpy(),
        "program_coord_logvars": model.program_coord_logvars.detach().cpu().numpy(),
        "embeddings": outputs["embeddings"].cpu().numpy(),
    }
    result = FitResult(
        n_points=coords_np.shape[0],
        n_genes=len(gene_categories),
        n_programs=cfg.n_programs,
        final_loss=history["loss"][-1],
        gene_categories=gene_categories,
        spatial_dims=spatial_dims,
        history=history,
    )
    return model, result, artifacts
