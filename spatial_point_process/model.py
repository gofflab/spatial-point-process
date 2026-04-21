"""PyTorch model definitions for latent spatial program discovery."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """Simple mean-aggregation message passing layer implemented in plain PyTorch."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])
        deg = torch.bincount(dst, minlength=h.shape[0]).clamp_min(1).to(h.dtype).unsqueeze(1)
        agg = agg / deg
        return self.mlp(torch.cat([h, agg], dim=1))


class PointProgramModel(nn.Module):
    """Graph-augmented latent program model over molecule points."""

    def __init__(self, n_genes: int, spatial_dims: int, n_programs: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.gene_emb = nn.Embedding(n_genes, hidden_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(spatial_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim) for _ in range(n_layers)])
        self.assignment_head = nn.Linear(hidden_dim, n_programs)
        self.program_gene_logits = nn.Parameter(torch.randn(n_programs, n_genes) * 0.01)
        self.program_prior_logits = nn.Parameter(torch.zeros(n_programs))
        self.program_coord_means = nn.Parameter(torch.randn(n_programs, spatial_dims) * 0.1)
        self.program_coord_logvars = nn.Parameter(torch.zeros(n_programs, spatial_dims))

    def forward(
        self,
        coords: torch.Tensor,
        gene_ids: torch.Tensor,
        edge_index: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        h = self.gene_emb(gene_ids) + self.coord_mlp(coords)
        for layer in self.layers:
            h = h + layer(h, edge_index)

        gene_log_probs = F.log_softmax(self.program_gene_logits, dim=1)
        gene_evidence = gene_log_probs[:, gene_ids].T

        coord_means = self.program_coord_means
        coord_logvars = self.program_coord_logvars.clamp(min=-4.0, max=4.0)
        diff = coords[:, None, :] - coord_means[None, :, :]
        coord_evidence = -0.5 * (
            ((diff**2) / coord_logvars.exp()[None, :, :])
            + coord_logvars[None, :, :]
            + math.log(2 * math.pi)
        ).sum(dim=2)

        logits = (self.assignment_head(h) + gene_evidence + coord_evidence + self.program_prior_logits[None, :]) / temperature
        q = F.softmax(logits, dim=1)
        gene_probs = gene_log_probs.exp()
        pred_gene_probs = q @ gene_probs
        pred_coord_logprob = torch.logsumexp(torch.log(q.clamp_min(1e-9)) + coord_evidence, dim=1)

        return {
            "embeddings": h,
            "assignment_logits": logits,
            "assignment_probs": q,
            "program_gene_probs": gene_probs,
            "pred_gene_probs": pred_gene_probs,
            "coord_evidence": coord_evidence,
            "pred_coord_logprob": pred_coord_logprob,
        }
