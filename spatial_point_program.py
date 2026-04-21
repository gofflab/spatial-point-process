#!/usr/bin/env python3
"""Minimal segmentation-free spatial point program discovery tool.

Input: parquet/csv table with columns x, y, optional z, and gene_id.
Output: point-level latent program assignments and program-by-gene profiles.

This is a practical prototype inspired by graph-based point handling and
probabilistic latent spatial programs. It does not require nuclei or masks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


REQUIRED_COLUMNS = {"x", "y", "gene_id"}
OPTIONAL_COLUMNS = {"z"}


@dataclass
class TrainConfig:
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
    n_points: int
    n_genes: int
    n_programs: int
    final_loss: float
    gene_categories: List[str]
    spatial_dims: int
    history: Dict[str, List[float]]


class MessagePassingLayer(nn.Module):
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

    def forward(self, coords: torch.Tensor, gene_ids: torch.Tensor, edge_index: torch.Tensor, temperature: float = 1.0):
        h = self.gene_emb(gene_ids) + self.coord_mlp(coords)
        for layer in self.layers:
            h = h + layer(h, edge_index)

        gene_log_probs = F.log_softmax(self.program_gene_logits, dim=1)
        gene_evidence = gene_log_probs[:, gene_ids].T

        coord_means = self.program_coord_means
        coord_logvars = self.program_coord_logvars.clamp(min=-4.0, max=4.0)
        diff = coords[:, None, :] - coord_means[None, :, :]
        coord_evidence = -0.5 * (((diff ** 2) / coord_logvars.exp()[None, :, :]) + coord_logvars[None, :, :] + math.log(2 * math.pi)).sum(dim=2)

        logits = (self.assignment_head(h) + gene_evidence + coord_evidence + self.program_prior_logits[None, :]) / temperature
        q = F.softmax(logits, dim=1)
        gene_probs = gene_log_probs.exp()
        pred_gene_probs = q @ gene_probs
        pred_coord_logprob = torch.logsumexp(
            torch.log(q.clamp_min(1e-9)) + coord_evidence, dim=1
        )
        return {
            "embeddings": h,
            "assignment_logits": logits,
            "assignment_probs": q,
            "program_gene_probs": gene_probs,
            "pred_gene_probs": pred_gene_probs,
            "coord_evidence": coord_evidence,
            "pred_coord_logprob": pred_coord_logprob,
        }



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)



def load_points(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read parquet file '{path}'. Install pyarrow or fastparquet. Original error: {e}"
            )
    elif ext in {".csv", ".tsv", ".txt"}:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .parquet, .csv, or .tsv")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(missing)}")

    cols = [c for c in ["x", "y", "z", "gene_id"] if c in df.columns]
    out = df[cols].copy()
    out = out.dropna(subset=["x", "y", "gene_id"])
    return out



def encode_inputs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    dims = 3 if "z" in df.columns else 2
    coord_cols = ["x", "y"] + (["z"] if dims == 3 else [])
    coords = df[coord_cols].to_numpy(dtype=np.float32)
    means = coords.mean(axis=0, keepdims=True)
    stds = coords.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    coords = (coords - means) / stds

    cats = pd.Categorical(df["gene_id"].astype(str))
    gene_ids = cats.codes.astype(np.int64)
    gene_categories = [str(x) for x in cats.categories]
    return coords, gene_ids, gene_categories, dims



def build_knn_graph(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
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



def compute_loss(outputs: Dict[str, torch.Tensor], gene_ids: torch.Tensor, edge_index: torch.Tensor, cfg: TrainConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    for epoch in range(cfg.epochs):
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
        "program_gene_probs": outputs["program_gene_probs"].cpu().numpy(),
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



def summarize_programs(program_gene_probs: np.ndarray, gene_categories: List[str], top_n: int = 10) -> pd.DataFrame:
    rows = []
    for k in range(program_gene_probs.shape[0]):
        order = np.argsort(program_gene_probs[k])[::-1][:top_n]
        rows.append(
            {
                "program": k,
                "top_genes": ";".join(gene_categories[i] for i in order),
                "top_gene_probs": ";".join(f"{program_gene_probs[k, i]:.4f}" for i in order),
            }
        )
    return pd.DataFrame(rows)



def save_outputs(outdir: str, df: pd.DataFrame, result: FitResult, artifacts: Dict[str, np.ndarray], cfg: TrainConfig) -> None:
    os.makedirs(outdir, exist_ok=True)
    meta = {
        "result": asdict(result),
        "config": asdict(cfg),
    }
    with open(os.path.join(outdir, "fit_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    point_df = df.copy().reset_index(drop=True)
    q = artifacts["assignment_probs"]
    point_df["program"] = q.argmax(axis=1)
    point_df["program_confidence"] = q.max(axis=1)
    for k in range(q.shape[1]):
        point_df[f"program_prob_{k}"] = q[:, k]
    point_df.to_csv(os.path.join(outdir, "point_assignments.csv"), index=False)

    summary = summarize_programs(artifacts["program_gene_probs"], result.gene_categories)
    summary.to_csv(os.path.join(outdir, "program_gene_summary.csv"), index=False)

    np.save(os.path.join(outdir, "embeddings.npy"), artifacts["embeddings"])
    np.save(os.path.join(outdir, "program_gene_probs.npy"), artifacts["program_gene_probs"])
    np.save(os.path.join(outdir, "edge_index.npy"), artifacts["edge_index"])



def generate_toy_dataset(n_per_domain: int = 250, seed: int = 0, include_z: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i}" for i in range(12)]
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.2]], dtype=np.float32)
    gene_weights = np.array([
        [8, 8, 8, 3, 2, 1, 1, 1, 1, 1, 0.5, 0.5],
        [1, 1, 1, 8, 8, 8, 3, 2, 1, 1, 0.5, 0.5],
        [1, 1, 1, 1, 1, 1, 3, 8, 8, 8, 4, 4],
    ], dtype=np.float64)
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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn latent spatial programs from point-level transcript coordinates.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train model on parquet/csv input")
    train.add_argument("--input", required=True, help="Path to parquet/csv/tsv file with x,y,[z],gene_id columns")
    train.add_argument("--outdir", required=True, help="Directory to write outputs")
    train.add_argument("--n-programs", type=int, default=6)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--n-layers", type=int, default=2)
    train.add_argument("--k-neighbors", type=int, default=12)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--lr", type=float, default=1e-2)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--smoothness-weight", type=float, default=0.8)
    train.add_argument("--entropy-weight", type=float, default=0.02)
    train.add_argument("--temperature", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")

    toy = sub.add_parser("make-toy", help="Generate a toy dataset")
    toy.add_argument("--output", required=True)
    toy.add_argument("--n-per-domain", type=int, default=250)
    toy.add_argument("--seed", type=int, default=0)
    toy.add_argument("--include-z", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if args.command == "make-toy":
        df = generate_toy_dataset(n_per_domain=args.n_per_domain, seed=args.seed, include_z=args.include_z)
        ext = os.path.splitext(args.output)[1].lower()
        if ext == ".parquet":
            df.to_parquet(args.output, index=False)
        elif ext == ".csv":
            df.to_csv(args.output, index=False)
        else:
            raise ValueError("Toy output must end in .parquet or .csv")
        print(f"Wrote toy dataset to {args.output} with shape {df.shape}")
        return

    cfg = TrainConfig(
        n_programs=args.n_programs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        k_neighbors=args.k_neighbors,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoothness_weight=args.smoothness_weight,
        entropy_weight=args.entropy_weight,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
    )
    df = load_points(args.input)
    _, result, artifacts = fit_model(df, cfg)
    save_outputs(args.outdir, df, result, artifacts, cfg)
    print(json.dumps({
        "n_points": result.n_points,
        "n_genes": result.n_genes,
        "n_programs": result.n_programs,
        "final_loss": result.final_loss,
        "outdir": args.outdir,
    }, indent=2))


if __name__ == "__main__":
    main()
