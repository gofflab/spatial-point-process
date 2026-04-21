"""Input/output and preprocessing helpers."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from spatial_point_process.types import FitResult, REQUIRED_COLUMNS, TrainConfig


def load_points(path: str) -> pd.DataFrame:
    """Load and validate point-level transcript coordinates from parquet/csv/tsv."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # pragma: no cover - exact backend error depends on env
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
    """Standardize coordinates and encode gene identifiers as integer categories."""
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


def summarize_programs(program_gene_probs: np.ndarray, gene_categories: List[str], top_n: int = 10) -> pd.DataFrame:
    """Build per-program top gene summary table."""
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


def save_outputs(
    outdir: str,
    df: pd.DataFrame,
    result: FitResult,
    artifacts: Dict[str, np.ndarray],
    cfg: TrainConfig,
) -> None:
    """Persist fit metadata, point assignments, and model artifacts."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "fit_metadata.json"), "w") as f:
        json.dump({"result": asdict(result), "config": asdict(cfg)}, f, indent=2)

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
