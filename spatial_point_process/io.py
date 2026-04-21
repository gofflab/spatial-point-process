"""Input/output and preprocessing helpers."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

from spatial_point_process.types import FitResult, REQUIRED_COLUMNS, TrainConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def build_training_history_table(history: Dict[str, List[float]]) -> pd.DataFrame:
    """Convert per-epoch metric history into a tabular diagnostics dataframe."""
    return pd.DataFrame({"epoch": np.arange(1, len(history["loss"]) + 1), **history})


def summarize_diagnostics(
    assignment_probs: np.ndarray,
    edge_index: np.ndarray,
    n_programs: int,
) -> Dict[str, Any]:
    """Summarize graph structure and assignment confidence for quick run inspection."""
    hard_assignments = assignment_probs.argmax(axis=1)
    confidence = assignment_probs.max(axis=1)
    entropy = -(assignment_probs * np.log(np.clip(assignment_probs, 1e-9, None))).sum(axis=1)
    src, dst = edge_index
    out_degree = np.bincount(src, minlength=assignment_probs.shape[0])
    in_degree = np.bincount(dst, minlength=assignment_probs.shape[0])
    program_counts = np.bincount(hard_assignments, minlength=n_programs)

    per_program = []
    for program_id in range(n_programs):
        mask = hard_assignments == program_id
        per_program.append(
            {
                "program": int(program_id),
                "n_points": int(program_counts[program_id]),
                "fraction_points": float(program_counts[program_id] / assignment_probs.shape[0]),
                "mean_confidence": float(confidence[mask].mean()) if np.any(mask) else 0.0,
                "mean_assignment_entropy": float(entropy[mask].mean()) if np.any(mask) else 0.0,
            }
        )

    return {
        "graph": {
            "n_edges": int(edge_index.shape[1]),
            "mean_out_degree": float(out_degree.mean()),
            "max_out_degree": int(out_degree.max()),
            "mean_in_degree": float(in_degree.mean()),
            "max_in_degree": int(in_degree.max()),
        },
        "assignments": {
            "mean_confidence": float(confidence.mean()),
            "median_confidence": float(np.median(confidence)),
            "min_confidence": float(confidence.min()),
            "max_confidence": float(confidence.max()),
            "mean_entropy": float(entropy.mean()),
            "median_entropy": float(np.median(entropy)),
            "max_entropy": float(entropy.max()),
        },
        "programs": per_program,
    }


def save_training_plot(history_df: pd.DataFrame, path: str) -> None:
    """Render a compact training metrics plot."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(history_df["epoch"], history_df["loss"], label="loss", linewidth=2.0)
    axes[0].plot(history_df["epoch"], history_df["nll"], label="nll", linewidth=1.5)
    axes[0].plot(history_df["epoch"], history_df["coord_nll"], label="coord_nll", linewidth=1.5)
    axes[0].set_ylabel("Objective")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    axes[1].plot(history_df["epoch"], history_df["smoothness"], label="smoothness", linewidth=1.5)
    axes[1].plot(history_df["epoch"], history_df["entropy"], label="entropy", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Regularizers")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_assignment_scatter_plot(point_df: pd.DataFrame, path: str) -> None:
    """Render a 2D scatter plot of point assignments and confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = point_df["x"].to_numpy()
    y = point_df["y"].to_numpy()
    program = point_df["program"].to_numpy()
    confidence = point_df["program_confidence"].to_numpy()

    scatter = axes[0].scatter(x, y, c=program, s=12, cmap="tab10", alpha=0.85, linewidths=0)
    axes[0].set_title("Predicted Program")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)

    conf = axes[1].scatter(x, y, c=confidence, s=12, cmap="viridis", alpha=0.85, linewidths=0, vmin=0.0, vmax=1.0)
    axes[1].set_title("Assignment Confidence")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(conf, ax=axes[1], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_spatial_latent_plot(
    point_df: pd.DataFrame,
    assignment_probs: np.ndarray,
    program_coord_means: np.ndarray,
    program_coord_logvars: np.ndarray,
    path: str,
) -> None:
    """Render learned spatial Gaussian fields over the observed 2D coordinate space."""
    if not {"x", "y"}.issubset(point_df.columns):
        return
    if program_coord_means.shape[1] != 2:
        return

    x = point_df["x"].to_numpy()
    y = point_df["y"].to_numpy()
    n_programs = assignment_probs.shape[1]
    ncols = min(3, n_programs)
    nrows = int(np.ceil(n_programs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows), squeeze=False)

    x_pad = max((x.max() - x.min()) * 0.08, 1e-3)
    y_pad = max((y.max() - y.min()) * 0.08, 1e-3)
    grid_x = np.linspace(x.min() - x_pad, x.max() + x_pad, 180)
    grid_y = np.linspace(y.min() - y_pad, y.max() + y_pad, 180)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    grid_points = np.stack([mesh_x, mesh_y], axis=-1)

    means = np.asarray(program_coord_means, dtype=np.float64)
    logvars = np.clip(np.asarray(program_coord_logvars, dtype=np.float64), -4.0, 4.0)
    vars_ = np.exp(logvars)
    point_program = assignment_probs.argmax(axis=1)

    for program_id, axis in enumerate(axes.flat):
        if program_id >= n_programs:
            axis.axis("off")
            continue

        diff = grid_points - means[program_id][None, None, :]
        log_density = -0.5 * (
            (diff[..., 0] ** 2) / vars_[program_id, 0]
            + (diff[..., 1] ** 2) / vars_[program_id, 1]
            + logvars[program_id, 0]
            + logvars[program_id, 1]
            + 2.0 * np.log(2.0 * np.pi)
        )
        density = np.exp(log_density - log_density.max())

        image = axis.imshow(
            density,
            origin="lower",
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
            cmap="magma",
            aspect="equal",
            alpha=0.92,
        )
        axis.scatter(x, y, c="white", s=5, alpha=0.14, linewidths=0)
        member_mask = point_program == program_id
        axis.scatter(x[member_mask], y[member_mask], c="cyan", s=8, alpha=0.45, linewidths=0)
        axis.scatter(
            means[program_id, 0],
            means[program_id, 1],
            c="lime",
            marker="x",
            s=90,
            linewidths=2.0,
        )
        axis.set_title(f"Program {program_id}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        std_x = float(np.sqrt(vars_[program_id, 0]))
        std_y = float(np.sqrt(vars_[program_id, 1]))
        axis.text(
            0.02,
            0.02,
            f"$\\mu$=({means[program_id, 0]:.2f}, {means[program_id, 1]:.2f})\n$\\sigma$=({std_x:.2f}, {std_y:.2f})",
            transform=axis.transAxes,
            fontsize=8,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2.5},
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Learned Spatial Latent Fields", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_program_gene_heatmap(
    program_gene_probs: np.ndarray,
    gene_categories: List[str],
    path: str,
) -> None:
    """Render a heatmap of learned program-gene probabilities."""
    fig_width = max(7.5, 0.38 * len(gene_categories))
    fig_height = max(3.5, 0.75 * program_gene_probs.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(program_gene_probs, aspect="auto", cmap="cividis")
    ax.set_title("Learned Program-Gene Distributions")
    ax.set_xlabel("Gene")
    ax.set_ylabel("Program")
    ax.set_xticks(np.arange(len(gene_categories)))
    ax.set_xticklabels(gene_categories, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(program_gene_probs.shape[0]))
    ax.set_yticklabels([f"Program {i}" for i in range(program_gene_probs.shape[0])])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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

    diagnostics = summarize_diagnostics(q, artifacts["edge_index"], result.n_programs)
    with open(os.path.join(outdir, "diagnostics_summary.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    history_df = build_training_history_table(result.history)
    history_df.to_csv(os.path.join(outdir, "training_history.csv"), index=False)

    save_training_plot(history_df, os.path.join(outdir, "training_metrics.png"))
    save_program_gene_heatmap(
        artifacts["program_gene_probs"],
        result.gene_categories,
        os.path.join(outdir, "program_gene_heatmap.png"),
    )
    if {"x", "y"}.issubset(point_df.columns):
        save_assignment_scatter_plot(point_df, os.path.join(outdir, "assignment_scatter.png"))
        save_spatial_latent_plot(
            point_df,
            q,
            artifacts["program_coord_means"],
            artifacts["program_coord_logvars"],
            os.path.join(outdir, "spatial_latent_fields.png"),
        )

    np.save(os.path.join(outdir, "embeddings.npy"), artifacts["embeddings"])
    np.save(os.path.join(outdir, "program_gene_probs.npy"), artifacts["program_gene_probs"])
    np.save(os.path.join(outdir, "edge_index.npy"), artifacts["edge_index"])
