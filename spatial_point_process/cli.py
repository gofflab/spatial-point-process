"""Command line interface for spatial point program prototype."""
from __future__ import annotations

import argparse
import json
import os

from spatial_point_process.io import load_points, save_outputs
from spatial_point_process.toy_data import generate_toy_dataset
from spatial_point_process.train import fit_model
from spatial_point_process.types import TrainConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
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
    toy.add_argument("--variant", choices=["simple", "structured"], default="simple")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.command == "make-toy":
        df = generate_toy_dataset(
            n_per_domain=args.n_per_domain,
            seed=args.seed,
            include_z=args.include_z,
            variant=args.variant,
        )
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
    print(
        json.dumps(
            {
                "n_points": result.n_points,
                "n_genes": result.n_genes,
                "n_programs": result.n_programs,
                "final_loss": result.final_loss,
                "outdir": args.outdir,
            },
            indent=2,
        )
    )
