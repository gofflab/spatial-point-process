#!/usr/bin/env python3
import json
import os
import tempfile

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from spatial_point_program import generate_toy_dataset, load_points, TrainConfig, fit_model, save_outputs


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        toy_path = os.path.join(tmpdir, "toy.parquet")
        outdir = os.path.join(tmpdir, "out")
        df = generate_toy_dataset(n_per_domain=220, seed=7, include_z=False)
        df.to_parquet(toy_path, index=False)

        train_df = load_points(toy_path)
        cfg = TrainConfig(
            n_programs=3,
            hidden_dim=48,
            n_layers=2,
            k_neighbors=12,
            epochs=240,
            lr=1e-2,
            smoothness_weight=0.8,
            entropy_weight=0.02,
            seed=7,
            device="cpu",
        )
        _, result, artifacts = fit_model(train_df, cfg)
        save_outputs(outdir, train_df, result, artifacts, cfg)

        pred = artifacts["assignment_probs"].argmax(axis=1)
        ari = adjusted_rand_score(df["true_domain"].to_numpy(), pred)
        loss_drop = result.history["loss"][0] - result.history["loss"][-1]

        point_assignments = pd.read_csv(os.path.join(outdir, "point_assignments.csv"))
        program_summary = pd.read_csv(os.path.join(outdir, "program_gene_summary.csv"))
        with open(os.path.join(outdir, "fit_metadata.json")) as f:
            meta = json.load(f)

        print(json.dumps({
            "ari": float(ari),
            "loss_start": float(result.history["loss"][0]),
            "loss_end": float(result.history["loss"][-1]),
            "loss_drop": float(loss_drop),
            "n_output_rows": int(point_assignments.shape[0]),
            "n_program_rows": int(program_summary.shape[0]),
            "meta_n_points": int(meta["result"]["n_points"])
        }, indent=2))

        assert ari > 0.80
        assert loss_drop > 0.1
        assert point_assignments.shape[0] == df.shape[0]
        assert program_summary.shape[0] == 3
        assert meta["result"]["n_points"] == df.shape[0]


if __name__ == "__main__":
    main()
