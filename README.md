# Spatial Point Program Tool

A minimal working prototype for segmentation-free latent spatial program discovery from point-level spatial transcriptomics data.

## Input
A `.parquet`, `.csv`, or `.tsv` file with columns:
- `x`
- `y`
- optional `z`
- `gene_id`

No nuclei masks, cell masks, or prior assignments are required.

## What it does
- reads raw molecule coordinates and gene identities
- builds a spatial k-nearest-neighbor graph over molecules
- learns latent spatial programs using a graph encoder plus probabilistic program-specific:
  - gene distributions
  - spatial Gaussian fields
- outputs soft and hard point-level program assignments

## Files
- `spatial_point_program.py` — backward-compatible CLI entrypoint
- `spatial_point_process/` — package modules (`io`, `graph`, `model`, `train`, `toy_data`, `cli`)
- `test_toy_run.py` — toy end-to-end test
- `.github/workflows/ci.yml` — CI workflow running toy test

## Quick start
Generate toy data:

```bash
python spatial_point_program.py make-toy --output toy.parquet
```

Train on toy data:

```bash
python spatial_point_program.py train \
  --input toy.parquet \
  --outdir toy_out \
  --n-programs 3 \
  --epochs 220 \
  --k-neighbors 12
```

Run the toy test:

```bash
python test_toy_run.py
```

## Outputs
- `fit_metadata.json`
- `point_assignments.csv`
- `program_gene_summary.csv`
- `embeddings.npy`
- `program_gene_probs.npy`
- `edge_index.npy`

## Notes
This is a prototype, not a production-scale billion-molecule implementation.
It is designed to be easy to inspect and extend.

A natural next step would be blockwise training, anchor/superpoint hierarchy, and minibatched distributed graph processing.
