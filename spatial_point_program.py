#!/usr/bin/env python3
"""Backward-compatible entrypoint and API re-export.

The implementation now lives under the ``spatial_point_process`` package.
"""

from spatial_point_process import (  # noqa: F401
    FitResult,
    MessagePassingLayer,
    OPTIONAL_COLUMNS,
    PointProgramModel,
    REQUIRED_COLUMNS,
    TrainConfig,
    build_knn_graph,
    compute_loss,
    encode_inputs,
    fit_model,
    generate_toy_dataset,
    load_points,
    main,
    parse_args,
    save_outputs,
    set_seed,
    summarize_programs,
)


if __name__ == "__main__":
    main()
