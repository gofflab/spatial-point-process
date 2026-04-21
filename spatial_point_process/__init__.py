"""Spatial point process prototype package."""

from spatial_point_process.cli import main, parse_args
from spatial_point_process.graph import build_knn_graph
from spatial_point_process.io import encode_inputs, load_points, save_outputs, summarize_programs
from spatial_point_process.model import MessagePassingLayer, PointProgramModel
from spatial_point_process.toy_data import generate_toy_dataset
from spatial_point_process.train import compute_loss, fit_model, set_seed
from spatial_point_process.types import FitResult, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, TrainConfig

__all__ = [
    "FitResult",
    "MessagePassingLayer",
    "OPTIONAL_COLUMNS",
    "PointProgramModel",
    "REQUIRED_COLUMNS",
    "TrainConfig",
    "build_knn_graph",
    "compute_loss",
    "encode_inputs",
    "fit_model",
    "generate_toy_dataset",
    "load_points",
    "main",
    "parse_args",
    "save_outputs",
    "set_seed",
    "summarize_programs",
]
