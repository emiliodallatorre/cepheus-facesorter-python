from __future__ import annotations

import argparse
from typing import Any

from omegaconf import DictConfig
from prefect import flow

from services.preprocessing_service.preprocessing_task import preprocessing_task
from utils.config import load_yaml_config
from utils.database import healthcheck_database_task


def cluster_faces_task(run_directives: DictConfig, prepared_inputs: Any) -> Any:
    raise NotImplementedError("Define `cluster_faces_task` later.")


def save_results_task(run_directives: DictConfig, clustered_faces: Any) -> None:
    raise NotImplementedError("Define `save_results_task` later.")


@flow(name="face-clustering-flow")
def run_face_clustering_flow(config_path: str) -> None:
    """Entrypoint flow for face clustering runs."""
    run_directives = load_yaml_config(config_path)

    healthcheck_database_task(run_directives)
    images_df = preprocessing_task(run_directives)
    clustered_faces = cluster_faces_task(run_directives, images_df)
    save_results_task(run_directives, clustered_faces)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face clustering flow")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML file with run directives (input_path, output_path, etc.)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_face_clustering_flow.serve(
        "face-clustering-flow", parameters={"config_path": args.config}
    )
