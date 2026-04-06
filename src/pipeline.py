from __future__ import annotations

import argparse
import json
from typing import Any

from omegaconf import DictConfig, OmegaConf
from prefect import flow

from services.preprocessing_service.preprocessing_task import preprocessing_task
from utils.database import healthcheck_database_task


def cluster_faces_task(run_directives: DictConfig, prepared_inputs: Any) -> Any:
    raise NotImplementedError("Define `cluster_faces_task` later.")


def save_results_task(run_directives: DictConfig, clustered_faces: Any) -> None:
    raise NotImplementedError("Define `save_results_task` later.")


@flow(name="face-clustering-flow")
def run_face_clustering_flow(run_directives: dict[str, Any]) -> None:
    """Entrypoint flow for face clustering runs."""
    directives_cfg = OmegaConf.create(run_directives)
    if not isinstance(directives_cfg, DictConfig):
        raise ValueError("run_directives must be a mapping.")

    healthcheck_database_task(directives_cfg)
    images_df = preprocessing_task(directives_cfg)
    clustered_faces = cluster_faces_task(directives_cfg, images_df)
    save_results_task(directives_cfg, clustered_faces)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face clustering flow")
    parser.add_argument(
        "--run-directives-json",
        required=True,
        help="JSON object with mapped run directives.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_directives = json.loads(args.run_directives_json)
    if not isinstance(run_directives, dict):
        raise ValueError("--run-directives-json must be a JSON object.")

    run_face_clustering_flow.serve(
        "face-clustering-flow", parameters={"run_directives": run_directives}
    )
