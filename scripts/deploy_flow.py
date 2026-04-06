from pathlib import Path
import sys
import argparse
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from pipeline import run_face_clustering_flow
from utils.config import load_yaml_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy face clustering flow")
    parser.add_argument(
        "--config",
        default="run_directives/default.yml",
        help="Path to YAML file containing run directives.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_directives_cfg = load_yaml_config(args.config)
    run_directives = OmegaConf.to_container(run_directives_cfg, resolve=True)
    if not isinstance(run_directives, dict):
        raise ValueError("Config root must be a mapping.")

    deployment_flow: Any = run_face_clustering_flow.from_source(
        source=str(ROOT), entrypoint="src/pipeline.py:run_face_clustering_flow"
    )

    deployment_flow.deploy(
        name="default",
        work_pool_name="local_docker",
        parameters={"run_directives": run_directives},
    )