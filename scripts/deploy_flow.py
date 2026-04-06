from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from pipeline import run_face_clustering_flow


if __name__ == "__main__":
    run_face_clustering_flow.from_source(
        source=str(ROOT), entrypoint="src/pipeline.py:run_face_clustering_flow"
    ).deploy(
        name="default",
        work_pool_name="local_docker",
        job_variables={
            "working_dir": str(ROOT),
        },
        parameters={"config_path": "run_directives/default.yml"},
    )