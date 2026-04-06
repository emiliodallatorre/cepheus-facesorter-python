from pathlib import Path
import sys

from prefect.docker import DockerImage

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from pipeline import run_face_clustering_flow


if __name__ == "__main__":
    run_face_clustering_flow.deploy(
        name="default",
        work_pool_name="local_docker",
        image=DockerImage(
            name="cepheus-facesorter-python",
            tag="latest",
            dockerfile="Dockerfile",
        ),
        push=False,
        job_variables={
            "image_pull_policy": "Never",
            "networks": ["facesorter-network"],
        },
        parameters={"config_path": "run_directives/default.yml"},
    )