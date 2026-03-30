#!/usr/bin/env bash
export PREFECT_API_URL=http://localhost:4200/api
uv run prefect deploy src/pipeline.py:run_face_clustering_flow -n default
uv run prefect deployment run "face-clustering-flow/default" --params '{"config_path":"run_directives/default.yml"}'
