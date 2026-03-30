#!/usr/bin/env bash
export PREFECT_API_URL=http://localhost:4200/api

uv run python scripts/deploy_flow.py
uv run prefect deployment run "face-clustering-flow/default" --params '{"config_path":"run_directives/default.yml"}'
