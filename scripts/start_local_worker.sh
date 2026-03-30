#!/usr/bin/env bash
export PREFECT_API_URL=http://localhost:4200/api
uv run prefect worker start --pool "local_docker"
