#!/usr/bin/env bash
export PREFECT_API_URL=http://localhost:4200/api

CONFIG_PATH=${CONFIG_PATH:-run_directives/default.yml}

RUN_DIRECTIVES_JSON=$(uv run python - <<'PY'
from omegaconf import OmegaConf
import json
import os

cfg = OmegaConf.load(os.getenv("CONFIG_PATH", "run_directives/default.yml"))
print(json.dumps(OmegaConf.to_container(cfg, resolve=True)))
PY
)

echo "Using configuration from $CONFIG_PATH"
echo "Run directives JSON: $RUN_DIRECTIVES_JSON"

if ! uv run prefect work-pool inspect "local_docker" >/dev/null 2>&1; then
	uv run prefect work-pool create "local_docker" --type process
fi

uv run python scripts/deploy_flow.py --config "$CONFIG_PATH"
uv run prefect deployment run "face-clustering-flow/default" --params "{\"run_directives\":$RUN_DIRECTIVES_JSON}"
