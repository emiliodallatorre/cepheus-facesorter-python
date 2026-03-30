from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_yaml_config(config_path: str) -> DictConfig:
	"""Load a YAML file with OmegaConf and return a dot-access config object."""
	path = Path(config_path)
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {path}")

	config = OmegaConf.load(path)
	if not isinstance(config, DictConfig):
		raise ValueError("YAML root must be a mapping to support dot access.")

	return config
