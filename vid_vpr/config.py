from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SECTIONS = {
    "experiment",
    "model",
    "data",
    "training",
    "evaluation",
}


def _set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"unknown config key: {dotted_key}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key: {dotted_key}")
    current = node[keys[-1]]
    if isinstance(current, float) and isinstance(value, str):
        value = float(value)
    elif isinstance(current, int) and not isinstance(current, bool) and isinstance(value, str):
        value = int(value)
    node[keys[-1]] = value


def load_config(config_path, overrides=()):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    unknown = set(config) - ALLOWED_SECTIONS
    if unknown:
        raise KeyError(f"unknown config sections: {sorted(unknown)}")
    config = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value syntax: {override}")
        key, raw_value = override.split("=", 1)
        _set_nested(config, key, yaml.safe_load(raw_value))
    return config


def project_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_sections(config, *sections):
    missing = [section for section in sections if section not in config]
    if missing:
        raise KeyError(f"missing config sections: {missing}")
