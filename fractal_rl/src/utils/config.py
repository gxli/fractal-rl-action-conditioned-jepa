"""YAML configuration with optional recursive overlays and dot-path CLI overrides."""
from pathlib import Path
import copy
import yaml

DEFAULT = Path(__file__).resolve().parents[2] / 'config' / 'default.yaml'


def merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def load_config(path=None, overlays=(), overrides=()):
    with DEFAULT.open() as f:
        cfg = yaml.safe_load(f)
    for name in ([path] if path else []) + list(overlays):
        with Path(name).open() as f:
            merge(cfg, yaml.safe_load(f) or {})
    for entry in overrides:
        key, sep, raw = entry.partition('=')
        if not sep:
            raise ValueError(f'Override must be section.key=value: {entry}')
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(key)
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(key)
        target[parts[-1]] = yaml.safe_load(raw)
    return cfg


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
