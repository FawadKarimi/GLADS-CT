from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import os
import re
import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(f"Environment variable {name!r} is required by config but is not set")
            return os.environ[name]
        return _ENV_PATTERN.sub(repl, value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")

    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)

    return _expand_env(cfg)
