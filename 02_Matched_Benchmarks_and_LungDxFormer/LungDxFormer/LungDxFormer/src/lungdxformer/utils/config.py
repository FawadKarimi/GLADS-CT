from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import yaml

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def parse_override_items(items: list[str]) -> dict[str, Any]:
    def parse_value(raw: str) -> Any:
        low = raw.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low == "none":
            return None
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            if raw.startswith("[") and raw.endswith("]"):
                vals = raw[1:-1].strip()
                if not vals:
                    return []
                return [parse_value(x.strip()) for x in vals.split(",")]
            return raw

    result: dict[str, Any] = {}
    for item in items:
        key, value = item.split("=", 1)
        value = parse_value(value)
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return result
