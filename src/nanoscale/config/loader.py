"""YAML loading, deep-merge and JSON-Schema export for NanoScale configs.

Resolution order (later wins):

1. the tier preset named by ``preset:`` in the YAML (or the ``tier`` argument),
2. the YAML body,
3. dotted-path CLI overrides (``--set train.max_steps=10``).

This keeps every YAML file short: it only records the *deltas* from a preset, which is
what makes an ablation config readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nanoscale.config.presets import get_preset
from nanoscale.config.schemas import ALL_CONFIG_MODELS, ExperimentConfig

__all__ = [
    "apply_overrides",
    "deep_merge",
    "export_json_schemas",
    "load_experiment",
    "parse_override",
    "save_experiment",
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse a ``dotted.path=value`` override into a key path and a typed value.

    Values go through the YAML scalar parser, so ``true``, ``3``, ``null`` and
    ``[1, 2]`` all get their natural types. YAML 1.1 does not recognise exponent
    literals without a decimal point (``1e-4``), which is exactly how learning rates
    are usually typed, so string results get a second pass through ``int``/``float``.
    """
    if "=" not in item:
        raise ValueError(f"Override {item!r} is not of the form key.path=value.")
    path, _, raw = item.partition("=")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    if isinstance(value, str):
        value = _coerce_numeric(value)
    return path.split("."), value


def _coerce_numeric(text: str) -> int | float | str:
    """Return ``text`` as an int or float when it parses as one, else unchanged."""
    stripped = text.strip()
    if not stripped:
        return text
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply a list of ``dotted.path=value`` overrides to a nested dict."""
    out: dict[str, Any] = json.loads(json.dumps(data, default=str))
    for item in overrides:
        keys, value = parse_override(item)
        cursor: dict[str, Any] = out
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = value
    return out


def load_experiment(
    path: str | Path | None = None,
    *,
    tier: str | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a preset, a YAML file, and CLI overrides.

    Args:
        path: Optional YAML file. It may contain a top-level ``preset:`` key naming the
            tier it extends.
        tier: Preset tier to start from. Defaults to the YAML's ``preset``, else
            ``"nano"``.
        overrides: ``dotted.path=value`` strings applied last.

    Returns:
        A validated, frozen experiment config.
    """
    body: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(f"Config file {path} must contain a mapping at the top level.")
        body = raw
    preset_name = tier or str(body.pop("preset", "nano"))
    base = get_preset(preset_name).dump_inputs(mode="json")
    merged = deep_merge(base, body)
    if overrides:
        merged = apply_overrides(merged, overrides)
    return ExperimentConfig.model_validate(merged)


def save_experiment(cfg: ExperimentConfig, path: str | Path) -> Path:
    """Write a fully-resolved experiment config to YAML and return the path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.dump_inputs(mode="json")
    dest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest


def export_json_schemas(out_dir: str | Path) -> list[Path]:
    """Export the JSON Schema of every config model; returns the written paths."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in ALL_CONFIG_MODELS:
        dest = root / f"{model.__name__}.json"
        dest.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        written.append(dest)
    return written
