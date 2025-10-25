"""Utilities for reading configuration paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

REQUIRED_KEYS = {
    "images_dir",
    "colmap_model_dir",
    "sam_ckpt",
    "masks_dir",
    "labels_dir",
    "outputs_dir",
    "device",
}

_CREATABLE_DIR_KEYS = {"outputs_dir", "masks_dir", "labels_dir"}


def load_config(config_path: str | Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load the YAML config file and validate required keys."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    missing = REQUIRED_KEYS.difference(data)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise KeyError(f"Config missing required keys: {missing_csv}")

    data["_config_dir"] = path.parent
    return data


def get_images_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "images_dir")


def get_colmap_model_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "colmap_model_dir")


def get_sam_checkpoint(config: Mapping[str, Any]) -> Path:
    path = _resolve_path(config, "sam_ckpt")
    if not path.is_file():
        raise FileNotFoundError(f"SAM checkpoint file not found: {path}")
    return path


def get_masks_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "masks_dir", create=True)


def get_labels_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "labels_dir", create=True)


def get_outputs_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "outputs_dir", create=True)


def get_sam_model_type(config: Mapping[str, Any]) -> str:
    value = str(config["sam_model_type"]).strip()
    if not value:
        raise ValueError("sam_model_type cannot be empty")
    return value


def get_device(config: Mapping[str, Any]) -> str:
    value = str(config["device"]).strip()
    if not value:
        raise ValueError("device cannot be empty")
    return value


def _require_dir(config: Mapping[str, Any], key: str, *, create: bool = False) -> Path:
    path = _resolve_path(config, key)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory for {key}: {path}")
    return path


def _resolve_path(config: Mapping[str, Any], key: str) -> Path:
    if key not in config:
        raise KeyError(f"Config key {key} is missing")
    raw_value = config[key]
    if not isinstance(raw_value, str):
        raise TypeError(f"Config value for {key} must be a string path")

    base_dir = Path(config.get("_config_dir", CONFIG_PATH.parent))
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


__all__ = [
    "CONFIG_PATH",
    "REQUIRED_KEYS",
    "get_colmap_model_dir",
    "get_device",
    "get_images_dir",
    "get_labels_dir",
    "get_masks_dir",
    "get_outputs_dir",
    "get_sam_checkpoint",
    "get_sam_model_type",
    "load_config",
]
