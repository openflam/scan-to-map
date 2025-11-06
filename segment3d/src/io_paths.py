"""Utilities for reading configuration paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

# Import the config module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config, REQUIRED_KEYS


def load_config(dataset_name: str) -> Dict[str, Any]:
    """
    Load configuration for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "Area2300"). Required.

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If dataset_name is not provided or not recognized
    """
    # Check for dataset_name in environment variable if not provided as argument
    if not dataset_name:
        env_dataset = os.environ.get("SCAN_TO_MAP_DATASET")
        if not env_dataset:
            raise ValueError(
                "dataset_name is required. Either pass it as an argument or set "
                "the SCAN_TO_MAP_DATASET environment variable."
            )
        dataset_name = env_dataset

    return get_config(dataset_name)


def get_images_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "images_dir")


def get_colmap_model_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "colmap_model_dir")


def get_sam_checkpoint(config: Mapping[str, Any]) -> Path:
    path = _resolve_path(config, "sam_ckpt")
    if not path.is_file():
        raise FileNotFoundError(f"SAM checkpoint file not found: {path}")
    return path


def get_fastsam_checkpoint(config: Mapping[str, Any]) -> Path:
    path = _resolve_path(config, "fastsam_ckpt")
    # if not path.is_file():
    #     raise FileNotFoundError(f"FastSAM checkpoint file not found: {path}")
    return path


def get_masks_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "masks_dir", create=True)


def get_masks_images_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "masks_images_dir", create=True)


def get_associations_dir(config: Mapping[str, Any]) -> Path:
    return _require_dir(config, "associations_dir", create=True)


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

    base_dir = Path(config.get("_config_dir", Path(__file__).resolve().parents[1]))
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


__all__ = [
    "REQUIRED_KEYS",
    "get_colmap_model_dir",
    "get_device",
    "get_images_dir",
    "get_associations_dir",
    "get_masks_dir",
    "get_masks_images_dir",
    "get_outputs_dir",
    "get_sam_checkpoint",
    "get_fastsam_checkpoint",
    "get_sam_model_type",
    "load_config",
]
