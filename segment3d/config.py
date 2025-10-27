"""
Configuration for different datasets.

This module provides dataset-specific configurations for the scan-to-map pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def get_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "Area2300")

    Returns:
        Dictionary containing all configuration paths and settings

    Raises:
        ValueError: If dataset_name is empty or invalid
    """
    if not dataset_name or not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a non-empty string")

    # Base paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    checkpoints_dir = base_dir / "checkpoints"
    outputs_base = base_dir / "outputs"

    # Generate paths based on dataset name using consistent structure
    config = {
        "images_dir": data_dir / dataset_name / "ns_data" / "images",
        "colmap_model_dir": data_dir
        / dataset_name
        / "hloc_data"
        / "sfm_reconstruction",
        "sam_model_type": "vit_h",
        "sam_ckpt": checkpoints_dir / "sam_vit_h_4b8939.pth",
        "masks_dir": outputs_base / dataset_name / "masks",
        "associations_dir": outputs_base / dataset_name / "associations",
        "outputs_dir": outputs_base / dataset_name,
        "device": "cuda",
    }

    # Convert all paths to strings for compatibility
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }

    # Add dataset name to config
    config["dataset_name"] = dataset_name

    # Add the config directory for reference
    config["_config_dir"] = str(Path(__file__).resolve().parent)

    return config


def list_datasets() -> list[str]:
    """
    Get list of available dataset names by scanning the data directory.

    Returns:
        List of dataset names (subdirectories in data/)
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    if not data_dir.exists():
        return []

    # Return all subdirectories in data/ as potential datasets
    datasets = [
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    return sorted(datasets)


# For backwards compatibility with config.yaml
REQUIRED_KEYS = {
    "images_dir",
    "colmap_model_dir",
    "sam_ckpt",
    "masks_dir",
    "associations_dir",
    "outputs_dir",
    "device",
}
