"""
Configuration for different datasets.

This module provides dataset-specific configurations for the scan-to-map pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


# Default parameters for the scan-to-map pipeline
DEFAULT_PARAMETERS = {
    # Segmentation parameters
    "use_full_sam": False,  # Use Full SAM instead of Fast SAM for segmentation
    "fastsam_imgsz": 1024,  # Input image size for FastSAM
    "fastsam_conf": 0.4,  # Confidence threshold for FastSAM
    "fastsam_iou": 0.7,  # IoU threshold for NMS in FastSAM
    "fastsam_batch_size": 32,  # Batch size for FastSAM inference
    "fastsam_num_workers": 4,  # Number of worker threads for parallel I/O in FastSAM
    
    # Mask graph parameters
    "K": 5,  # Number of nearest neighbors for mask graph
    "tau": 0.2,  # Jaccard similarity threshold for mask graph
    "min_points_in_3D_segment": 5,  # Minimum points in 3D segment for mask graph
    
    # Bounding box parameters
    "percentile": 95.0,  # Percentile threshold for bbox outlier removal
    "min_fraction": 0.3,  # Minimum fraction of visible points for projection
    
    # Captioning parameters
    "caption_n_images": 1,  # Number of top images to use for captioning
    "captioner_type": "vllm",  # Type of captioner to use
    "caption_model": "Qwen/Qwen2.5-VL-7B-Instruct",  # VLM model to use for captioning
    "caption_device": 0,  # GPU device ID for captioning
    "caption_batch_size": 512,  # Batch size for captioning inference
    
    # CLIP embedding parameters
    "clip_model": "ViT-H-14",  # OpenCLIP model name for embeddings
    "clip_pretrained": "laion2B-s32B-b79K",  # Pretrained weights for CLIP model
    "clip_batch_size": 32,  # Batch size for CLIP embedding generation
    "clip_device": 0,  # GPU device ID for CLIP embeddings
}


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
        "fastsam_ckpt": checkpoints_dir / "FastSAM-x.pt",
        "masks_dir": outputs_base / dataset_name / "masks",
        "masks_images_dir": outputs_base / dataset_name / "masks_images",
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
    "masks_images_dir",
    "associations_dir",
    "outputs_dir",
    "device",
}
