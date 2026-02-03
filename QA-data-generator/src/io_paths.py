"""
I/O path utilities for QA data generation.
Provides functions to get file paths for datasets, images, and components.
"""

from pathlib import Path
from typing import Dict


def get_project_root() -> Path:
    """Get the project root directory (scan-to-map)."""
    return Path(__file__).parent.parent.parent


def get_image_path(dataset_name: str, image_name: str) -> Path:
    """
    Get the path to an image file.

    Args:
        dataset_name: Name of the dataset (e.g., 'ProjectLabStudio_NoNeg')
        image_name: Name of the image without extension (e.g., 'frame_00001')

    Returns:
        Path to the image file
    """
    project_root = get_project_root()
    image_path = (
        project_root
        / "data"
        / dataset_name
        / "ns_data"
        / "images"
        / f"{image_name}.jpg"
    )
    return image_path


def get_components_json_path(dataset_name: str, image_name: str) -> Path:
    """
    Get the path to the components JSON file for a specific image.

    Args:
        dataset_name: Name of the dataset (e.g., 'ProjectLabStudio_NoNeg')
        image_name: Name of the image without extension (e.g., 'frame_00001')

    Returns:
        Path to the components JSON file
    """
    project_root = get_project_root()
    components_path = (
        project_root
        / "outputs"
        / dataset_name
        / "per_image_components"
        / f"{image_name}_components.json"
    )
    return components_path


def get_query_types_json_path() -> Path:
    """
    Get the path to the query types JSON file.

    Returns:
        Path to the query_types.json file
    """
    return Path(__file__).parent.parent / "query_types.json"


def get_qa_output_path(dataset_name: str, image_name: str) -> Path:
    """
    Get the path where QA synthetic data should be saved.

    Args:
        dataset_name: Name of the dataset (e.g., 'ProjectLabStudio_NoNeg')
        image_name: Name of the image without extension (e.g., 'frame_00001')

    Returns:
        Path to the QA output JSON file
    """
    project_root = get_project_root()
    qa_dir = project_root / "outputs" / dataset_name / "qa_synthetic_data"
    qa_dir.mkdir(parents=True, exist_ok=True)
    return qa_dir / f"{image_name}_qa.json"


def get_all_paths(dataset_name: str, image_name: str) -> Dict[str, Path]:
    """
    Get all required paths for a dataset and image.

    Args:
        dataset_name: Name of the dataset (e.g., 'ProjectLabStudio_NoNeg')
        image_name: Name of the image without extension (e.g., 'frame_00001')

    Returns:
        Dictionary with keys: 'image', 'components', 'query_types', 'qa_output'
    """
    return {
        "image": get_image_path(dataset_name, image_name),
        "components": get_components_json_path(dataset_name, image_name),
        "query_types": get_query_types_json_path(),
        "qa_output": get_qa_output_path(dataset_name, image_name),
    }


def validate_paths(paths: Dict[str, Path], skip_keys: set = None) -> None:
    """
    Validate that all required paths exist.

    Args:
        paths: Dictionary of paths to validate
        skip_keys: Set of keys to skip validation (e.g., output paths that will be created)

    Raises:
        FileNotFoundError: If any required file does not exist
    """
    if skip_keys is None:
        skip_keys = {"qa_output"}  # Default: skip output paths

    for key, path in paths.items():
        if key in skip_keys:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{key} file not found: {path}")
