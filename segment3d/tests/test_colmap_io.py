"""
Test the colmap_io module functions using pytest.
"""

import os
import numpy as np
import pytest

from colmap_io import load_colmap_model, index_image_metadata, index_point3d
from io_paths import load_config, get_colmap_model_dir


@pytest.fixture(scope="module")
def colmap_data():
    """Load COLMAP model data for testing."""
    config = load_config()
    model_dir = get_colmap_model_dir(config)

    if not os.path.exists(model_dir):
        pytest.skip(f"Model directory {model_dir} not found. Skipping COLMAP tests.")

    cameras, images, points3D = load_colmap_model(str(model_dir))
    return cameras, images, points3D


def test_load_colmap_model(colmap_data):
    """Test that COLMAP model loads successfully."""
    cameras, images, points3D = colmap_data

    assert isinstance(cameras, dict), "cameras should be a dictionary"
    assert isinstance(images, dict), "images should be a dictionary"
    assert isinstance(points3D, dict), "points3D should be a dictionary"

    assert len(cameras) > 0, "Should have at least one camera"
    assert len(images) > 0, "Should have at least one image"
    assert len(points3D) > 0, "Should have at least one 3D point"


def test_index_image_metadata(colmap_data):
    """Test image metadata indexing."""
    cameras, images, points3D = colmap_data
    image_metadata = index_image_metadata(images)

    assert isinstance(image_metadata, dict), "image_metadata should be a dictionary"
    assert len(image_metadata) == len(images), "Should have metadata for all images"

    # Test structure of a sample image
    if image_metadata:
        sample_id = list(image_metadata.keys())[0]
        sample_meta = image_metadata[sample_id]

        assert "name" in sample_meta, "Metadata should contain 'name'"
        assert "xys" in sample_meta, "Metadata should contain 'xys'"
        assert "point3D_ids" in sample_meta, "Metadata should contain 'point3D_ids'"

        assert isinstance(sample_meta["name"], str), "name should be a string"
        assert isinstance(sample_meta["xys"], np.ndarray), "xys should be a numpy array"
        assert isinstance(
            sample_meta["point3D_ids"], np.ndarray
        ), "point3D_ids should be a numpy array"

        assert sample_meta["xys"].ndim == 2, "xys should be 2D array"
        assert sample_meta["xys"].shape[1] == 2, "xys should have 2 columns (x, y)"
        assert sample_meta["point3D_ids"].ndim == 1, "point3D_ids should be 1D array"

        # xys and point3D_ids should have same number of entries
        assert len(sample_meta["xys"]) == len(
            sample_meta["point3D_ids"]
        ), "xys and point3D_ids should have same length"


def test_index_point3d(colmap_data):
    """Test 3D point indexing."""
    cameras, images, points3D = colmap_data
    point3d_index = index_point3d(points3D)

    assert isinstance(point3d_index, dict), "point3d_index should be a dictionary"
    assert len(point3d_index) == len(points3D), "Should have index for all 3D points"

    # Test structure of a sample 3D point
    if point3d_index:
        sample_id = list(point3d_index.keys())[0]
        sample_point = point3d_index[sample_id]

        assert "xyz" in sample_point, "Point should contain 'xyz'"
        assert "rgb" in sample_point, "Point should contain 'rgb'"

        assert isinstance(
            sample_point["xyz"], np.ndarray
        ), "xyz should be a numpy array"
        assert isinstance(
            sample_point["rgb"], np.ndarray
        ), "rgb should be a numpy array"

        assert sample_point["xyz"].shape == (3,), "xyz should be shape (3,)"
        assert sample_point["rgb"].shape == (3,), "rgb should be shape (3,)"


def test_image_metadata_consistency(colmap_data):
    """Test that image metadata keys match original image keys."""
    cameras, images, points3D = colmap_data
    image_metadata = index_image_metadata(images)

    assert set(image_metadata.keys()) == set(
        images.keys()
    ), "Image metadata keys should match original image keys"


def test_point3d_consistency(colmap_data):
    """Test that 3D point index keys match original point keys."""
    cameras, images, points3D = colmap_data
    point3d_index = index_point3d(points3D)

    assert set(point3d_index.keys()) == set(
        points3D.keys()
    ), "Point3D index keys should match original point keys"


import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
SEGMENT3D_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SEGMENT3D_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colmap_io import load_colmap_model, index_image_metadata, index_point3d
from io_paths import load_config, get_colmap_model_dir


def test_colmap_io():
    """Test the colmap_io functions with a sample COLMAP model."""

    # Load config and get the COLMAP model directory
    config = load_config()
    model_dir = get_colmap_model_dir(config)

    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Skipping test.")
        return

    print(f"Testing with model directory: {model_dir}")

    # Test load_colmap_model
    try:
        cameras, images, points3D = load_colmap_model(model_dir)
        print(f"Loaded COLMAP model successfully")
        print(f"  - {len(cameras)} cameras")
        print(f"  - {len(images)} images")
        print(f"  - {len(points3D)} 3D points")

        # Test index_image_metadata
        image_metadata = index_image_metadata(images)
        print(f"Indexed image metadata for {len(image_metadata)} images")

        # Show sample image metadata
        if image_metadata:
            sample_id = list(image_metadata.keys())[0]
            sample_meta = image_metadata[sample_id]
            print(f"  Sample image {sample_id}:")
            print(f"    - name: {sample_meta['name']}")
            print(f"    - keypoints shape: {sample_meta['xys'].shape}")
            print(f"    - point3D_ids shape: {sample_meta['point3D_ids'].shape}")

        # Test index_point3d
        point3d_index = index_point3d(points3D)
        print(f"Indexed 3D points for {len(point3d_index)} points")

        # Show sample 3D point
        if point3d_index:
            sample_id = list(point3d_index.keys())[0]
            sample_point = point3d_index[sample_id]
            print(f"  Sample point {sample_id}:")
            print(f"    - xyz: {sample_point['xyz']}")
            print(f"    - rgb: {sample_point['rgb']}")
            print(f"    - error: {sample_point['error']}")

        print("All tests passed!")

    except Exception as e:
        print(f"Error testing colmap_io: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_colmap_io()
