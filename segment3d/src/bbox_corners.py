"""Compute bounding box corners for sets of 3D points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .colmap_io import load_colmap_model


def get_bbox(
    point3D_ids: List[int], colmap_model_dir: str | Path, percentile: float = 95.0
) -> Dict[str, Any]:
    """
    Compute bounding box corners for a set of 3D point IDs.

    Filters out outliers by cutting off points beyond the specified percentile
    on all axes, then returns the corners of the bounding box.

    Args:
        point3D_ids: List of COLMAP 3D point IDs
        colmap_model_dir: Path to COLMAP model directory
        percentile: Percentile threshold for outlier removal (default: 95.0)

    Returns:
        Dictionary containing:
        - "corners": np.ndarray(8, 3) - Eight corners of the bounding box
        - "min": np.ndarray(3,) - Minimum coordinates [x, y, z]
        - "max": np.ndarray(3,) - Maximum coordinates [x, y, z]
        - "center": np.ndarray(3,) - Center of bounding box
        - "size": np.ndarray(3,) - Size of bounding box [width, height, depth]
        - "num_points": int - Number of points after filtering
        - "num_filtered": int - Number of points filtered out
    """
    # Load COLMAP model
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))

    # Collect 3D coordinates for valid point IDs
    coords_list = []
    for point_id in point3D_ids:
        if point_id in points3D:
            coords_list.append(points3D[point_id].xyz)

    if len(coords_list) == 0:
        raise ValueError("No valid 3D points found for the given IDs")

    # Convert to numpy array (N, 3)
    coords = np.array(coords_list)
    num_original = len(coords)

    # Filter outliers: keep only points within percentile range on all axes
    percentile_low = (100.0 - percentile) / 2.0
    percentile_high = 100.0 - percentile_low

    # Compute percentile bounds for each axis
    min_bounds = np.percentile(coords, percentile_low, axis=0)
    max_bounds = np.percentile(coords, percentile_high, axis=0)

    # Keep points within bounds on all axes
    mask = np.all((coords >= min_bounds) & (coords <= max_bounds), axis=1)
    filtered_coords = coords[mask]

    if len(filtered_coords) == 0:
        raise ValueError("All points were filtered out as outliers")

    # Compute bounding box
    bbox_min = filtered_coords.min(axis=0)
    bbox_max = filtered_coords.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min

    # Compute 8 corners of the bounding box
    # Order: (min_x, min_y, min_z), (max_x, min_y, min_z), ...
    corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],  # 0: min, min, min
            [bbox_max[0], bbox_min[1], bbox_min[2]],  # 1: max, min, min
            [bbox_max[0], bbox_max[1], bbox_min[2]],  # 2: max, max, min
            [bbox_min[0], bbox_max[1], bbox_min[2]],  # 3: min, max, min
            [bbox_min[0], bbox_min[1], bbox_max[2]],  # 4: min, min, max
            [bbox_max[0], bbox_min[1], bbox_max[2]],  # 5: max, min, max
            [bbox_max[0], bbox_max[1], bbox_max[2]],  # 6: max, max, max
            [bbox_min[0], bbox_max[1], bbox_max[2]],  # 7: min, max, max
        ]
    )

    return {
        "corners": corners,
        "min": bbox_min,
        "max": bbox_max,
        "center": bbox_center,
        "size": bbox_size,
        "num_points": len(filtered_coords),
        "num_filtered": num_original - len(filtered_coords),
    }
