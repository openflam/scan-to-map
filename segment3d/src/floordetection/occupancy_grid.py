"""Generate occupancy grid from filtered floor points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..colmap_io import load_colmap_model, index_point3d
from ..io_paths import get_colmap_model_dir, get_outputs_dir, load_config


def load_filtered_floor_points(filepath: Path) -> List[Dict]:
    """Load filtered floor points from JSON file.

    Args:
        filepath: Path to filtered floor points JSON

    Returns:
        List of dictionaries with 'point_id' and 'coords'
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'.")
        sys.exit(1)


def remove_outliers(all_points: Dict) -> Dict:
    """Remove outlier points based on percentiles in all dimensions.

    Args:
        all_points: Dictionary of all 3D points from COLMAP (indexed)
        up_axis_index: Index of the up axis (1 for y, 2 for z)

    Returns:
        Filtered dictionary without outliers
    """
    if not all_points:
        return {}

    # Extract all coordinates
    coords = np.array([p["xyz"] for p in all_points.values()])
    point_ids = list(all_points.keys())

    # Calculate percentile thresholds for each dimension
    lower_percentiles = np.percentile(coords, 5, axis=0)
    upper_percentiles = np.percentile(coords, 95, axis=0)

    # Filter points within bounds for all dimensions
    mask = np.all((coords >= lower_percentiles) & (coords <= upper_percentiles), axis=1)

    # Create filtered dictionary
    filtered = {}
    for i, pid in enumerate(point_ids):
        if mask[i]:
            filtered[pid] = all_points[pid]

    print(
        f"Removed {len(all_points) - len(filtered)} outlier points "
        f"(outside 5th-95th percentile range)"
    )
    return filtered


def remove_ceiling_points_from_colmap(
    all_points: Dict, up_axis_index: int, percentile: float = 80.0
) -> Dict:
    """Remove points that likely correspond to the ceiling from COLMAP points.

    Uses a simple heuristic: removes points above a certain percentile
    in the up-axis direction.

    Args:
        all_points: Dictionary of all 3D points from COLMAP (indexed)
        up_axis_index: Index of the up axis (1 for y, 2 for z)
        percentile: Percentile threshold (default 80.0)

    Returns:
        Filtered dictionary without ceiling points
    """
    if not all_points:
        return {}

    # Extract all coordinates
    coords = np.array([p["xyz"] for p in all_points.values()])
    height_values = coords[:, up_axis_index]

    # Calculate threshold - remove points above this percentile
    threshold = np.percentile(height_values, percentile)

    # Filter points
    filtered = {}
    for pid, point_data in all_points.items():
        if point_data["xyz"][up_axis_index] <= threshold:
            filtered[pid] = point_data

    print(
        f"Removed {len(all_points) - len(filtered)} ceiling points from COLMAP data "
        f"(threshold: {threshold:.4f})"
    )
    return filtered


def create_occupancy_grid(
    floor_points: List[Dict],
    all_points: Dict,
    up_axis_index: int,
    cell_size: float = 0.1,
    floor_threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """Create an occupancy grid from floor points.

    Args:
        floor_points: List of filtered floor point dictionaries
        all_points: Dictionary of all 3D points from COLMAP
        up_axis_index: Index of the up axis (1 for y, 2 for z)
        cell_size: Size of each grid cell in meters (default 0.1)
        floor_threshold: Fraction of floor points required to mark cell as floor

    Returns:
        Tuple of (grid, metadata) where:
        - grid: 2D numpy array (0=floor, 1=occupied)
        - metadata: Dictionary with grid information
    """
    # Determine which axes are horizontal (not up)
    axes = [0, 1, 2]
    axes.remove(up_axis_index)
    x_axis, y_axis = axes[0], axes[1]

    # Get all floor point IDs for fast lookup
    floor_point_ids = set(p["point_id"] for p in floor_points)

    # Extract coordinates from floor points
    floor_coords = np.array([p["coords"] for p in floor_points])
    floor_xy = floor_coords[:, [x_axis, y_axis]]

    # Extract all point coordinates
    all_coords = []
    for pid, point_data in all_points.items():
        all_coords.append(point_data["xyz"])
    all_coords = np.array(all_coords)
    all_xy = all_coords[:, [x_axis, y_axis]]

    # Determine grid bounds from all points
    min_x, min_y = all_xy.min(axis=0)
    max_x, max_y = all_xy.max(axis=0)

    # Add padding
    padding = cell_size * 2
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    # Calculate grid dimensions
    n_cells_x = int(np.ceil((max_x - min_x) / cell_size))
    n_cells_y = int(np.ceil((max_y - min_y) / cell_size))

    print(f"Grid size: {n_cells_x} x {n_cells_y} cells")
    print(f"Cell size: {cell_size}m")
    print(f"Total cells: {n_cells_x * n_cells_y}")

    # Initialize grid (0 = floor, 1 = occupied)
    grid = np.ones((n_cells_y, n_cells_x), dtype=np.uint8)

    # For each cell, count floor vs all points
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Calculate cell bounds
            cell_min_x = min_x + j * cell_size
            cell_max_x = cell_min_x + cell_size
            cell_min_y = min_y + i * cell_size
            cell_max_y = cell_min_y + cell_size

            # Find points in this cell from all points
            in_cell_mask = (
                (all_xy[:, 0] >= cell_min_x)
                & (all_xy[:, 0] < cell_max_x)
                & (all_xy[:, 1] >= cell_min_y)
                & (all_xy[:, 1] < cell_max_y)
            )
            n_total = in_cell_mask.sum()

            if n_total == 0:
                # No points in cell, mark as floor. It's likely empty space.
                grid[i, j] = 0
                continue

            # Find floor points in this cell
            in_cell_floor_mask = (
                (floor_xy[:, 0] >= cell_min_x)
                & (floor_xy[:, 0] < cell_max_x)
                & (floor_xy[:, 1] >= cell_min_y)
                & (floor_xy[:, 1] < cell_max_y)
            )
            n_floor = in_cell_floor_mask.sum()

            # Calculate fraction of floor points
            floor_fraction = n_floor / n_total if n_total > 0 else 0

            # Mark cell as floor if fraction exceeds threshold
            if floor_fraction >= floor_threshold:
                grid[i, j] = 0
            else:
                grid[i, j] = 1

    # Calculate statistics
    n_floor_cells = (grid == 0).sum()
    n_occupied_cells = (grid == 1).sum()

    print(f"Floor cells: {n_floor_cells} ({100*n_floor_cells/grid.size:.1f}%)")
    print(f"Occupied cells: {n_occupied_cells} ({100*n_occupied_cells/grid.size:.1f}%)")

    # Create metadata
    metadata = {
        "cell_size": cell_size,
        "grid_shape": [int(n_cells_y), int(n_cells_x)],
        "origin": [float(min_x), float(min_y)],
        "bounds": {
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y),
        },
        "axes": {
            "x_axis_index": x_axis,
            "y_axis_index": y_axis,
            "up_axis_index": up_axis_index,
        },
        "statistics": {
            "total_cells": int(grid.size),
            "floor_cells": int(n_floor_cells),
            "occupied_cells": int(n_occupied_cells),
            "floor_percentage": float(100 * n_floor_cells / grid.size),
            "occupied_percentage": float(100 * n_occupied_cells / grid.size),
        },
        "floor_threshold": floor_threshold,
    }

    return grid, metadata


def save_occupancy_grid(
    grid: np.ndarray, metadata: Dict, output_dir: Path, all_points: Dict
) -> None:
    """Save occupancy grid and metadata to files.

    Args:
        grid: 2D occupancy grid array
        metadata: Grid metadata dictionary
        output_dir: Output directory path
        all_points: Dictionary of all 3D points from COLMAP
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save grid as text for easy viewing
    text_file = output_dir / "occupancy_grid.txt"
    with open(text_file, "w") as f:
        for row in grid:
            f.write("".join(["." if cell == 0 else "#" for cell in row]) + "\n")
    print(f"Saved text visualization to: {text_file}")

    # Save as bounding boxes in JSON format
    bbox_file = output_dir / "occupancy_bbox.json"
    bboxes, floor_height_data = _grid_to_bboxes(grid, metadata, all_points)
    with open(bbox_file, "w") as f:
        json.dump(bboxes, f, indent=2)
    print(f"Saved bounding boxes to: {bbox_file}")

    # Save metadata as JSON
    metadata_file = output_dir / "occupancy_grid_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")

    # Save grid as npy
    npy_file = output_dir / "occupancy_grid.npy"
    np.save(npy_file, grid)
    print(f"Saved grid to: {npy_file}")

    # Save floor height
    floor_height_file = output_dir / "floor_height.json"
    with open(floor_height_file, "w") as f:
        json.dump(floor_height_data, f, indent=2)
    print(f"Saved floor height to: {floor_height_file}")


def _grid_to_bboxes(
    grid: np.ndarray, metadata: Dict, all_points: Dict
) -> Tuple[List[Dict], Dict]:
    """Convert occupancy grid to bounding box format.

    Args:
        grid: 2D occupancy grid array
        metadata: Grid metadata dictionary
        all_points: Dictionary of all 3D points from COLMAP

    Returns:
        Tuple of (bboxes, floor_height_data) where:
        - bboxes: List of bounding box dictionaries in bbox_corners.json format
        - floor_height_data: Dictionary with floor height information
    """
    cell_size = metadata["cell_size"]
    origin = metadata["origin"]
    x_axis_index = metadata["axes"]["x_axis_index"]
    y_axis_index = metadata["axes"]["y_axis_index"]
    up_axis_index = metadata["axes"]["up_axis_index"]

    # Calculate floor height as 10th percentile of up-axis values
    all_coords = np.array([p["xyz"] for p in all_points.values()])
    height_values = all_coords[:, up_axis_index]
    floor_height = np.percentile(height_values, 10)

    # Prepare floor height data for return
    floor_height_data = {
        "floor_height": float(floor_height),
        "up_axis_index": up_axis_index,
        "percentile": 10,
    }

    # Fixed height for boxes
    box_height = 0.1

    bboxes = []
    cell_id = 0

    n_cells_y, n_cells_x = grid.shape

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_type = int(grid[i, j])  # 0 = floor, 1 = occupied

            if cell_type == 1:
                continue  # Skip occupied cells for bbox generation

            # Calculate cell bounds in 2D
            cell_min_x = origin[0] + j * cell_size
            cell_max_x = cell_min_x + cell_size
            cell_min_y = origin[1] + i * cell_size
            cell_max_y = cell_min_y + cell_size

            # Create 3D coordinates based on up axis
            # Position boxes at floor_height with fixed box_height
            if up_axis_index == 2:  # z is up
                # x_axis=0, y_axis=1, up_axis=2
                min_3d = [cell_min_x, cell_min_y, floor_height]
                max_3d = [cell_max_x, cell_max_y, floor_height + box_height]
            elif up_axis_index == 1:  # y is up
                # x_axis=0, y_axis=2, up_axis=1
                min_3d = [cell_min_x, floor_height, cell_min_y]
                max_3d = [cell_max_x, floor_height + box_height, cell_max_y]
            else:  # x is up (unlikely but handle it)
                # x_axis=1, y_axis=2, up_axis=0
                min_3d = [floor_height, cell_min_x, cell_min_y]
                max_3d = [floor_height + box_height, cell_max_x, cell_max_y]

            # Calculate 8 corners of the bounding box
            corners = [
                [min_3d[0], min_3d[1], min_3d[2]],  # 0: min, min, min
                [max_3d[0], min_3d[1], min_3d[2]],  # 1: max, min, min
                [max_3d[0], max_3d[1], min_3d[2]],  # 2: max, max, min
                [min_3d[0], max_3d[1], min_3d[2]],  # 3: min, max, min
                [min_3d[0], min_3d[1], max_3d[2]],  # 4: min, min, max
                [max_3d[0], min_3d[1], max_3d[2]],  # 5: max, min, max
                [max_3d[0], max_3d[1], max_3d[2]],  # 6: max, max, max
                [min_3d[0], max_3d[1], max_3d[2]],  # 7: min, max, max
            ]

            # Calculate center
            center = [
                (min_3d[0] + max_3d[0]) / 2,
                (min_3d[1] + max_3d[1]) / 2,
                (min_3d[2] + max_3d[2]) / 2,
            ]

            # Calculate size
            size = [
                max_3d[0] - min_3d[0],
                max_3d[1] - min_3d[1],
                max_3d[2] - min_3d[2],
            ]

            bbox_dict = {
                "cell_type": "floor" if cell_type == 0 else "occupied",
                "bbox": {
                    "corners": corners,
                    "min": min_3d,
                    "max": max_3d,
                    "center": center,
                    "size": size,
                },
            }

            bboxes.append(bbox_dict)
            cell_id += 1

    return bboxes, floor_height_data


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate occupancy grid from filtered floor points"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to process"
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        choices=["y", "z", "Y", "Z"],
        default="z",
        help="The vertical axis (default: z)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.3,
        help="Size of each grid cell in meters (default: 0.3)",
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=0.7,
        help="Fraction of floor points to mark cell as floor (default: 0.7)",
    )

    args = parser.parse_args()

    # Normalize up axis
    up_axis_map = {"y": 1, "z": 2, "Y": 1, "Z": 2}
    up_axis_index = up_axis_map[args.up_axis]

    # Load configuration
    config = load_config(args.dataset)
    outputs_dir = Path(get_outputs_dir(config))
    colmap_model_dir = Path(get_colmap_model_dir(config))

    # Load filtered floor points
    floor_points_file = outputs_dir / "filtered_floor_3d_points.json"
    floor_points = load_filtered_floor_points(floor_points_file)
    print(f"Loaded {len(floor_points)} filtered floor points")

    # Load COLMAP model
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))
    print(f"Loaded {len(points3D)} 3D points from COLMAP")

    # Index all points
    all_points = index_point3d(points3D)

    # Remove outliers from COLMAP data
    print("\nRemoving outliers from COLMAP data...")
    all_points = remove_outliers(all_points)
    print(f"Remaining COLMAP points: {len(all_points)}")

    # Remove ceiling points from COLMAP data
    print("\nRemoving ceiling points from COLMAP data...")
    all_points = remove_ceiling_points_from_colmap(all_points, up_axis_index)
    print(f"Remaining COLMAP points: {len(all_points)}")

    # Create occupancy grid
    print("\nCreating occupancy grid...")
    grid, metadata = create_occupancy_grid(
        floor_points=floor_points,
        all_points=all_points,
        up_axis_index=up_axis_index,
        cell_size=args.cell_size,
        floor_threshold=args.floor_threshold,
    )

    # Save results
    print("\nSaving occupancy grid...")
    save_occupancy_grid(grid, metadata, outputs_dir, all_points)

    print(f"\nOccupancy grid generation complete!")


if __name__ == "__main__":
    main()
