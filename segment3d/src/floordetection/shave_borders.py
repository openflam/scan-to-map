"""Shave the outer border cells off a generated occupancy grid.

The occupancy grid marks empty cells (no COLMAP points) as floor, which causes
the floor region to extend past the actual scanned area as a padded border of
free space around the edges.  This script removes that border by marking the
outermost ``--border`` rings of cells as occupied and regenerating all derived
files (``.npy``, ``.txt``, ``occupancy_bbox.json`` and the metadata statistics)
so they stay consistent with :mod:`occupancy_grid`.

It operates purely on the already-generated outputs, so there is no need to
reload COLMAP or rerun any earlier stage of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..io_paths import get_outputs_dir, load_config

# Must match the box height used in occupancy_grid._grid_to_bboxes.
BOX_HEIGHT = 0.1


def _load_json(filepath: Path) -> Dict:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Run occupancy_grid first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'.")
        sys.exit(1)


def shave_borders(grid: np.ndarray, border: int) -> np.ndarray:
    """Mark the outermost ``border`` rings of cells as occupied (1).

    Args:
        grid: 2D occupancy grid array (0=floor, 1=occupied)
        border: Number of cell rings to remove from every edge

    Returns:
        A new grid with the border cells set to occupied.
    """
    if border <= 0:
        return grid.copy()

    n_rows, n_cols = grid.shape
    if 2 * border >= n_rows or 2 * border >= n_cols:
        print(
            f"Error: border={border} is too large for a {n_rows}x{n_cols} grid; "
            f"it would remove the entire grid."
        )
        sys.exit(1)

    shaved = grid.copy()
    shaved[:border, :] = 1
    shaved[-border:, :] = 1
    shaved[:, :border] = 1
    shaved[:, -border:] = 1
    return shaved


def grid_to_bboxes(
    grid: np.ndarray, metadata: Dict, floor_height: float
) -> List[Dict]:
    """Convert floor cells of the grid to bounding boxes.

    Mirrors occupancy_grid._grid_to_bboxes, but takes the floor height directly
    (from floor_height.json) instead of recomputing it from COLMAP points.
    """
    cell_size = metadata["cell_size"]
    origin = metadata["origin"]
    up_axis_index = metadata["axes"]["up_axis_index"]

    bboxes: List[Dict] = []
    n_cells_y, n_cells_x = grid.shape

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            if int(grid[i, j]) == 1:
                continue  # Skip occupied cells

            cell_min_x = origin[0] + j * cell_size
            cell_max_x = cell_min_x + cell_size
            cell_min_y = origin[1] + i * cell_size
            cell_max_y = cell_min_y + cell_size

            if up_axis_index == 2:  # z is up
                min_3d = [cell_min_x, cell_min_y, floor_height]
                max_3d = [cell_max_x, cell_max_y, floor_height + BOX_HEIGHT]
            elif up_axis_index == 1:  # y is up
                min_3d = [cell_min_x, floor_height, cell_min_y]
                max_3d = [cell_max_x, floor_height + BOX_HEIGHT, cell_max_y]
            else:  # x is up
                min_3d = [floor_height, cell_min_x, cell_min_y]
                max_3d = [floor_height + BOX_HEIGHT, cell_max_x, cell_max_y]

            corners = [
                [min_3d[0], min_3d[1], min_3d[2]],
                [max_3d[0], min_3d[1], min_3d[2]],
                [max_3d[0], max_3d[1], min_3d[2]],
                [min_3d[0], max_3d[1], min_3d[2]],
                [min_3d[0], min_3d[1], max_3d[2]],
                [max_3d[0], min_3d[1], max_3d[2]],
                [max_3d[0], max_3d[1], max_3d[2]],
                [min_3d[0], max_3d[1], max_3d[2]],
            ]
            center = [
                (min_3d[0] + max_3d[0]) / 2,
                (min_3d[1] + max_3d[1]) / 2,
                (min_3d[2] + max_3d[2]) / 2,
            ]
            size = [
                max_3d[0] - min_3d[0],
                max_3d[1] - min_3d[1],
                max_3d[2] - min_3d[2],
            ]

            bboxes.append(
                {
                    "cell_type": "floor",
                    "bbox": {
                        "corners": corners,
                        "min": min_3d,
                        "max": max_3d,
                        "center": center,
                        "size": size,
                    },
                }
            )

    return bboxes


def save_outputs(
    grid: np.ndarray, metadata: Dict, floor_height: float, outputs_dir: Path
) -> None:
    """Write the shaved grid and all derived files back to the outputs dir."""
    # Update metadata statistics to reflect the shaved grid.
    n_floor_cells = int((grid == 0).sum())
    n_occupied_cells = int((grid == 1).sum())
    metadata["statistics"] = {
        "total_cells": int(grid.size),
        "floor_cells": n_floor_cells,
        "occupied_cells": n_occupied_cells,
        "floor_percentage": float(100 * n_floor_cells / grid.size),
        "occupied_percentage": float(100 * n_occupied_cells / grid.size),
    }

    npy_file = outputs_dir / "occupancy_grid.npy"
    np.save(npy_file, grid)
    print(f"Saved grid to: {npy_file}")

    text_file = outputs_dir / "occupancy_grid.txt"
    with open(text_file, "w") as f:
        for row in grid:
            f.write("".join(["." if cell == 0 else "#" for cell in row]) + "\n")
    print(f"Saved text visualization to: {text_file}")

    bbox_file = outputs_dir / "occupancy_bbox.json"
    bboxes = grid_to_bboxes(grid, metadata, floor_height)
    with open(bbox_file, "w") as f:
        json.dump(bboxes, f, indent=2)
    print(f"Saved bounding boxes to: {bbox_file}")

    metadata_file = outputs_dir / "occupancy_grid_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shave the outer border cells off a generated occupancy grid"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to process"
    )
    parser.add_argument(
        "--border",
        type=int,
        default=2,
        help="Number of cell rings to remove from every edge (default: 2)",
    )
    args = parser.parse_args()

    config = load_config(args.dataset)
    outputs_dir = Path(get_outputs_dir(config))

    grid = np.load(outputs_dir / "occupancy_grid.npy")
    metadata = _load_json(outputs_dir / "occupancy_grid_metadata.json")
    floor_height = _load_json(outputs_dir / "floor_height.json")["floor_height"]

    print(f"Loaded grid of shape {grid.shape[0]} x {grid.shape[1]}")
    n_floor_before = int((grid == 0).sum())

    shaved = shave_borders(grid, args.border)
    n_floor_after = int((shaved == 0).sum())
    print(
        f"Shaved {args.border} border ring(s): "
        f"floor cells {n_floor_before} -> {n_floor_after} "
        f"({n_floor_before - n_floor_after} removed)"
    )

    save_outputs(shaved, metadata, floor_height, outputs_dir)
    print("\nBorder shaving complete!")


if __name__ == "__main__":
    main()
