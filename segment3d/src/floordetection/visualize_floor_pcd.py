"""Visualize floor points in a colored point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from ..colmap_io import load_colmap_model, index_point3d
from ..io_paths import get_colmap_model_dir, get_outputs_dir, load_config


def create_floor_point_cloud(
    dataset_name: str, output_filename: str = "floor_visualization.ply"
) -> None:
    """Create and save a point cloud with floor points colored red and others white.

    Args:
        dataset_name: Name of the dataset to process
        output_filename: Name of the output PLY file (default: "floor_visualization.ply")
    """
    # Load configuration
    config = load_config(dataset_name)
    colmap_model_dir = get_colmap_model_dir(config)
    outputs_dir = Path(get_outputs_dir(config))

    # Load floor points from JSON
    floor_json_path = outputs_dir / "filtered_floor_3d_points.json"
    if not floor_json_path.exists():
        raise FileNotFoundError(
            f"Floor points JSON not found: {floor_json_path}\n"
            "Please run the floor detection first."
        )

    print(f"Loading floor points from: {floor_json_path}")
    with open(floor_json_path, "r") as f:
        floor_points_data = json.load(f)

    # Create set of floor point IDs for fast lookup
    floor_point_ids = {point["point_id"] for point in floor_points_data}
    print(f"Loaded {len(floor_point_ids)} floor points")

    # Load COLMAP model
    print(f"Loading COLMAP model from: {colmap_model_dir}")
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))
    print(f"Loaded {len(points3D)} total 3D points")

    # Index the 3D points
    point3d_index = index_point3d(points3D)

    # Prepare point cloud data
    all_coords = []
    all_colors = []

    for point_id, point_data in point3d_index.items():
        xyz = point_data["xyz"]
        all_coords.append(xyz)

        # Color: red for floor points, white for others
        if point_id in floor_point_ids:
            color = [1.0, 0.0, 0.0]  # Red
        else:
            color = [1.0, 1.0, 1.0]  # White

        all_colors.append(color)

    # Convert to numpy arrays
    points = np.array(all_coords, dtype=np.float64)
    colors = np.array(all_colors, dtype=np.float64)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save point cloud
    output_path = outputs_dir / output_filename
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"\nSaved point cloud to: {output_path}")

    # Print summary
    print(f"Created point cloud with {len(points)} points")
    print(f"  Floor points (red): {len(floor_point_ids)}")
    print(f"  Other points (white): {len(points) - len(floor_point_ids)}")
    print(f"  %age of floor points: {100 * len(floor_point_ids) / len(points):.2f}%")
    print("Bounds of coordinates:")
    print(f"  X: {points[:,0].min():.2f} to {points[:,0].max():.2f}")
    print(f"  Y: {points[:,1].min():.2f} to {points[:,1].max():.2f}")
    print(f"  Z: {points[:,2].min():.2f} to {points[:,2].max():.2f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize floor points in a colored point cloud"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to process"
    )
    args = parser.parse_args()

    create_floor_point_cloud(args.dataset)


if __name__ == "__main__":
    main()
