"""Project 3D bounding boxes onto images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from scipy.spatial.transform import Rotation

from .colmap_io import load_colmap_model, reverse_index_points3D


def project_point_to_image(point_3d: np.ndarray, image, camera) -> np.ndarray | None:
    """
    Project a 3D point onto an image using COLMAP camera model.

    Args:
        point_3d: 3D point coordinates (3,)
        image: COLMAP Image namedtuple
        camera: COLMAP Camera namedtuple

    Returns:
        2D pixel coordinates (2,) or None if point is behind camera
    """
    # Transform point to camera coordinate system
    # COLMAP uses quaternion (qw, qx, qy, qz) and translation vector
    qvec = image.qvec
    tvec = image.tvec

    # Convert quaternion to rotation matrix using scipy
    # COLMAP format is (qw, qx, qy, qz), scipy expects (qx, qy, qz, qw)
    qw, qx, qy, qz = qvec
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    R = rotation.as_matrix()

    # Transform to camera coordinates
    point_cam = R @ point_3d + tvec

    # Check if point is behind camera
    if point_cam[2] <= 0:
        return None

    # Project to normalized image plane
    point_norm = point_cam[:2] / point_cam[2]

    # Apply camera intrinsics
    # COLMAP SIMPLE_RADIAL: params = [f, cx, cy, k]
    # COLMAP PINHOLE: params = [fx, fy, cx, cy]
    # COLMAP RADIAL: params = [f, cx, cy, k1, k2]

    if camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k = camera.params
        r2 = point_norm[0] ** 2 + point_norm[1] ** 2
        radial = 1 + k * r2
        u = f * radial * point_norm[0] + cx
        v = f * radial * point_norm[1] + cy
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        u = fx * point_norm[0] + cx
        v = fy * point_norm[1] + cy
    elif camera.model == "RADIAL":
        f, cx, cy, k1, k2 = camera.params
        r2 = point_norm[0] ** 2 + point_norm[1] ** 2
        radial = 1 + k1 * r2 + k2 * r2**2
        u = f * radial * point_norm[0] + cx
        v = f * radial * point_norm[1] + cy
    else:
        # Fallback: simple projection without distortion
        if len(camera.params) >= 4:
            fx, fy, cx, cy = camera.params[:4]
        else:
            f = camera.params[0]
            fx = fy = f
            cx = camera.params[1] if len(camera.params) > 1 else camera.width / 2
            cy = camera.params[2] if len(camera.params) > 2 else camera.height / 2
        u = fx * point_norm[0] + cx
        v = fy * point_norm[1] + cy

    return np.array([u, v])


def bbox_3d_to_2d(
    bbox_corners_3d: np.ndarray,
    set_of_point3DIds: List[int],
    colmap_model: any,
    min_fraction: float = 0.3,
) -> Dict[str, Any]:
    """
    Project 3D bounding box corners onto images.

    Args:
        bbox_corners_3d: Array of 8 corners (8, 3)
        set_of_point3DIds: List of 3D point IDs in this connected component
        colmap_model_dir: Path to COLMAP model directory
        min_fraction: Minimum fraction of points that must be visible in an image

    Returns:
        Dictionary mapping image_name to 2D bounding box information:
        {
            "image_name": {
                "image_id": int,
                "bbox_2d": [x_min, y_min, x_max, y_max],
                "corners_2d": [[x, y], ...],  # 8 projected corners (may have None)
                "visible_points": int,
                "total_points": int,
                "fraction_visible": float
            }
        }
    """
    # Load COLMAP model
    cameras, images, points3D = colmap_model

    # Create reverse index
    point_to_images = reverse_index_points3D(points3D)

    # Find images where points appear
    image_point_counts: Dict[int, int] = {}
    for point_id in set_of_point3DIds:
        if point_id in point_to_images:
            for image_id in point_to_images[point_id]:
                image_point_counts[image_id] = image_point_counts.get(image_id, 0) + 1

    # Filter images by minimum fraction
    total_points = len(set_of_point3DIds)
    min_visible = int(min_fraction * total_points)

    valid_images = {
        img_id: count
        for img_id, count in image_point_counts.items()
        if count >= min_visible
    }

    if not valid_images:
        return {}

    # Project bounding box corners to each valid image
    result = {}

    for image_id, visible_count in valid_images.items():
        image = images[image_id]
        camera = cameras[image.camera_id]
        image_name = image.name

        # Project all 8 corners
        corners_2d = []
        valid_corners_2d = []

        for corner_3d in bbox_corners_3d:
            projected = project_point_to_image(corner_3d, image, camera)
            corners_2d.append(projected.tolist() if projected is not None else None)
            if projected is not None:
                valid_corners_2d.append(projected)

        # If at least some corners are visible, compute 2D bounding box
        if valid_corners_2d:
            valid_corners_array = np.array(valid_corners_2d)
            x_min = float(valid_corners_array[:, 0].min())
            y_min = float(valid_corners_array[:, 1].min())
            x_max = float(valid_corners_array[:, 0].max())
            y_max = float(valid_corners_array[:, 1].max())

            # Clamp to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(camera.width, x_max)
            y_max = min(camera.height, y_max)

            result[image_name] = {
                "image_id": int(image_id),
                "bbox_2d": [x_min, y_min, x_max, y_max],
                "corners_2d": corners_2d,
                "visible_points": visible_count,
                "total_points": total_points,
                "fraction_visible": visible_count / total_points,
                "image_width": camera.width,
                "image_height": camera.height,
            }

    return result


def project_all_bboxes_cli(dataset_name: str, min_fraction: float = 0.3) -> None:
    """
    Project all 3D bounding boxes onto images and save crop coordinates.

    Args:
        dataset_name: Name of the dataset
        min_fraction: Minimum fraction of 3D points that must be visible in an image
    """
    import json
    from .io_paths import get_colmap_model_dir, get_outputs_dir, load_config

    # Load configuration
    config = load_config(dataset_name)

    outputs_dir = get_outputs_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    print(f"Outputs directory: {outputs_dir}")
    print(f"COLMAP model directory: {colmap_model_dir}")
    print(f"Minimum fraction threshold: {min_fraction}")

    # Load COLMAP model
    print(f"\nLoading COLMAP model from: {colmap_model_dir}")
    colmap_model = load_colmap_model(colmap_model_dir)

    # Load connected components
    components_path = outputs_dir / "connected_components.json"
    if not components_path.exists():
        raise FileNotFoundError(
            f"Connected components file not found: {components_path}\n"
            "Please run src.mask_graph first."
        )

    print(f"\nLoading connected components from: {components_path}")
    with components_path.open("r", encoding="utf-8") as f:
        connected_components = json.load(f)

    # Load bounding box corners
    bbox_corners_path = outputs_dir / "bbox_corners.json"
    if not bbox_corners_path.exists():
        raise FileNotFoundError(
            f"Bounding box corners file not found: {bbox_corners_path}\n"
            "Please run src.bbox_corners first."
        )

    print(f"Loading bounding box corners from: {bbox_corners_path}")
    with bbox_corners_path.open("r", encoding="utf-8") as f:
        bbox_corners_data = json.load(f)

    print(f"Found {len(connected_components)} connected components")
    print(f"Found {len(bbox_corners_data)} bounding boxes")

    # Create lookup for bbox corners by component ID
    bbox_lookup = {
        bbox["connected_comp_id"]: bbox["bbox"]["corners"] for bbox in bbox_corners_data
    }

    # Project each component's bounding box
    all_image_crops = {}

    for idx, component in enumerate(connected_components):
        comp_id = component["connected_comp_id"]
        point3D_ids = component["set_of_point3DIds"]

        print(
            f"\n[{idx}/{len(connected_components)}] Processing component {comp_id} ({len(point3D_ids)} points)..."
        )

        # Get bbox corners for this component
        if comp_id not in bbox_lookup:
            print(f"  Warning: No bounding box found for component {comp_id}, skipping")
            continue

        bbox_corners_3d = np.array(bbox_lookup[comp_id])

        try:
            # Project to images
            image_projections = bbox_3d_to_2d(
                bbox_corners_3d,
                point3D_ids,
                colmap_model,
                min_fraction=min_fraction,
            )

            print(f"  Projected to {len(image_projections)} images")

            # Structure results by component ID
            component_crops = []
            for image_name, projection_data in image_projections.items():
                component_crops.append(
                    {
                        "image_name": image_name,
                        "crop_coordinates": projection_data["bbox_2d"],
                        "image_id": projection_data["image_id"],
                        "corners_2d": projection_data["corners_2d"],
                        "visible_points": projection_data["visible_points"],
                        "total_points": projection_data["total_points"],
                        "fraction_visible": projection_data["fraction_visible"],
                        "image_width": projection_data["image_width"],
                        "image_height": projection_data["image_height"],
                    }
                )

            all_image_crops[str(comp_id)] = component_crops

        except Exception as e:
            print(f"  Error projecting component {comp_id}: {e}")
            continue

    # Save results
    output_path = outputs_dir / "image_crop_coordinates.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_image_crops, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved crop coordinates to: {output_path}")
    print(f"Total components with crops: {len(all_image_crops)}")

    # Print statistics
    total_crops = sum(len(crops) for crops in all_image_crops.values())
    print(f"Total crop regions: {total_crops}")

    if all_image_crops:
        crops_per_component = [len(crops) for crops in all_image_crops.values()]
        print(
            f"Average crops per component: {sum(crops_per_component) / len(crops_per_component):.2f}"
        )
        print(f"Max crops in single component: {max(crops_per_component)}")

    # Save summary statistics
    stats = {
        "total_components": len(all_image_crops),
        "total_crops": total_crops,
        "min_fraction_threshold": min_fraction,
    }

    if all_image_crops:
        crops_per_component = [len(crops) for crops in all_image_crops.values()]
        stats["crops_per_component"] = {
            "min": min(crops_per_component),
            "max": max(crops_per_component),
            "mean": sum(crops_per_component) / len(crops_per_component),
        }

    stats_path = outputs_dir / "crop_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


def main() -> None:
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Project 3D bounding boxes onto images"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.3,
        help="Minimum fraction of 3D points that must be visible (default: 0.3)",
    )

    args = parser.parse_args()

    if args.min_fraction <= 0 or args.min_fraction > 1:
        parser.error("min-fraction must be between 0 and 1")

    project_all_bboxes_cli(
        dataset_name=args.dataset_name, min_fraction=args.min_fraction
    )


if __name__ == "__main__":
    main()
