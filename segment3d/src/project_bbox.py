"""Project 3D bounding boxes onto images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Removed scipy dependency for speed
from .colmap_io import load_colmap_model, reverse_index_points3D
from .io_paths import get_colmap_model_dir, get_outputs_dir, load_config


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Fast NumPy implementation to convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.
    Avoids the overhead of scipy.spatial.transform.Rotation.
    """
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def project_points_vectorized(
    points_3d: np.ndarray, image: Any, camera: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized projection of N points to 2D.

    Args:
        points_3d: (N, 3) array of 3D points
        image: COLMAP Image object
        camera: COLMAP Camera object

    Returns:
        tuple:
            - uv_coords: (N, 2) array of projected coordinates
            - valid_mask: (N,) boolean array indicating points in front of camera
    """
    # 1. Transform to camera coordinates
    # Calculate R once per image call
    R = qvec2rotmat(image.qvec)
    t = image.tvec

    # Vectorized transformation: (N, 3) @ (3, 3).T + (3,)
    points_cam = points_3d @ R.T + t

    # 2. Check depth > 0 (Z-axis)
    valid_mask = points_cam[:, 2] > 1e-5  # Use small epsilon

    # Avoid division by zero for invalid points by replacing Z with 1 temporarily
    # (We will filter them out later using valid_mask)
    z_vals = points_cam[:, 2].copy()
    z_vals[~valid_mask] = 1.0

    # 3. Normalize (Perspective Division)
    points_norm = points_cam[:, :2] / z_vals[:, None]

    # 4. Apply Distortion (Intrinsics)
    u_norm, v_norm = points_norm[:, 0], points_norm[:, 1]
    r2 = u_norm**2 + v_norm**2

    # Initialize output arrays
    u, v = np.zeros_like(u_norm), np.zeros_like(v_norm)

    params = camera.params
    model = camera.model

    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        radial = 1 + k * r2
        u = f * radial * u_norm + cx
        v = f * radial * v_norm + cy

    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        u = fx * u_norm + cx
        v = fy * v_norm + cy

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        radial = 1 + k1 * r2 + k2 * r2**2
        u = f * radial * u_norm + cx
        v = f * radial * v_norm + cy

    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        radial = 1 + k1 * r2 + k2 * r2**2
        # Add tangential distortion
        u = (
            fx
            * (u_norm * radial + 2 * p1 * u_norm * v_norm + p2 * (r2 + 2 * u_norm**2))
            + cx
        )
        v = (
            fy
            * (v_norm * radial + p1 * (r2 + 2 * v_norm**2) + 2 * p2 * u_norm * v_norm)
            + cy
        )

    else:
        # Fallback (assume simple pinhole if params exist)
        if len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        else:
            fx = fy = params[0]
            cx = params[1]
            cy = params[2]
        u = fx * u_norm + cx
        v = fy * v_norm + cy

    coords = np.stack([u, v], axis=1)
    return coords, valid_mask


def process_component_bbox(
    bbox_corners_3d: np.ndarray,
    set_of_point3DIds: List[int],
    colmap_data: Tuple[Dict, Dict, Dict],  # cameras, images, points3D
    point_to_images_index: Dict[int, List[int]],  # Pre-computed index
    min_fraction: float = 0.3,
) -> Dict[str, Any]:
    """
    Process a single component: determine visibility and project corners.
    """
    cameras, images, _ = colmap_data

    # 1. Find visible images (Counting)
    image_point_counts: Dict[int, int] = {}

    # This is still a loop, but point_to_images_index is now passed in,
    # avoiding the massive overhead of recomputing it.
    for point_id in set_of_point3DIds:
        # Using .get() with empty list default is safer/cleaner
        for image_id in point_to_images_index.get(point_id, []):
            image_point_counts[image_id] = image_point_counts.get(image_id, 0) + 1

    # 2. Filter images
    min_visible = int(min_fraction * len(set_of_point3DIds))
    if min_visible == 0:
        min_visible = 1  # Edge case safety

    valid_images = [
        (img_id, count)
        for img_id, count in image_point_counts.items()
        if count >= min_visible
    ]

    if not valid_images:
        return {}

    result = {}

    # 3. Project corners for valid images
    for image_id, visible_count in valid_images:
        image = images[image_id]
        camera = cameras[image.camera_id]

        # Vectorized projection of all 8 corners at once
        projected_coords, valid_mask = project_points_vectorized(
            bbox_corners_3d, image, camera
        )

        # If no corners are in front of the camera, skip
        if not np.any(valid_mask):
            continue

        # Filter only valid corners for bbox calculation
        valid_coords = projected_coords[valid_mask]

        if valid_coords.shape[0] > 0:
            x_min, y_min = valid_coords.min(axis=0)
            x_max, y_max = valid_coords.max(axis=0)

            # Clamp
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(float(camera.width), x_max)
            y_max = min(float(camera.height), y_max)

            # Prepare corners list (None for invalid points)
            corners_2d = []
            for i in range(len(bbox_corners_3d)):
                if valid_mask[i]:
                    corners_2d.append(projected_coords[i].tolist())
                else:
                    corners_2d.append(None)

            result[image.name] = {
                "image_id": int(image_id),
                "bbox_2d": [x_min, y_min, x_max, y_max],
                "corners_2d": corners_2d,
                "visible_points": visible_count,
                "total_points": len(set_of_point3DIds),
                "fraction_visible": visible_count / len(set_of_point3DIds),
                "image_width": camera.width,
                "image_height": camera.height,
            }

    return result


def project_all_bboxes_cli(dataset_name: str, min_fraction: float = 0.3) -> None:
    # Load configuration
    config = load_config(dataset_name)
    outputs_dir = get_outputs_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    print(f"Loading COLMAP model from: {colmap_model_dir}")
    colmap_model = load_colmap_model(colmap_model_dir)
    # Unpack for clarity
    _, _, points3D = colmap_model

    # OPTIMIZATION: Build reverse index ONCE globally
    print("Building global point-to-image index...")
    point_to_images = reverse_index_points3D(points3D)
    print("Index built.")

    # Load Data Files
    components_path = outputs_dir / "connected_components.json"
    bbox_corners_path = outputs_dir / "bbox_corners.json"

    with components_path.open("r", encoding="utf-8") as f:
        connected_components = json.load(f)

    with bbox_corners_path.open("r", encoding="utf-8") as f:
        bbox_corners_data = json.load(f)

    # Create lookup
    bbox_lookup = {
        bbox["connected_comp_id"]: np.array(bbox["bbox"]["corners"])
        for bbox in bbox_corners_data
    }

    all_image_crops = {}
    total_components = len(connected_components)

    print(f"Processing {total_components} components...")

    for idx, component in enumerate(connected_components):
        comp_id = component["connected_comp_id"]
        point3D_ids = component["set_of_point3DIds"]

        if comp_id not in bbox_lookup:
            continue

        bbox_corners_3d = bbox_lookup[comp_id]

        # Progress logging every 100 items
        if idx % 100 == 0:
            print(f"[{idx}/{total_components}] Processing...")

        try:
            image_projections = process_component_bbox(
                bbox_corners_3d,
                point3D_ids,
                colmap_model,
                point_to_images,  # Pass the pre-computed index
                min_fraction=min_fraction,
            )

            if image_projections:
                # Transform dict to list format expected by output
                component_crops = []
                for image_name, data in image_projections.items():
                    # Inject image_name into the dict as requested
                    data["image_name"] = image_name
                    # Rename key for output consistency
                    data["crop_coordinates"] = data.pop("bbox_2d")
                    component_crops.append(data)

                all_image_crops[str(comp_id)] = component_crops

        except Exception as e:
            print(f"Error on component {comp_id}: {e}")
            continue

    # Save results
    output_path = outputs_dir / "image_crop_coordinates.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_image_crops, f, indent=2)

    print(f"Saved to {output_path}")

    # Stats Logic (Simplified for brevity)
    total_crops = sum(len(c) for c in all_image_crops.values())
    print(f"Total crop regions generated: {total_crops}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--min-fraction", type=float, default=0.3)
    args = parser.parse_args()

    project_all_bboxes_cli(args.dataset_name, args.min_fraction)


if __name__ == "__main__":
    main()
