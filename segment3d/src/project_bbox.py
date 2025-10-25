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
    colmap_model_dir: str | Path,
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
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))

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
