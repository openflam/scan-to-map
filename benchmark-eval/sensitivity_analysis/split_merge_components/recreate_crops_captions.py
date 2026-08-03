"""Reproject perturbed boxes, crop source images, and caption affected components.

The projection equations intentionally mirror ``segment3d/src/project_bbox.py``.
This module keeps the sensitivity preparation CLI lightweight by importing the
COLMAP and captioning stacks only when recreation is explicitly requested.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


@dataclass
class ProjectionContext:
    """Source-scene data needed to project perturbed component boxes."""

    dataset_name: str
    images_dir: Path
    cameras: dict[Any, Any]
    images: dict[Any, Any]
    points3d: dict[Any, Any]
    point_to_images: dict[int, list[int]]
    image_ids_by_name: dict[str, int]
    component_point_ids: dict[int, list[int]] | None
    source_manifest: dict[str, Any]
    min_fraction: float
    max_crops_per_component: int = 5
    component_membership_protocol: str = "component_point_visibility"
    nearest_point_component_ids: set[int] = field(default_factory=set)
    split_visibility_fallback_component_ids: set[int] = field(default_factory=set)

    @property
    def visibility_protocol(self) -> str:
        if self.component_point_ids is not None:
            return self.component_membership_protocol
        return "source_manifest_visibility_fallback"


def _ensure_segment3d_import_path(repo_root: Path) -> None:
    segment3d_dir = str(repo_root / "segment3d")
    if segment3d_dir not in sys.path:
        sys.path.insert(0, segment3d_dir)


def source_data_status(
    repo_root: Path,
    dataset_name: str,
    outputs_dir: Path | None = None,
    scannetpp_data_root: Path | None = None,
) -> dict[str, Any]:
    """Report standard source paths without importing heavyweight dependencies."""
    source_outputs_dir = outputs_dir if outputs_dir is not None else repo_root / "outputs"
    components_path = source_outputs_dir / dataset_name / "connected_components.json"
    bboxes_path = source_outputs_dir / dataset_name / "bbox_corners.json"
    if scannetpp_data_root is not None:
        scene_id = dataset_name.removeprefix("scannetpp_")
        dslr_dir = scannetpp_data_root / scene_id / "dslr"
        images_dir = dslr_dir / "resized_images"
        colmap_dir = dslr_dir / "colmap"
        membership_available = bboxes_path.is_file()
        membership_protocol = "bbox_derived_colmap_point_visibility"
    else:
        dataset_dir = repo_root / "data" / dataset_name
        images_dir = dataset_dir / "ns_data" / "images"
        transformed = dataset_dir / "alignment" / "sfm_reconstruction_transformed"
        default = dataset_dir / "hloc_data" / "sfm_reconstruction"
        colmap_dir = transformed if transformed.is_dir() else default
        membership_available = components_path.is_file()
        membership_protocol = "component_point_visibility"
    return {
        "dataset_name": dataset_name,
        "images_dir": str(images_dir),
        "images_available": images_dir.is_dir(),
        "colmap_model_dir": str(colmap_dir),
        "colmap_available": colmap_dir.is_dir(),
        "connected_components_path": str(components_path),
        "point_membership_available": membership_available,
        "point_membership_protocol": membership_protocol,
    }


def _load_scannetpp_colmap_text(
    colmap_dir: Path,
) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any], dict[int, list[int]]]:
    """Load only camera poses, point coordinates, and tracks from COLMAP text."""
    cameras: dict[int, Any] = {}
    with (colmap_dir / "cameras.txt").open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            camera_id = int(values[0])
            cameras[camera_id] = SimpleNamespace(
                id=camera_id,
                model=values[1],
                width=int(values[2]),
                height=int(values[3]),
                params=[float(value) for value in values[4:]],
            )

    images: dict[int, Any] = {}
    expect_image_record = True
    with (colmap_dir / "images.txt").open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not stripped:
                if not expect_image_record:
                    expect_image_record = True
                continue
            if expect_image_record:
                values = stripped.split(maxsplit=9)
                image_id = int(values[0])
                images[image_id] = SimpleNamespace(
                    id=image_id,
                    qvec=[float(value) for value in values[1:5]],
                    tvec=[float(value) for value in values[5:8]],
                    camera_id=int(values[8]),
                    name=values[9],
                )
            expect_image_record = not expect_image_record

    points3d: dict[int, Any] = {}
    point_to_images: dict[int, list[int]] = {}
    with (colmap_dir / "points3D.txt").open("r", encoding="utf-8") as file:
        for line in file:
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if len(values) < 8:
                continue
            point_id = int(values[0])
            image_ids = [int(value) for value in values[8::2]]
            points3d[point_id] = SimpleNamespace(
                xyz=[float(value) for value in values[1:4]],
                image_ids=image_ids,
            )
            point_to_images[point_id] = image_ids
    return cameras, images, points3d, point_to_images


def _bbox_limits(bbox_record: dict[str, Any]) -> tuple[list[float], list[float]]:
    corners = bbox_record.get("bbox", {}).get("corners")
    if not isinstance(corners, list) or len(corners) != 8:
        raise ValueError("Source bounding box must contain eight corners")
    return (
        [min(float(corner[axis]) for corner in corners) for axis in range(3)],
        [max(float(corner[axis]) for corner in corners) for axis in range(3)],
    )


def _derive_bbox_point_memberships(
    points3d: dict[int, Any],
    source_bbox_map: dict[int, dict[str, Any]],
    required_component_ids: set[int],
) -> tuple[dict[int, list[int]], set[int]]:
    """Approximate component memberships with DSLR sparse points inside its box."""
    import numpy as np

    point_ids = np.fromiter(points3d, dtype=np.int64, count=len(points3d))
    coordinates = np.asarray(
        [points3d[int(point_id)].xyz for point_id in point_ids],
        dtype=np.float64,
    )
    memberships: dict[int, list[int]] = {}
    nearest_point_component_ids: set[int] = set()
    for component_id in sorted(required_component_ids):
        if component_id not in source_bbox_map:
            raise ValueError(f"Bounding box missing for component {component_id}")
        minimum, maximum = _bbox_limits(source_bbox_map[component_id])
        selected = None
        for padding in (0.0, 0.01, 0.025, 0.05, 0.1):
            lower = np.asarray(minimum) - padding
            upper = np.asarray(maximum) + padding
            mask = np.all((coordinates >= lower) & (coordinates <= upper), axis=1)
            selected = point_ids[mask]
            if selected.size:
                break
        if selected is None or not selected.size:
            lower = np.asarray(minimum)
            upper = np.asarray(maximum)
            outside = np.maximum(np.maximum(lower - coordinates, 0), coordinates - upper)
            distances = np.linalg.norm(outside, axis=1)
            nearest_count = min(32, len(point_ids))
            nearest_indices = np.argpartition(distances, nearest_count - 1)[:nearest_count]
            if float(distances[nearest_indices].min()) > 0.5:
                raise ValueError(
                    f"No DSLR COLMAP points within 0.5 m of component {component_id}"
                )
            selected = point_ids[nearest_indices]
            nearest_point_component_ids.add(component_id)
        memberships[component_id] = [int(point_id) for point_id in selected]
    return memberships, nearest_point_component_ids


def load_projection_context(
    repo_root: Path,
    dataset_name: str,
    source_manifest: dict[str, Any],
    min_fraction: float,
    source_scene_dir: Path | None = None,
    source_bbox_map: dict[int, dict[str, Any]] | None = None,
    required_component_ids: set[int] | None = None,
    scannetpp_data_root: Path | None = None,
    max_crops_per_component: int = 5,
) -> ProjectionContext:
    """Load source images, cameras, point visibility, and optional memberships."""
    if scannetpp_data_root is not None:
        scene_id = dataset_name.removeprefix("scannetpp_")
        dslr_dir = scannetpp_data_root / scene_id / "dslr"
        images_dir = dslr_dir / "resized_images"
        colmap_model_dir = dslr_dir / "colmap"
        if not images_dir.is_dir():
            raise NotADirectoryError(f"ScanNet++ DSLR images not found: {images_dir}")
        if not colmap_model_dir.is_dir():
            raise NotADirectoryError(f"ScanNet++ COLMAP model not found: {colmap_model_dir}")
        cameras, images, points3d, point_to_images = _load_scannetpp_colmap_text(
            colmap_model_dir
        )
    else:
        _ensure_segment3d_import_path(repo_root)
        from src.colmap_io import load_colmap_model, reverse_index_points3D
        from src.io_paths import get_colmap_model_dir, get_images_dir, load_config

        config = load_config(dataset_name)
        images_dir = get_images_dir(config)
        colmap_model_dir = get_colmap_model_dir(config)
        cameras, images, points3d = load_colmap_model(str(colmap_model_dir))
        point_to_images = reverse_index_points3D(points3d)

    image_ids_by_name: dict[str, int] = {}
    basename_ids: dict[str, set[int]] = {}
    for image_id, image in images.items():
        numeric_id = int(image_id)
        image_ids_by_name[str(image.name)] = numeric_id
        basename_ids.setdefault(Path(str(image.name)).name, set()).add(numeric_id)
    for basename, image_ids in basename_ids.items():
        if len(image_ids) == 1:
            image_ids_by_name.setdefault(basename, next(iter(image_ids)))

    scene_dir = (
        source_scene_dir
        if source_scene_dir is not None
        else repo_root / "outputs" / dataset_name
    )
    components_path = scene_dir / "connected_components.json"
    component_point_ids: dict[int, list[int]] | None = None
    membership_protocol = "component_point_visibility"
    nearest_point_component_ids: set[int] = set()
    if scannetpp_data_root is not None:
        if source_bbox_map is None or required_component_ids is None:
            raise ValueError(
                "External ScanNet++ projection requires source boxes and component IDs"
            )
        component_point_ids, nearest_point_component_ids = _derive_bbox_point_memberships(
            points3d,
            source_bbox_map,
            required_component_ids,
        )
        membership_protocol = "bbox_derived_colmap_point_visibility"
    elif components_path.is_file():
        with components_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"Connected components must be a list: {components_path}")
        component_point_ids = {}
        for record in records:
            component_id = int(record["connected_comp_id"])
            component_point_ids[component_id] = [
                int(point_id) for point_id in record["set_of_point3DIds"]
            ]

    return ProjectionContext(
        dataset_name=dataset_name,
        images_dir=images_dir,
        cameras=cameras,
        images=images,
        points3d=points3d,
        point_to_images=point_to_images,
        image_ids_by_name=image_ids_by_name,
        component_point_ids=component_point_ids,
        source_manifest=source_manifest,
        min_fraction=min_fraction,
        max_crops_per_component=max_crops_per_component,
        component_membership_protocol=membership_protocol,
        nearest_point_component_ids=nearest_point_component_ids,
    )


def split_component_point_ids(
    context: ProjectionContext,
    source_component_id: int,
    axis: int,
    midpoint: float,
) -> tuple[list[int] | None, list[int] | None]:
    """Partition a source component's points using the same split plane as its box."""
    if context.component_point_ids is None:
        return None, None
    source_ids = context.component_point_ids.get(source_component_id)
    if source_ids is None:
        raise ValueError(
            f"Point membership missing for {context.dataset_name}:{source_component_id}"
        )
    first: list[int] = []
    second: list[int] = []
    for point_id in source_ids:
        point = context.points3d.get(point_id)
        if point is None:
            continue
        coordinate = float(point.xyz[axis])
        (first if coordinate <= midpoint else second).append(point_id)
    if not first or not second:
        context.split_visibility_fallback_component_ids.add(source_component_id)
        if not first:
            first = list(source_ids)
        if not second:
            second = list(source_ids)
    return first, second


def merge_component_point_ids(
    context: ProjectionContext,
    source_component_ids: Sequence[int],
) -> list[int] | None:
    """Union the point memberships of components participating in a merge."""
    if context.component_point_ids is None:
        return None
    merged: set[int] = set()
    for component_id in source_component_ids:
        point_ids = context.component_point_ids.get(component_id)
        if point_ids is None:
            raise ValueError(
                f"Point membership missing for {context.dataset_name}:{component_id}"
            )
        merged.update(point_ids)
    if not merged:
        raise ValueError(
            f"Merged point membership is empty for {context.dataset_name}:"
            f"{list(source_component_ids)}"
        )
    return sorted(merged)


def qvec2rotmat(qvec: Sequence[float]) -> list[list[float]]:
    """Convert a COLMAP quaternion (qw, qx, qy, qz) to a rotation matrix."""
    w, x, y, z = (float(value) for value in qvec)
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]


def project_points(
    points_3d: Sequence[Sequence[float]],
    image: Any,
    camera: Any,
) -> tuple[list[list[float]], list[bool]]:
    """Project 3-D points with the camera models handled by project_bbox.py."""
    rotation = qvec2rotmat(image.qvec)
    translation = [float(value) for value in image.tvec]
    params = [float(value) for value in camera.params]
    model = str(camera.model)
    coordinates: list[list[float]] = []
    valid_mask: list[bool] = []

    for point in points_3d:
        camera_point = [
            sum(rotation[row][column] * float(point[column]) for column in range(3))
            + translation[row]
            for row in range(3)
        ]
        valid = camera_point[2] > 1e-5
        valid_mask.append(valid)
        z_value = camera_point[2] if valid else 1.0
        u_norm = camera_point[0] / z_value
        v_norm = camera_point[1] / z_value
        radius_squared = u_norm * u_norm + v_norm * v_norm

        if model == "SIMPLE_RADIAL":
            focal, cx, cy, k = params
            radial = 1 + k * radius_squared
            u = focal * radial * u_norm + cx
            v = focal * radial * v_norm + cy
        elif model == "PINHOLE":
            fx, fy, cx, cy = params
            u = fx * u_norm + cx
            v = fy * v_norm + cy
        elif model == "RADIAL":
            focal, cx, cy, k1, k2 = params
            radial = 1 + k1 * radius_squared + k2 * radius_squared**2
            u = focal * radial * u_norm + cx
            v = focal * radial * v_norm + cy
        elif model == "OPENCV":
            fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
            radial = 1 + k1 * radius_squared + k2 * radius_squared**2
            u = fx * (
                u_norm * radial
                + 2 * p1 * u_norm * v_norm
                + p2 * (radius_squared + 2 * u_norm**2)
            ) + cx
            v = fy * (
                v_norm * radial
                + p1 * (radius_squared + 2 * v_norm**2)
                + 2 * p2 * u_norm * v_norm
            ) + cy
        elif model == "OPENCV_FISHEYE":
            fx, fy, cx, cy, k1, k2, k3, k4 = params[:8]
            radius = math.sqrt(radius_squared)
            theta = math.atan(radius)
            theta_squared = theta * theta
            theta_distorted = theta * (
                1
                + k1 * theta_squared
                + k2 * theta_squared**2
                + k3 * theta_squared**3
                + k4 * theta_squared**4
            )
            scale = theta_distorted / radius if radius > 1e-12 else 1.0
            u = fx * scale * u_norm + cx
            v = fy * scale * v_norm + cy
        else:
            if len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            else:
                fx = fy = params[0]
                cx, cy = params[1:3]
            u = fx * u_norm + cx
            v = fy * v_norm + cy
        coordinates.append([u, v])

    return coordinates, valid_mask


def _manifest_candidates(
    context: ProjectionContext,
    source_component_ids: Sequence[int],
) -> dict[int, tuple[int, int]]:
    totals: dict[int, int] = {}
    per_source: dict[int, dict[int, int]] = {}
    for component_id in source_component_ids:
        entry = context.source_manifest.get(str(component_id), {})
        crops = entry.get("crops", [])
        source_counts: dict[int, int] = {}
        total = 0
        for crop in crops:
            raw_image_id = crop.get("image_id")
            image_id = int(raw_image_id) if raw_image_id is not None else None
            if image_id not in context.images:
                image_id = context.image_ids_by_name.get(str(crop.get("source_image", "")))
            if image_id is None or image_id not in context.images:
                continue
            visible = int(crop.get("visible_points") or 0)
            source_counts[image_id] = max(source_counts.get(image_id, 0), visible)
            total = max(total, int(crop.get("total_points") or 0))
        if not source_counts:
            raise ValueError(
                f"No source-manifest images match COLMAP for "
                f"{context.dataset_name}:{component_id}"
            )
        totals[component_id] = max(total, 1)
        per_source[component_id] = source_counts

    combined_total = sum(totals.values())
    candidates: dict[int, tuple[int, int]] = {}
    for image_id in set().union(*(set(value) for value in per_source.values())):
        visible = sum(counts.get(image_id, 0) for counts in per_source.values())
        candidates[image_id] = (visible, combined_total)
    return candidates


def _point_visibility_candidates(
    context: ProjectionContext,
    point_ids: Sequence[int],
) -> tuple[dict[int, tuple[int, int]], bool]:
    counts: dict[int, int] = {}
    for point_id in point_ids:
        for image_id in context.point_to_images.get(int(point_id), []):
            counts[int(image_id)] = counts.get(int(image_id), 0) + 1
    minimum_visible = max(1, int(context.min_fraction * len(point_ids)))
    eligible = {
        image_id: (visible, len(point_ids))
        for image_id, visible in counts.items()
        if visible >= minimum_visible
    }
    if eligible:
        return eligible, False
    return (
        {
            image_id: (visible, len(point_ids))
            for image_id, visible in counts.items()
        },
        True,
    )


def project_component_bbox(
    context: ProjectionContext,
    bbox_record: dict[str, Any],
    source_component_ids: Sequence[int],
    point_ids: Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Project a perturbed box into valid source views."""
    corners = bbox_record.get("bbox", {}).get("corners")
    if not isinstance(corners, list) or len(corners) != 8:
        raise ValueError("Perturbed bounding box must contain eight corners")
    if point_ids is not None:
        candidates, threshold_fallback = _point_visibility_candidates(
            context,
            point_ids,
        )
    else:
        candidates = _manifest_candidates(context, source_component_ids)
        threshold_fallback = False
    visibility_protocol = (
        "component_point_visibility"
        if point_ids is not None
        else "source_manifest_visibility_fallback"
    )
    if point_ids is not None:
        visibility_protocol = context.component_membership_protocol
    if set(source_component_ids) & context.nearest_point_component_ids:
        visibility_protocol += "_nearest_point_fallback"
    if set(source_component_ids) & context.split_visibility_fallback_component_ids:
        visibility_protocol += "_split_partition_fallback"
    if threshold_fallback:
        visibility_protocol += "_below_threshold_best_views"
    projections: list[dict[str, Any]] = []
    for image_id, (visible_points, total_points) in sorted(candidates.items()):
        image = context.images.get(image_id)
        if image is None:
            continue
        camera = context.cameras[image.camera_id]
        coordinates, valid_mask = project_points(corners, image, camera)
        valid_coordinates = [
            coordinate for coordinate, valid in zip(coordinates, valid_mask) if valid
        ]
        if not valid_coordinates:
            continue
        x_min = max(0.0, min(coordinate[0] for coordinate in valid_coordinates))
        y_min = max(0.0, min(coordinate[1] for coordinate in valid_coordinates))
        x_max = min(float(camera.width), max(coordinate[0] for coordinate in valid_coordinates))
        y_max = min(float(camera.height), max(coordinate[1] for coordinate in valid_coordinates))
        if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        projections.append(
            {
                "image_name": str(image.name),
                "image_id": int(image_id),
                "crop_coordinates": [x_min, y_min, x_max, y_max],
                "corners_2d": [
                    coordinate if valid else None
                    for coordinate, valid in zip(coordinates, valid_mask)
                ],
                "visible_points": visible_points,
                "total_points": total_points,
                "fraction_visible": visible_points / total_points,
                "image_width": int(camera.width),
                "image_height": int(camera.height),
                "visibility_protocol": visibility_protocol,
            }
        )
    projections.sort(
        key=lambda record: (
            -int(record["visible_points"]),
            -float(record["fraction_visible"]),
            record["image_name"],
        )
    )
    if not projections:
        raise ValueError(
            f"No valid source-image projections for {context.dataset_name}:"
            f"{list(source_component_ids)}"
        )
    return projections[: context.max_crops_per_component]


def _safe_source_image(images_dir: Path, image_name: str) -> Path:
    source_path = (images_dir / image_name).resolve()
    try:
        source_path.relative_to(images_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"COLMAP image name escapes the image directory: {image_name}") from exc
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    return source_path


def crop_component_projections(
    context: ProjectionContext,
    target_crops_dir: Path,
    target_component_id: int,
    projections: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Create fresh JPEG crops matching segment3d/src/crop_images.py semantics."""
    from PIL import Image

    component_dir = target_crops_dir / f"component_{target_component_id}"
    component_dir.mkdir()
    manifest_crops: list[dict[str, Any]] = []
    for crop_index, projection in enumerate(projections):
        source_path = _safe_source_image(context.images_dir, projection["image_name"])
        with Image.open(source_path) as image:
            width, height = image.size
            x_min, y_min, x_max, y_max = (
                int(round(value)) for value in projection["crop_coordinates"]
            )
            if x_max <= x_min or y_max <= y_min:
                continue
            x_min = min(max(x_min, 0), width - 1)
            y_min = min(max(y_min, 0), height - 1)
            x_max = min(max(x_max, x_min + 1), width)
            y_max = min(max(y_max, y_min + 1), height)
            cropped = image.crop((x_min, y_min, x_max, y_max))
            if cropped.mode not in ("RGB", "L"):
                cropped = cropped.convert("RGB")
            output_name = f"{source_path.stem}_crop{crop_index:03d}.jpg"
            cropped.save(component_dir / output_name, format="JPEG")

        manifest_crops.append(
            {
                "crop_filename": output_name,
                "source_image": projection["image_name"],
                "crop_index": crop_index,
                "crop_coordinates": [x_min, y_min, x_max, y_max],
                "image_id": projection["image_id"],
                "fraction_visible": projection["fraction_visible"],
                "visible_points": projection["visible_points"],
                "total_points": projection["total_points"],
                "visibility_protocol": projection["visibility_protocol"],
            }
        )
    if not manifest_crops:
        raise ValueError(
            f"No crops were created for {context.dataset_name}:{target_component_id}"
        )
    return {
        "component_id": target_component_id,
        "total_crops": len(manifest_crops),
        "crops": manifest_crops,
    }


def recreate_component_crops(
    context: ProjectionContext,
    target_crops_dir: Path,
    target_component_id: int,
    bbox_record: dict[str, Any],
    source_component_ids: Sequence[int],
    point_ids: Sequence[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project one perturbed component and create all of its crop images."""
    projections = project_component_bbox(
        context,
        bbox_record,
        source_component_ids,
        point_ids,
    )
    manifest = crop_component_projections(
        context,
        target_crops_dir,
        target_component_id,
        projections,
    )
    return manifest, projections


def create_captioner(
    repo_root: Path,
    captioner_type: str,
    model: str,
    device: int,
) -> Any:
    """Create a captioner through segment3d's captioning factory."""
    _ensure_segment3d_import_path(repo_root)
    from src.captioning.captioner_base import create_captioner as factory

    return factory(captioner_type=captioner_type, model=model, device=device)


def caption_recreated_components(
    repo_root: Path,
    captioner: Any,
    crops_dir: Path,
    manifests: dict[int, dict[str, Any]],
    component_ids: Iterable[int],
    n_images: int,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    """Caption only affected components using segment3d's ranking and captioner API."""
    _ensure_segment3d_import_path(repo_root)
    from src.captioning.orchestrator import get_top_images

    manifest_data = {str(component_id): manifests[component_id] for component_id in manifests}
    ordered_ids = sorted(set(component_ids))
    captions: dict[int, dict[str, Any]] = {}
    for batch_start in range(0, len(ordered_ids), batch_size):
        batch_ids = ordered_ids[batch_start : batch_start + batch_size]
        metadata = {
            component_id: get_top_images(component_id, manifest_data, n=n_images)
            for component_id in batch_ids
        }
        results = captioner.caption_batch(
            [(component_id, metadata[component_id]) for component_id in batch_ids],
            crops_dir,
        )
        for result in results:
            component_id = int(result.component_id)
            if result.error:
                raise RuntimeError(
                    f"Captioning failed for component {component_id}: {result.error}"
                )
            top_images = metadata[component_id]
            captions[component_id] = {
                "component_id": component_id,
                "caption": result.caption,
                "num_images_used": len(top_images),
                "crop_filenames": [image["crop_filename"] for image in top_images],
            }
    missing = set(ordered_ids) - set(captions)
    if missing:
        raise RuntimeError(f"Captioner returned no result for components {sorted(missing)}")
    return captions
