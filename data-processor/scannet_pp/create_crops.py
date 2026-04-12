from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment3d.src.utils.read_write_model import qvec2rotmat, read_model

PixelObservation = tuple[int, int, int]


def associate_2d_3d(
    image_paths: Sequence[str | Path],
    depth_paths: Sequence[str | Path],
    poses: Sequence[npt.ArrayLike],
    intrinsics: npt.ArrayLike,
    ply_path: str | Path,
    depth_tolerance: float = 0.05,
    depth_relative_tolerance: float = 0.02,
) -> dict[int, set[PixelObservation]]:
    """
    Associate mesh vertices with image pixels across a sequence of frames.

    Args:
        image_paths: Ordered list of RGB image paths.
        depth_paths: Ordered list of single-channel ``uint16`` depth PNG paths.
            Each depth map must correspond to the image at the same index in
            ``image_paths``. Depth values are interpreted in millimeters and
            ``0`` denotes invalid depth.
        poses: Camera-to-world pose for each image as a sequence of 4x4 matrices.
            The order must match ``image_paths``.
        intrinsics: Camera intrinsics as a 4x4 matrix. Only ``fx``, ``fy``,
            ``cx``, and ``cy`` are used from the top-left 3x3 block.
        ply_path: Mesh path in world coordinates.
        depth_tolerance: Absolute depth threshold, in the same units as the
            mesh and poses, used for the visibility test.
        depth_relative_tolerance: Relative depth threshold used in addition to
            ``depth_tolerance`` for the visibility test.

    Returns:
        A dictionary mapping each vertex index to the set of corresponding pixel
        observations. Each observation is a tuple ``(image_id, u, v)`` where
        ``image_id`` is the zero-based image index and ``(u, v)`` is the pixel.

    Notes:
        A vertex is kept for an image only if it is in front of the camera,
        projects inside the image bounds, and passes the depth visibility test.
        Mesh/pose units are assumed to be meters, so depth PNG values are
        converted from millimeters to meters before comparison.
        Vertices with no valid observations are omitted from the output.
    """
    image_paths = [Path(path).resolve() for path in image_paths]
    depth_paths = [Path(path).resolve() for path in depth_paths]
    ply_path = Path(ply_path).resolve()

    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    if len(depth_paths) != len(image_paths):
        raise ValueError(
            "Number of depth paths must match number of images: "
            f"{len(depth_paths)} vs {len(image_paths)}"
        )
    if len(poses) != len(image_paths):
        raise ValueError(
            "Number of poses must match number of images: "
            f"{len(poses)} vs {len(image_paths)}"
        )
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
    for depth_path in depth_paths:
        if not depth_path.is_file():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        if depth_path.suffix.lower() != ".png":
            raise ValueError(f"Depth file must be a PNG path, got: {depth_path}")

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    vertices_world = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices_world.size == 0:
        raise ValueError(f"No mesh vertices found in {ply_path}")

    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (4, 4):
        raise ValueError(
            f"Expected intrinsics to have shape (4, 4), got {intrinsics.shape}"
        )

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    sample_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if sample_image is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")
    image_height, image_width = sample_image.shape[:2]

    associations: defaultdict[int, set[PixelObservation]] = defaultdict(set)

    for image_id, (image_path, depth_path, pose) in enumerate(
        tqdm(
            zip(image_paths, depth_paths, poses),
            total=len(image_paths),
            desc="Associating 2D-3D",
            unit="image",
        )
    ):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(
                f"Expected pose {image_id} to have shape (4, 4), got {pose.shape}"
            )

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not read depth PNG: {depth_path}")
        if depth.ndim != 2:
            raise ValueError(
                f"Expected single-channel depth PNG for {depth_path}, got shape {depth.shape}"
            )
        if depth.dtype != np.uint16:
            raise ValueError(
                f"Expected uint16 depth PNG for {depth_path}, got dtype {depth.dtype}"
            )

        depth = depth.astype(np.float32) / 1000.0

        depth_height, depth_width = depth.shape[:2]

        rotation_c2w = pose[:3, :3]
        translation_c2w = pose[:3, 3]
        rotation_w2c = rotation_c2w.T
        translation_w2c = -rotation_w2c @ translation_c2w

        vertices_cam = vertices_world @ rotation_w2c.T + translation_w2c
        z = vertices_cam[:, 2]
        in_front = np.isfinite(z) & (z > 1e-6)
        if not np.any(in_front):
            continue

        safe_z = z.copy()
        safe_z[~in_front] = 1.0
        u = fx * (vertices_cam[:, 0] / safe_z) + cx
        v = fy * (vertices_cam[:, 1] / safe_z) + cy

        u_pix = np.rint(u).astype(np.int32)
        v_pix = np.rint(v).astype(np.int32)

        inside_image = (
            (u_pix >= 0) & (u_pix < image_width) & (v_pix >= 0) & (v_pix < image_height)
        )
        visible = in_front & inside_image
        if not np.any(visible):
            continue

        u_depth = np.rint(
            (u_pix.astype(np.float64) + 0.5) * depth_width / image_width - 0.5
        ).astype(np.int32)
        v_depth = np.rint(
            (v_pix.astype(np.float64) + 0.5) * depth_height / image_height - 0.5
        ).astype(np.int32)

        inside_depth = (
            (u_depth >= 0)
            & (u_depth < depth_width)
            & (v_depth >= 0)
            & (v_depth < depth_height)
        )
        visible &= inside_depth
        if not np.any(visible):
            continue

        visible_indices = np.flatnonzero(visible)
        sampled_depth = depth[v_depth[visible_indices], u_depth[visible_indices]]
        vertex_depth = z[visible_indices]

        valid_depth = np.isfinite(sampled_depth) & (sampled_depth > 0)
        tolerance = np.maximum(
            depth_tolerance,
            depth_relative_tolerance * sampled_depth,
        )
        depth_consistent = valid_depth & (
            np.abs(sampled_depth - vertex_depth) <= tolerance
        )
        final_indices = visible_indices[depth_consistent]

        for vertex_idx in final_indices.tolist():
            associations[vertex_idx].add(
                (image_id, int(u_pix[vertex_idx]), int(v_pix[vertex_idx]))
            )

    return dict(associations)


def get_crops(
    image_paths: Sequence[str | Path],
    depth_paths: Sequence[str | Path],
    poses: Sequence[npt.ArrayLike],
    intrinsics: npt.ArrayLike,
    ply_path: str | Path,
    semantic_ply_path: str | Path,
    segments_json_path: str | Path,
    segments_anno_json_path: str | Path,
    output_dir: str | Path,
    depth_tolerance: float = 0.05,
    depth_relative_tolerance: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """
    Generate per-object image crops from 3D-to-2D vertex associations.

    This function first calls :func:`associate_2d_3d` on the mesh in
    ``ply_path``. It then uses ``segments.json`` and ``segments_anno.json`` to
    group vertices into the same ``connected_comp_id`` values used by
    [create_bboxes.py](/home/sagar/Repos/openFLAME-repos/scan-to-map/data-processor/scannet_pp/create_bboxes.py),
    namely the enumeration index of each entry in ``segGroups``.

    Args:
        image_paths: Ordered list of RGB image paths.
        depth_paths: Ordered list of single-channel ``uint16`` depth PNG paths,
            aligned with ``image_paths``.
        poses: Ordered camera-to-world 4x4 pose matrices, aligned with
            ``image_paths``.
        intrinsics: Camera intrinsics as a 4x4 matrix.
        ply_path: Mesh path used for 3D-to-2D association.
        semantic_ply_path: Path to ``mesh_aligned_0.05_semantic.ply``. It is
            used to validate vertex-count consistency with ``segments.json``.
        segments_json_path: Path to ``segments.json``.
        segments_anno_json_path: Path to ``segments_anno.json``.
        output_dir: Directory under which crops are saved to ``output_dir/crops``.
        depth_tolerance: Absolute depth threshold in meters for visibility.
        depth_relative_tolerance: Relative depth threshold for visibility.

    Returns:
        A manifest dictionary matching the existing crops format. The top-level
        keys are stringified component IDs. Each value contains
        ``component_id``, ``total_crops``, and ``crops``.

    Notes:
        Crops are saved under ``<output_dir>/crops/component_<id>/``.
        Groups without any visible associated points are skipped.
    """
    semantic_ply_path = Path(semantic_ply_path).resolve()
    segments_json_path = Path(segments_json_path).resolve()
    segments_anno_json_path = Path(segments_anno_json_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not semantic_ply_path.is_file():
        raise FileNotFoundError(f"Semantic PLY file not found: {semantic_ply_path}")
    if not segments_json_path.is_file():
        raise FileNotFoundError(f"segments.json not found: {segments_json_path}")
    if not segments_anno_json_path.is_file():
        raise FileNotFoundError(
            f"segments_anno.json not found: {segments_anno_json_path}"
        )

    associations = associate_2d_3d(
        image_paths=image_paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsics=intrinsics,
        ply_path=ply_path,
        depth_tolerance=depth_tolerance,
        depth_relative_tolerance=depth_relative_tolerance,
    )

    semantic_mesh = o3d.io.read_triangle_mesh(str(semantic_ply_path))
    semantic_vertices = np.asarray(semantic_mesh.vertices, dtype=np.float64)
    if semantic_vertices.size == 0:
        raise ValueError(f"No mesh vertices found in {semantic_ply_path}")

    with segments_json_path.open("r", encoding="utf-8") as f:
        segments_data = json.load(f)
    with segments_anno_json_path.open("r", encoding="utf-8") as f:
        segments_anno_data = json.load(f)

    seg_indices = np.asarray(segments_data["segIndices"], dtype=np.int64)
    seg_groups = (
        segments_anno_data["segGroups"]
        if isinstance(segments_anno_data, dict)
        else segments_anno_data
    )

    num_vertices = semantic_vertices.shape[0]
    if seg_indices.shape[0] != num_vertices:
        raise ValueError(
            "segments.json and semantic mesh disagree on vertex count: "
            f"{seg_indices.shape[0]} vs {num_vertices}"
        )

    max_segment_id = int(seg_indices.max(initial=-1))
    for group in seg_groups:
        segments = group.get("segments", [])
        if segments:
            max_segment_id = max(max_segment_id, int(max(segments)))

    segment_to_component = np.full(max_segment_id + 1, -1, dtype=np.int32)
    for connected_comp_id, group in enumerate(seg_groups):
        group_segments = np.asarray(group.get("segments", []), dtype=np.int64)
        if group_segments.size == 0:
            continue

        segment_to_component[group_segments] = connected_comp_id

    component_for_vertex = np.full(num_vertices, -1, dtype=np.int32)
    valid_segment_mask = (seg_indices >= 0) & (seg_indices <= max_segment_id)
    component_for_vertex[valid_segment_mask] = segment_to_component[
        seg_indices[valid_segment_mask]
    ]
    num_vertices_by_component = np.bincount(
        component_for_vertex[component_for_vertex >= 0],
        minlength=len(seg_groups),
    )

    image_paths = [Path(path).resolve() for path in image_paths]
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    points_by_component_and_image: defaultdict[
        tuple[int, int], list[tuple[int, int]]
    ] = defaultdict(list)
    for vertex_idx, observation_set in associations.items():
        if vertex_idx < 0 or vertex_idx >= num_vertices:
            raise ValueError(
                f"Vertex index {vertex_idx} is out of bounds for {semantic_ply_path}"
            )

        connected_comp_id = int(component_for_vertex[vertex_idx])
        if connected_comp_id < 0:
            continue

        for image_id, u, v in observation_set:
            points_by_component_and_image[(connected_comp_id, image_id)].append((u, v))

    crop_records: dict[str, dict[str, Any]] = {}
    selected_records_by_component: dict[int, list[dict[str, Any]]] = {}
    records_by_image_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)

    for connected_comp_id, group in enumerate(
        tqdm(seg_groups, desc="Preparing crops", unit="component")
    ):
        component_records: list[dict[str, Any]] = []
        component_dir = crops_dir / f"component_{connected_comp_id}"
        group_num_vertices = int(num_vertices_by_component[connected_comp_id])
        if group_num_vertices <= 0:
            continue

        for image_id in range(len(image_paths)):
            points_2d = points_by_component_and_image.get((connected_comp_id, image_id))
            if not points_2d:
                continue

            points_array = np.asarray(points_2d, dtype=np.int32)
            x_min = int(points_array[:, 0].min())
            y_min = int(points_array[:, 1].min())
            x_max = int(points_array[:, 0].max())
            y_max = int(points_array[:, 1].max())
            crop_area = int((x_max - x_min + 1) * (y_max - y_min + 1))
            num_points = int(points_array.shape[0])

            record = {
                "connected_comp_id": connected_comp_id,
                "image_id": image_id,
                "source_image": image_paths[image_id].name,
                "crop_coordinates": [
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max),
                ],
                "visible_points": num_points,
                "total_points": group_num_vertices,
                "fraction_visible": num_points / group_num_vertices,
                "crop_area": crop_area,
            }
            component_records.append(record)

        if component_records:
            component_records.sort(
                key=lambda record: (
                    -record["fraction_visible"],
                    -record["crop_area"],
                    record["image_id"],
                )
            )
            component_records = component_records[:5]
            component_dir.mkdir(parents=True, exist_ok=True)
            for crop_index, record in enumerate(component_records):
                crop_filename = (
                    f"{Path(record['source_image']).stem}_crop{crop_index:03d}.jpg"
                )
                record["crop_index"] = crop_index
                record["crop_filename"] = crop_filename
                record["crop_path"] = str(component_dir / crop_filename)
                records_by_image_id[record["image_id"]].append(record)
            selected_records_by_component[connected_comp_id] = component_records

    for image_id, records in tqdm(
        records_by_image_id.items(),
        total=len(records_by_image_id),
        desc="Saving crops",
        unit="image",
    ):
        image = cv2.imread(str(image_paths[image_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_paths[image_id]}")

        image_height, image_width = image.shape[:2]
        for record in records:
            x_min, y_min, x_max, y_max = record["crop_coordinates"]
            x_min = max(0, min(int(round(x_min)), image_width - 1))
            y_min = max(0, min(int(round(y_min)), image_height - 1))
            x_max = max(0, min(int(round(x_max)), image_width - 1))
            y_max = max(0, min(int(round(y_max)), image_height - 1))

            if x_max < x_min or y_max < y_min:
                continue

            crop = image[y_min : y_max + 1, x_min : x_max + 1]
            if crop.size == 0:
                continue

            cv2.imwrite(record["crop_path"], crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            record["crop_coordinates"] = [
                float(x_min),
                float(y_min),
                float(x_max),
                float(y_max),
            ]

    for connected_comp_id, component_records in selected_records_by_component.items():
        crop_records[str(connected_comp_id)] = {
            "component_id": connected_comp_id,
            "total_crops": len(component_records),
            "crops": [
                {
                    "crop_filename": record["crop_filename"],
                    "source_image": record["source_image"],
                    "crop_index": record["crop_index"],
                    "crop_coordinates": record["crop_coordinates"],
                    "image_id": record["image_id"],
                    "fraction_visible": record["fraction_visible"],
                    "visible_points": record["visible_points"],
                    "total_points": record["total_points"],
                }
                for record in component_records
            ],
        }

    return crop_records


def _build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "scan_id",
        type=str,
        help='ScanNet++ scan ID, for example "09c1414f1b".',
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help=(
            "Root ScanNet++ data directory, for example "
            '"/home/sagar/Repos/open-datasets/ScanNetPP/data/data".'
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help=f"Directory under which scan outputs are written (default: {REPO_ROOT / 'outputs'})",
    )
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.05,
        help="Absolute depth tolerance in meters for visibility filtering.",
    )
    parser.add_argument(
        "--depth-relative-tolerance",
        type=float,
        default=0.02,
        help="Relative depth tolerance for visibility filtering.",
    )
    return parser


def _intrinsics_from_colmap_camera(camera: Any) -> np.ndarray:
    intrinsics = np.eye(4, dtype=np.float64)
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
    }:
        f, cx, cy = camera.params[:3]
        fx = f
        fy = f
    elif camera.model in {"RADIAL", "RADIAL_FISHEYE", "FOV"}:
        f, cx, cy = camera.params[:3]
        fx = f
        fy = f
    else:
        raise NotImplementedError(
            f"Unsupported COLMAP camera model for intrinsics extraction: {camera.model}"
        )
    intrinsics[0, 0] = float(fx)
    intrinsics[1, 1] = float(fy)
    intrinsics[0, 2] = float(cx)
    intrinsics[1, 2] = float(cy)
    return intrinsics


def _write_crop_records(
    crop_records: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    metadata_path = output_dir / "crops" / "manifest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(crop_records, f, indent=2)

    print(f"Saved crops to {output_dir / 'crops'}")
    print(f"Saved crop metadata to {metadata_path}")


def _load_iphone_pose_data(
    pose_json_path: Path,
    image_paths: Sequence[Path],
) -> tuple[list[np.ndarray], np.ndarray]:
    with pose_json_path.open("r", encoding="utf-8") as f:
        pose_data = json.load(f)

    if (
        isinstance(pose_data, dict)
        and "intrinsic" in pose_data
        and "aligned_poses" in pose_data
    ):
        intrinsic_3x3 = np.asarray(pose_data["intrinsic"], dtype=np.float64)
        aligned_poses = pose_data["aligned_poses"]
        poses = [
            np.asarray(aligned_poses[idx], dtype=np.float64)
            for idx in range(len(image_paths))
        ]
    elif isinstance(pose_data, dict):
        first_stem = image_paths[0].stem
        if first_stem not in pose_data:
            raise KeyError(
                f"Could not find frame entry {first_stem} in {pose_json_path}"
            )
        intrinsic_3x3 = np.asarray(pose_data[first_stem]["intrinsic"], dtype=np.float64)
        poses = []
        for image_path in image_paths:
            frame_key = image_path.stem
            if frame_key not in pose_data:
                raise KeyError(
                    f"Could not find frame entry {frame_key} in {pose_json_path}"
                )
            poses.append(
                np.asarray(pose_data[frame_key]["aligned_pose"], dtype=np.float64)
            )
    else:
        raise ValueError(f"Unexpected iPhone pose JSON format in {pose_json_path}")

    if intrinsic_3x3.shape != (3, 3):
        raise ValueError(
            f"Expected iPhone intrinsic matrix to have shape (3, 3), got {intrinsic_3x3.shape}"
        )

    intrinsics = np.eye(4, dtype=np.float64)
    intrinsics[:3, :3] = intrinsic_3x3

    for idx, pose in enumerate(poses):
        if pose.shape != (4, 4):
            raise ValueError(
                f"Expected aligned iPhone pose {idx} to have shape (4, 4), got {pose.shape}"
            )

    return poses, intrinsics


def main_dslr(args: argparse.Namespace) -> None:
    scan_dir = args.data_root.resolve() / args.scan_id
    dslr_dir = scan_dir / "dslr"
    scans_dir = scan_dir / "scans"
    images_dir = dslr_dir / "resized_undistorted_images"
    depth_dir = dslr_dir / "render_depth"
    colmap_dir = dslr_dir / "colmap"

    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan directory not found: {scan_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not colmap_dir.is_dir():
        raise FileNotFoundError(f"COLMAP directory not found: {colmap_dir}")
    if not scans_dir.is_dir():
        raise FileNotFoundError(f"Scans directory not found: {scans_dir}")

    ply_path = scans_dir / "mesh_aligned_0.05.ply"
    semantic_ply_path = scans_dir / "mesh_aligned_0.05_semantic.ply"
    segments_json_path = scans_dir / "segments.json"
    segments_anno_json_path = scans_dir / "segments_anno.json"
    output_dir = args.output_root.resolve() / f"scannetpp_{args.scan_id}"

    cameras, images, _ = read_model(str(colmap_dir))
    if not cameras:
        raise ValueError(f"No cameras found in COLMAP model: {colmap_dir}")
    if not images:
        raise ValueError(f"No images found in COLMAP model: {colmap_dir}")

    ordered_images = sorted(images.values(), key=lambda image: image.name)
    intrinsics = _intrinsics_from_colmap_camera(cameras[ordered_images[0].camera_id])

    image_paths: list[Path] = []
    depth_paths: list[Path] = []
    poses: list[np.ndarray] = []

    for image in ordered_images:
        image_path = images_dir / image.name
        depth_path = depth_dir / f"{Path(image.name).stem}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Depth PNG not found: {depth_path}")

        rotation_w2c = qvec2rotmat(image.qvec)
        translation_w2c = image.tvec
        rotation_c2w = rotation_w2c.T
        translation_c2w = -rotation_c2w @ translation_w2c

        pose_c2w = np.eye(4, dtype=np.float64)
        pose_c2w[:3, :3] = rotation_c2w
        pose_c2w[:3, 3] = translation_c2w

        image_paths.append(image_path)
        depth_paths.append(depth_path)
        poses.append(pose_c2w)

    crop_records = get_crops(
        image_paths=image_paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsics=intrinsics,
        ply_path=ply_path,
        semantic_ply_path=semantic_ply_path,
        segments_json_path=segments_json_path,
        segments_anno_json_path=segments_anno_json_path,
        output_dir=output_dir,
        depth_tolerance=args.depth_tolerance,
        depth_relative_tolerance=args.depth_relative_tolerance,
    )

    _write_crop_records(crop_records, output_dir)


def main_iphone(args: argparse.Namespace) -> None:
    scan_dir = args.data_root.resolve() / args.scan_id
    iphone_dir = scan_dir / "iphone"
    scans_dir = scan_dir / "scans"
    images_dir = iphone_dir / "rgb"
    depth_dir = iphone_dir / "depth"
    pose_json_path = iphone_dir / "pose_intrinsic_imu.json"

    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan directory not found: {scan_dir}")
    if not iphone_dir.is_dir():
        raise FileNotFoundError(f"iPhone directory not found: {iphone_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"iPhone RGB directory not found: {images_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"iPhone depth directory not found: {depth_dir}")
    if not pose_json_path.is_file():
        raise FileNotFoundError(f"iPhone pose JSON not found: {pose_json_path}")
    if not scans_dir.is_dir():
        raise FileNotFoundError(f"Scans directory not found: {scans_dir}")

    ply_path = scans_dir / "mesh_aligned_0.05.ply"
    semantic_ply_path = scans_dir / "mesh_aligned_0.05_semantic.ply"
    segments_json_path = scans_dir / "segments.json"
    segments_anno_json_path = scans_dir / "segments_anno.json"
    output_dir = args.output_root.resolve() / f"scannetpp_{args.scan_id}"

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No iPhone RGB images found in {images_dir}")

    depth_paths: list[Path] = []
    for image_path in image_paths:
        depth_path = depth_dir / f"{image_path.stem}.png"
        if not depth_path.is_file():
            raise FileNotFoundError(f"iPhone depth PNG not found: {depth_path}")
        depth_paths.append(depth_path)

    poses, intrinsics = _load_iphone_pose_data(pose_json_path, image_paths)
    if len(poses) != len(image_paths):
        raise ValueError(
            "Number of iPhone poses must match number of RGB images: "
            f"{len(poses)} vs {len(image_paths)}"
        )

    crop_records = get_crops(
        image_paths=image_paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsics=intrinsics,
        ply_path=ply_path,
        semantic_ply_path=semantic_ply_path,
        segments_json_path=segments_json_path,
        segments_anno_json_path=segments_anno_json_path,
        output_dir=output_dir,
        depth_tolerance=args.depth_tolerance,
        depth_relative_tolerance=args.depth_relative_tolerance,
    )

    _write_crop_records(crop_records, output_dir)


def main(args_list: Sequence[str] | None = None) -> None:
    parser = _build_common_parser(
        "Create ScanNet++ object crops from DSLR or iPhone inputs."
    )
    parser.add_argument(
        "--crop-source",
        type=str,
        choices=("dslr", "iphone"),
        default="dslr",
        help="Input source to use for crops (default: dslr).",
    )
    args = parser.parse_args(args_list)

    if args.crop_source == "iphone":
        main_iphone(args)
        return

    main_dslr(args)


if __name__ == "__main__":
    main()
