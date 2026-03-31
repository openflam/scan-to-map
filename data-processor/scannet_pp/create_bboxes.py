from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment3d.src.utils.read_write_model import read_model


DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def _resolve_data_dir(scene_dir: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidate = Path(scene_dir).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (data_root / scene_dir).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mesh_xyz(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh_xyz = np.asarray(mesh.vertices)
    if mesh_xyz.size == 0:
        raise ValueError(f"Open3D could not read any mesh vertices from {path}")
    return mesh_xyz


def _load_colmap_points(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    _, _, points3D = read_model(str(model_dir))
    if not points3D:
        raise ValueError(f"COLMAP model at {model_dir} contains no 3D points")

    point_ids = np.fromiter(points3D.keys(), dtype=np.int64, count=len(points3D))
    point_xyz = np.asarray(
        [points3D[point_id].xyz for point_id in point_ids],
        dtype=np.float64,
    )
    return point_ids, point_xyz


def _compute_bbox(points_xyz: np.ndarray) -> dict[str, list[list[float]] | list[float]]:
    bbox_min = points_xyz.min(axis=0)
    bbox_max = points_xyz.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    size = bbox_max - bbox_min

    corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=np.float64,
    )

    return {
        "corners": corners.tolist(),
        "min": bbox_min.astype(np.float64).tolist(),
        "max": bbox_max.astype(np.float64).tolist(),
        "center": center.astype(np.float64).tolist(),
        "size": size.astype(np.float64).tolist(),
    }


def get_bounding_boxes(
    data_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir).resolve()
    scan_dir = data_dir / "scans"
    colmap_model_dir = data_dir / "dslr" / "colmap"
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan directory not found: {scan_dir}")
    if not colmap_model_dir.is_dir():
        raise FileNotFoundError(f"COLMAP model directory not found: {colmap_model_dir}")

    segments_anno = _load_json(scan_dir / "segments_anno.json")
    segments_data = _load_json(scan_dir / "segments.json")
    mesh_xyz = _load_mesh_xyz(scan_dir / "mesh_aligned_0.05.ply")
    point3d_ids, point3d_xyz = _load_colmap_points(colmap_model_dir)
    seg_indices = np.asarray(segments_data["segIndices"], dtype=np.int64)
    seg_groups = segments_anno["segGroups"]

    num_vertices = mesh_xyz.shape[0]
    if seg_indices.shape[0] != num_vertices:
        raise ValueError(
            "segments.json and mesh_aligned_0.05.ply disagree on vertex count: "
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

        previous = segment_to_component[group_segments]
        overlapping = previous[(previous != -1) & (previous != connected_comp_id)]
        if overlapping.size > 0:
            raise ValueError(
                "A segment was assigned to more than one segGroup in "
                f"{scan_dir / 'segments_anno.json'}"
            )

        segment_to_component[group_segments] = connected_comp_id

    component_for_vertex = np.full(num_vertices, -1, dtype=np.int32)
    valid_segment_mask = (seg_indices >= 0) & (seg_indices <= max_segment_id)
    component_for_vertex[valid_segment_mask] = segment_to_component[
        seg_indices[valid_segment_mask]
    ]

    nearest_point_id_for_vertex = np.full(num_vertices, -1, dtype=np.int64)
    if np.any(valid_segment_mask):
        point_tree = KDTree(point3d_xyz)
        _, nearest_point_idx = point_tree.query(mesh_xyz[valid_segment_mask], workers=-1)
        nearest_point_id_for_vertex[valid_segment_mask] = point3d_ids[
            np.asarray(nearest_point_idx, dtype=np.int64)
        ]

    bbox_results: list[dict[str, Any]] = []
    connected_components_results: list[dict[str, Any]] = []
    for connected_comp_id, group in enumerate(seg_groups):
        vertex_mask = component_for_vertex == connected_comp_id
        if not np.any(vertex_mask):
            print(
                f"Warning: segGroup {group.get('id')} produced no mesh vertices; "
                "skipping"
            )
            continue

        component_xyz = mesh_xyz[vertex_mask]
        component_point3d_ids = np.unique(nearest_point_id_for_vertex[vertex_mask])
        component_point3d_ids = component_point3d_ids[component_point3d_ids >= 0]

        connected_components_results.append(
            {
                "connected_comp_id": connected_comp_id,
                "set_of_point3DIds": component_point3d_ids.astype(np.int64).tolist(),
            }
        )

        bbox_results.append(
            {
                "connected_comp_id": connected_comp_id,
                "num_point3d_ids": int(component_point3d_ids.shape[0]),
                "num_points_used": int(component_xyz.shape[0]),
                "num_filtered": 0,
                "bbox": _compute_bbox(component_xyz),
                "label": group.get("label"),
                "num_vertices": int(component_xyz.shape[0]),
                "num_segments": int(len(group.get("segments", []))),
                "source_seg_group_id": group.get("id"),
                "source_object_id": group.get("objectId"),
                "part_id": group.get("partId"),
                "annotation_index": group.get("index"),
            }
        )

    output_dir = Path(output_root).resolve() / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_output_path = output_dir / "bbox_corners.json"
    components_output_path = output_dir / "connected_components.json"

    with bbox_output_path.open("w", encoding="utf-8") as f:
        json.dump(bbox_results, f, indent=2)
    with components_output_path.open("w", encoding="utf-8") as f:
        json.dump(connected_components_results, f, indent=2)

    print(f"Saved {len(connected_components_results)} connected components to {components_output_path}")
    print(f"Saved {len(bbox_results)} bounding boxes to {bbox_output_path}")
    return bbox_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ScanNet++ scene under data/ into bbox_corners.json for "
            "the scan-to-map pipeline"
        )
    )
    parser.add_argument(
        "scene_dir",
        help=(
            "ScanNet++ scene directory name under data/ "
            "(for example: scannetpp_7b6477cb95) or an absolute path"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base data directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base outputs directory (default: {DEFAULT_OUTPUT_ROOT})",
    )

    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.scene_dir, data_root=args.data_root)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {data_dir}")

    get_bounding_boxes(data_dir=data_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
