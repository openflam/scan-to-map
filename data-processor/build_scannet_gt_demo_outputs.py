#!/usr/bin/env python3
"""
Build scan-to-map *outputs* + *data* layout for ScanNet using **ground-truth**
instance segmentation (aggregation + segs + mesh), so the semantic-3d-search-demo
and search-server can use the same files as ScanNet++ / Polycam pipelines.

What this creates (matches the repo’s expectations):

  outputs/<dataset_name>/
    bbox_corners.json
    component_captions.json
    connected_components.json
    image_crop_coordinates.json
    crops/
      manifest.json
      component_<id>/*.jpg

  data/<dataset_name>/polycam_data/raw.glb   ← mesh for /load_mesh in app.py

Downstream:

  • search-server/spatial_db/create_tables.py reads outputs/<dataset_name>/
    (component_captions.json + bbox_corners.json + crops/manifest.json).
  • search-server/app.py serves the mesh from data/<dataset_name>/polycam_data/raw.glb.

Requires a prepared scan-to-map dataset from data-processor/scannet.py (COLMAP
under data/<dataset_name>/hloc_data/sfm_reconstruction/ and images under
ns_data/images/).

GT ↔ COLMAP association: each COLMAP 3D point is matched to the nearest mesh
vertex (ScanNet *_vh_clean_2.ply); the vertex carries a segment id, which maps
to ScanNet object ids via aggregation.json. This tolerates mesh subsampling
in scannet.py as long as COLMAP points lie on real vertices.

Usage (from scan-to-map/data-processor/):

  python build_scannet_gt_demo_outputs.py /path/to/scans/scene0000_00 \\
      --scan-to-map-root .. \\
      --dataset-name scannet_scene0000_00

Dependencies: numpy, scipy, opencv-python, open3d (for PLY→GLB), and the
segment3d package on PYTHONPATH (this script adds ../segment3d automatically).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# segment3d imports (COLMAP + projection)
_REPO = Path(__file__).resolve().parents[1]
_SEGMENT3D = _REPO / "segment3d"
if str(_SEGMENT3D) not in sys.path:
    sys.path.insert(0, str(_SEGMENT3D))

from src.colmap_io import load_colmap_model, reverse_index_points3D  # noqa: E402
from src.project_bbox import process_component_bbox  # noqa: E402


# ---------------------------------------------------------------------------
# ScanNet GT loading
# ---------------------------------------------------------------------------


def _load_mesh_vertices(ply_path: Path) -> np.ndarray:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    return np.asarray(mesh.vertices, dtype=np.float64)


def _vertex_object_ids(
    scene_dir: Path, scene_id: str, n_verts: int
) -> np.ndarray:
    """
    Per mesh vertex: ScanNet objectId, or -1 if unknown / background.
    """
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)
    with open(segs_path, "r", encoding="utf-8") as f:
        segs = json.load(f)

    seg_indices: List[int] = segs["segIndices"]
    if len(seg_indices) != n_verts:
        raise ValueError(
            f"segIndices length {len(seg_indices)} != mesh vertices {n_verts}"
        )

    seg_to_obj: Dict[int, int] = {}
    for group in agg.get("segGroups", []):
        oid = int(group["objectId"])
        for sid in group["segments"]:
            seg_to_obj[int(sid)] = oid

    out = np.full(n_verts, -1, dtype=np.int32)
    for i, sid in enumerate(seg_indices):
        sid = int(sid)
        if sid in seg_to_obj:
            out[i] = seg_to_obj[sid]
    return out


def _object_vertex_mask(
    scene_dir: Path, scene_id: str, object_id: int, verts: np.ndarray
) -> np.ndarray:
    """Boolean mask over vertices for one GT object."""
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)
    with open(segs_path, "r", encoding="utf-8") as f:
        segs = json.load(f)

    target_segs: Set[int] = set()
    for group in agg.get("segGroups", []):
        if int(group["objectId"]) == object_id:
            target_segs.update(int(s) for s in group["segments"])
    if not target_segs:
        return np.zeros(len(verts), dtype=bool)

    seg_arr = np.array(segs["segIndices"], dtype=np.int32)
    mask = np.zeros(len(seg_arr), dtype=bool)
    for sid in target_segs:
        mask |= seg_arr == sid
    return mask


def _labels_from_aggregation(scene_dir: Path, scene_id: str) -> Dict[int, str]:
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)
    out: Dict[int, str] = {}
    for group in agg.get("segGroups", []):
        oid = int(group["objectId"])
        label = str(group.get("label") or "").strip()
        out[oid] = label
    return out


def _aabb_dict_from_minmax(bbox_min: np.ndarray, bbox_max: np.ndarray) -> Dict[str, Any]:
    mn = bbox_min.astype(float).tolist()
    mx = bbox_max.astype(float).tolist()
    center = ((bbox_min + bbox_max) / 2.0).astype(float).tolist()
    size = (bbox_max - bbox_min).astype(float).tolist()
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
        dtype=float,
    )
    return {
        "corners": corners.tolist(),
        "min": mn,
        "max": mx,
        "center": center,
        "size": size,
    }


def _assign_colmap_points_to_objects(
    points3d: Dict,
    mesh_tree: Any,
    vertex_obj: np.ndarray,
    tol: float,
) -> Dict[int, List[int]]:
    """
    Map each COLMAP point id → ScanNet object id (or skip if no match).

    Returns: object_id → list of COLMAP point3D ids
    """
    by_obj: Dict[int, List[int]] = defaultdict(list)
    for pid, p in points3d.items():
        xyz = np.asarray(p.xyz, dtype=np.float64)
        dist, idx = mesh_tree.query(xyz, k=1)
        if dist > tol:
            continue
        oid = int(vertex_obj[idx])
        if oid < 0:
            continue
        by_obj[oid].append(int(pid))
    return dict(by_obj)


# ---------------------------------------------------------------------------
# Crops + manifest
# ---------------------------------------------------------------------------


def _write_crops_and_manifest(
    image_crop_coordinates: Dict[str, Any],
    images_dir: Path,
    crops_dir: Path,
) -> None:
    import cv2

    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}

    for comp_key, crop_list in image_crop_coordinates.items():
        comp_dir = crops_dir / f"component_{comp_key}"
        comp_dir.mkdir(parents=True, exist_ok=True)
        crops_entries: List[Dict[str, Any]] = []

        for idx, crop_info in enumerate(crop_list):
            image_name = crop_info["image_name"]
            image_path = images_dir / image_name
            coords = crop_info["crop_coordinates"]
            out_name = f"{Path(image_name).stem}_crop{idx:03d}.jpg"
            out_path = comp_dir / out_name

            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            x_min, y_min, x_max, y_max = [int(round(c)) for c in coords]
            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(x_min + 1, min(x_max, w))
            y_max = max(y_min + 1, min(y_max, h))
            crop_img = img[y_min:y_max, x_min:x_max]
            if crop_img.size == 0:
                continue
            cv2.imwrite(str(out_path), crop_img)

            crops_entries.append(
                {
                    "crop_filename": out_name,
                    "source_image": image_name,
                    "crop_index": idx,
                    "crop_coordinates": coords,
                    "image_id": crop_info.get("image_id"),
                    "fraction_visible": crop_info.get("fraction_visible"),
                    "visible_points": crop_info.get("visible_points"),
                    "total_points": crop_info.get("total_points"),
                }
            )

        manifest[comp_key] = {
            "component_id": int(comp_key),
            "total_crops": len(crops_entries),
            "crops": crops_entries,
        }

    with (crops_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def build_outputs(
    scannet_scene_dir: Path,
    scan_to_map_root: Path,
    dataset_name: str,
    position_tol: float = 0.002,
    min_colmap_points: int = 5,
    min_fraction: float = 0.05,
    max_crops_per_component: int = 30,
    max_objects: Optional[int] = None,
    skip_glb: bool = False,
    skip_crops: bool = False,
) -> None:
    scene_id = scannet_scene_dir.name
    ply_path = scannet_scene_dir / f"{scene_id}_vh_clean_2.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing mesh: {ply_path}")

    data_dir = scan_to_map_root / "data" / dataset_name
    outputs_dir = scan_to_map_root / "outputs" / dataset_name
    colmap_dir = data_dir / "hloc_data" / "sfm_reconstruction"
    images_dir = data_dir / "ns_data" / "images"

    if not colmap_dir.is_dir():
        raise FileNotFoundError(
            f"COLMAP model not found at {colmap_dir}. Run data-processor/scannet.py first."
        )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images not found at {images_dir}")

    print(f"Loading COLMAP from {colmap_dir}")
    colmap_model = load_colmap_model(str(colmap_dir))
    cameras, images, points3D = colmap_model
    point_to_images = reverse_index_points3D(points3D)

    print(f"Loading mesh {ply_path}")
    verts = _load_mesh_vertices(ply_path)
    vertex_obj = _vertex_object_ids(scannet_scene_dir, scene_id, len(verts))
    labels = _labels_from_aggregation(scannet_scene_dir, scene_id)

    from scipy.spatial import cKDTree

    mesh_tree = cKDTree(verts)

    print("Assigning COLMAP points to GT objects (nearest mesh vertex)…")
    obj_to_pids = _assign_colmap_points_to_objects(
        points3D, mesh_tree, vertex_obj, position_tol
    )

    # Stable object order: numeric object ids present in aggregation
    object_ids = sorted(obj_to_pids.keys(), key=int)
    if max_objects is not None:
        object_ids = object_ids[: max_objects]

    connected_components: List[Dict[str, Any]] = []
    bbox_corners_out: List[Dict[str, Any]] = []
    captions_out: List[Dict[str, Any]] = []
    image_crop_all: Dict[str, Any] = {}

    for oid in object_ids:
        pids = obj_to_pids.get(oid, [])
        if len(pids) < min_colmap_points:
            print(
                f"  Skip object {oid}: only {len(pids)} COLMAP points "
                f"(need >= {min_colmap_points})"
            )
            continue

        mask = _object_vertex_mask(scannet_scene_dir, scene_id, oid, verts)
        if not mask.any():
            print(f"  Skip object {oid}: no mesh vertices")
            continue

        pts = verts[mask]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_inner = _aabb_dict_from_minmax(bbox_min, bbox_max)
        corners_8 = np.array(bbox_inner["corners"], dtype=np.float64)

        connected_components.append(
            {
                "connected_comp_id": int(oid),
                "mask_id_set": [f"gt_seg_{oid}"],
                "set_of_point3DIds": sorted(pids),
            }
        )

        bbox_corners_out.append(
            {
                "connected_comp_id": int(oid),
                "num_point3d_ids": len(pids),
                "num_points_used": len(pids),
                "num_filtered": 0,
                "bbox": bbox_inner,
            }
        )

        cap = labels.get(oid, "").strip() or f"object_{oid}"
        captions_out.append(
            {
                "component_id": int(oid),
                "caption": cap,
                "num_images_used": 1,
                "crop_filenames": [],
            }
        )

        # Projection + 2D crops
        if not skip_crops:
            projections = process_component_bbox(
                corners_8,
                pids,
                colmap_model,
                point_to_images,
                min_fraction=min_fraction,
            )
            if projections:
                comp_crops = []
                for image_name, data in projections.items():
                    row = dict(data)
                    row["image_name"] = image_name
                    row["crop_coordinates"] = row.pop("bbox_2d")
                    comp_crops.append(row)
                comp_crops.sort(
                    key=lambda x: float(x.get("fraction_visible") or 0.0), reverse=True
                )
                comp_crops = comp_crops[:max_crops_per_component]
                image_crop_all[str(oid)] = comp_crops

    outputs_dir.mkdir(parents=True, exist_ok=True)

    with (outputs_dir / "connected_components.json").open("w", encoding="utf-8") as f:
        json.dump(connected_components, f, indent=2)
    with (outputs_dir / "bbox_corners.json").open("w", encoding="utf-8") as f:
        json.dump(bbox_corners_out, f, indent=2)
    with (outputs_dir / "component_captions.json").open("w", encoding="utf-8") as f:
        json.dump(captions_out, f, indent=2)
    with (outputs_dir / "image_crop_coordinates.json").open("w", encoding="utf-8") as f:
        json.dump(image_crop_all, f, indent=2)

    if not skip_crops and image_crop_all:
        print(f"Writing crops under {outputs_dir / 'crops'}")
        _write_crops_and_manifest(image_crop_all, images_dir, outputs_dir / "crops")
    elif not skip_crops:
        (outputs_dir / "crops").mkdir(parents=True, exist_ok=True)
        with (outputs_dir / "crops" / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump({}, f)

    # GLB for the demo viewer (load_mesh route)
    if not skip_glb:
        import open3d as o3d

        poly_dir = scan_to_map_root / "data" / dataset_name / "polycam_data"
        poly_dir.mkdir(parents=True, exist_ok=True)
        glb_path = poly_dir / "raw.glb"
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        ok = o3d.io.write_triangle_mesh(str(glb_path), mesh)
        if not ok:
            print(f"Warning: open3d failed to write {glb_path}")
        else:
            print(f"Wrote mesh for viewer: {glb_path}")

    print("\nDone.")
    print(f"  outputs → {outputs_dir}")
    print(f"  Next: cd search-server/spatial_db && python create_tables.py {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GT-based outputs + raw.glb for ScanNet semantic demo."
    )
    parser.add_argument(
        "scannet_scene_dir",
        type=Path,
        help="Path to ScanNet scene (e.g. .../scans/scene0000_00)",
    )
    parser.add_argument(
        "--scan-to-map-root",
        type=Path,
        default=_REPO,
        help="scan-to-map repo root containing data/ and outputs/ (default: parent of data-processor/)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Must match the folder under data/ from scannet.py (default: scannet_<scene_id>)",
    )
    parser.add_argument("--position-tol", type=float, default=0.002)
    parser.add_argument("--min-colmap-points", type=int, default=5)
    parser.add_argument("--min-fraction", type=float, default=0.05)
    parser.add_argument("--max-crops-per-component", type=int, default=30)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--skip-glb", action="store_true")
    parser.add_argument("--skip-crops", action="store_true")

    args = parser.parse_args()
    scene_dir = args.scannet_scene_dir.resolve()
    scene_id = scene_dir.name
    dataset_name = args.dataset_name or f"scannet_{scene_id}"

    build_outputs(
        scannet_scene_dir=scene_dir,
        scan_to_map_root=args.scan_to_map_root.resolve(),
        dataset_name=dataset_name,
        position_tol=args.position_tol,
        min_colmap_points=args.min_colmap_points,
        min_fraction=args.min_fraction,
        max_crops_per_component=args.max_crops_per_component,
        max_objects=args.max_objects,
        skip_glb=args.skip_glb,
        skip_crops=args.skip_crops,
    )


if __name__ == "__main__":
    main()
