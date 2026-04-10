"""
Export GT or predicted segments as a colored PLY for offline viewing.

This is meant for SSH/server workflows: it does not open a window. Instead it
writes:
  1. a colored point-cloud PLY
  2. a metadata JSON with IDs, labels, colors, counts, and AABBs

Examples
--------
GT objects from ScanNet labels:
    python eval/export_segment_vis.py \
        --source gt \
        --scene-id scene0000_00 \
        --scannet-root /path/to/ScanQA/data/data

Predicted scan-to-map connected components:
    python eval/export_segment_vis.py \
        --source pred \
        --scene-id scene0000_00 \
        --data-root /path/to/scan-to-map/data \
        --outputs-root /path/to/scan-to-map/outputs
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_ply_vertices(ply_path: Path) -> Optional[np.ndarray]:
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        return verts if len(verts) > 0 else None
    except ImportError:
        pass

    import struct

    with open(ply_path, "rb") as fh:
        header_lines = []
        while True:
            line = fh.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        header_text = b"".join(header_lines).decode("ascii", errors="replace")
        lines = header_text.splitlines()

        fmt = "ascii"
        for line in lines:
            if line.startswith("format"):
                fmt = line.split()[1]

        n_verts = 0
        for line in lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])

        prop_types: List[Tuple[str, str]] = []
        in_vertex = False
        for line in lines:
            if line.startswith("element vertex"):
                in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                parts = line.split()
                prop_types.append((parts[1], parts[2]))

        type_sizes = {
            "float": 4,
            "float32": 4,
            "double": 8,
            "float64": 8,
            "int": 4,
            "int32": 4,
            "uint": 4,
            "uint32": 4,
            "short": 2,
            "int16": 2,
            "ushort": 2,
            "uint16": 2,
            "char": 1,
            "int8": 1,
            "uchar": 1,
            "uint8": 1,
        }
        type_fmts = {
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
            "int": "i",
            "int32": "i",
            "uint": "I",
            "uint32": "I",
            "short": "h",
            "int16": "h",
            "ushort": "H",
            "uint16": "H",
            "char": "b",
            "int8": "b",
            "uchar": "B",
            "uint8": "B",
        }

        xyz_indices = {
            name: i for i, (_, name) in enumerate(prop_types) if name in ("x", "y", "z")
        }
        if len(xyz_indices) < 3:
            return None

        row_size = sum(type_sizes.get(t, 4) for t, _ in prop_types)
        verts = np.empty((n_verts, 3), dtype=np.float32)
        xi, yi, zi = xyz_indices["x"], xyz_indices["y"], xyz_indices["z"]

        if fmt == "ascii":
            for i in range(n_verts):
                row = fh.readline().split()
                verts[i] = [float(row[xi]), float(row[yi]), float(row[zi])]
        else:
            endian = "<" if "little" in fmt else ">"
            row_fmt = endian + "".join(type_fmts.get(t, "f") for t, _ in prop_types)
            unpacker = struct.Struct(row_fmt)
            for i in range(n_verts):
                vals = unpacker.unpack(fh.read(row_size))
                verts[i] = [vals[xi], vals[yi], vals[zi]]

    return verts


def _color_for_id(idx: int) -> Tuple[int, int, int]:
    # Deterministic distinct-ish palette in HSV space.
    hue = ((idx * 0.61803398875) % 1.0)
    sat = 0.65
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def _subsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[idx]


def load_gt_segments(
    scannet_root: Path,
    scene_id: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    scene_dir = scannet_root / "scans" / scene_id
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"

    with open(agg_path, "r") as f:
        agg = json.load(f)
    with open(segs_path, "r") as f:
        segs = json.load(f)
    verts = _read_ply_vertices(ply_path)
    if verts is None:
        raise FileNotFoundError(f"Could not load vertices from {ply_path}")

    seg_indices = np.array(segs["segIndices"], dtype=np.int32)

    id_to_points: Dict[int, np.ndarray] = {}
    id_to_label: Dict[int, str] = {}
    for group in agg.get("segGroups", []):
        object_id = int(group["objectId"])
        label = str(group.get("label", f"object_{object_id}"))
        seg_ids = list(group["segments"])
        mask = np.isin(seg_indices, seg_ids)
        points = verts[mask]
        if len(points) < 3:
            continue
        id_to_points[object_id] = points.astype(np.float32)
        id_to_label[object_id] = label

    return id_to_points, id_to_label


def load_pred_segments(
    data_root: Path,
    outputs_root: Path,
    scene_id: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    dataset_name = f"scannet_{scene_id}"
    points3d_path = data_root / dataset_name / "hloc_data" / "sfm_reconstruction" / "points3D.txt"
    cc_path = outputs_root / dataset_name / "connected_components.json"

    point_xyz: Dict[int, np.ndarray] = {}
    with open(points3d_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            point_xyz[point_id] = xyz

    with open(cc_path, "r") as f:
        components = json.load(f)

    id_to_points: Dict[int, np.ndarray] = {}
    id_to_label: Dict[int, str] = {}
    for comp in components:
        comp_id = int(comp["connected_comp_id"])
        point_ids = comp.get("set_of_point3DIds", [])
        pts = [point_xyz[pid] for pid in point_ids if pid in point_xyz]
        if len(pts) < 3:
            continue
        id_to_points[comp_id] = np.stack(pts).astype(np.float32)
        id_to_label[comp_id] = f"component_{comp_id}"

    return id_to_points, id_to_label


def write_colored_ply(
    out_path: Path,
    segment_points: Dict[int, np.ndarray],
    segment_labels: Dict[int, str],
    max_points_per_segment: int,
) -> List[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    metadata: List[dict] = []

    for seg_id in sorted(segment_points):
        pts = _subsample_points(segment_points[seg_id], max_points_per_segment, seed=seg_id)
        color = _color_for_id(seg_id)
        rgb = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))

        all_xyz.append(pts)
        all_rgb.append(rgb)
        metadata.append(
            {
                "id": seg_id,
                "label": segment_labels.get(seg_id, str(seg_id)),
                "color_rgb": list(color),
                "num_points_total": int(len(segment_points[seg_id])),
                "num_points_written": int(len(pts)),
                "bbox_min": segment_points[seg_id].min(axis=0).tolist(),
                "bbox_max": segment_points[seg_id].max(axis=0).tolist(),
            }
        )

    if not all_xyz:
        raise ValueError("No segments with points found.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GT or predicted segments as a colored PLY."
    )
    parser.add_argument("--scene-id", required=True, help="ScanNet scene id, e.g. scene0000_00")
    parser.add_argument(
        "--source",
        default="gt",
        choices=["gt", "pred"],
        help="Visualize ground-truth ScanNet objects or predicted scan-to-map components",
    )
    parser.add_argument(
        "--scannet-root",
        default=None,
        help="Root containing scans/<scene_id>/...  (required for --source gt)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="scan-to-map data root containing scannet_<scene_id>/...  (required for --source pred)",
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="scan-to-map outputs root containing scannet_<scene_id>/...  (required for --source pred)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/vis",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--max-points-per-segment",
        type=int,
        default=5000,
        help="Subsample each object/component to at most this many points",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "gt":
        if not args.scannet_root:
            raise ValueError("--scannet-root is required for --source gt")
        segment_points, segment_labels = load_gt_segments(Path(args.scannet_root), args.scene_id)
    else:
        if not args.data_root or not args.outputs_root:
            raise ValueError("--data-root and --outputs-root are required for --source pred")
        segment_points, segment_labels = load_pred_segments(
            Path(args.data_root), Path(args.outputs_root), args.scene_id
        )

    stem = f"{args.scene_id}_{args.source}_segments"
    ply_path = output_dir / f"{stem}.ply"
    meta_path = output_dir / f"{stem}.json"

    metadata = write_colored_ply(
        ply_path,
        segment_points,
        segment_labels,
        args.max_points_per_segment,
    )

    with open(meta_path, "w") as f:
        json.dump(
            {
                "scene_id": args.scene_id,
                "source": args.source,
                "num_segments": len(metadata),
                "segments": metadata,
            },
            f,
            indent=2,
        )

    print(f"Wrote colored PLY: {ply_path}")
    print(f"Wrote metadata  : {meta_path}")
    print(f"Segments        : {len(metadata)}")
    print(f"Tip             : download the .ply and open it in MeshLab or CloudCompare.")


if __name__ == "__main__":
    main()
