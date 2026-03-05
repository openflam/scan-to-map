"""
associate2d3d.py - Associate per-object SAM3 mask observations with COLMAP 3D points.

Reads the mask JSON files produced by sam3_runner.py from:

    <outputs_dir>/object_level_masks/masks/<obj_slug>/seq_<i>/<frame_name>.json

Each frame JSON is a list of per-instance annotations:
    [{"obj_id": <int>, "segmentation": <COCO RLE>, "area": <float>}, ...]

For every (object, sequence, sam3_instance_id) triple the script collects all
COLMAP 2D feature points that fall inside the mask (across *all* frames in
the sequence that contain that tracking ID) and records their 3D point IDs.

Output is a single JSON file at:

    <outputs_dir>/object_level_masks/object_3d_associations.json

Structured as  { <obj_slug>: { <seq_idx>: { <obj_id>: [3D point IDs] } } }.

Usage (from the segment3d/ directory):
    python -m src.per_object_sam3.associate2d3d --dataset ProjectLabStudio_inv_method
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from ..io_paths import get_outputs_dir, get_colmap_model_dir, load_config
from ..colmap_io import index_image_metadata, load_colmap_model

# Type alias
RLE = Dict[str, Any]


# ---------------------------------------------------------------------------
# Mask ↔ point association helpers  (same logic as src/associate2d3d.py)
# ---------------------------------------------------------------------------


def _get_mask_bbox_areas(rle_list: List[RLE]) -> np.ndarray:
    """Bounding-box area for each entry in *rle_list* (uses 'segmentation' key)."""
    bboxes = mask_utils.toBbox([ann["segmentation"] for ann in rle_list])
    return bboxes[:, 2] * bboxes[:, 3]


def _build_id_map(rle_list: List[RLE], areas: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Painter's Algorithm: paint *mask index* onto an (H, W) canvas,
    large masks first so that small masks on top win.
    """
    id_map = np.full((H, W), -1, dtype=np.int32)
    sorted_indices = np.argsort(areas)[::-1]

    for mask_idx in sorted_indices:
        rle = rle_list[mask_idx]["segmentation"]
        mask_bool = mask_utils.decode(rle).astype(bool)
        if mask_bool.shape[:2] != (H, W):
            continue
        id_map[mask_bool] = mask_idx

    return id_map


def _points_in_masks(
    xys: np.ndarray,
    point3D_ids: np.ndarray,
    rle_list: List[RLE],
    H: int,
    W: int,
) -> List[List[int]]:
    """
    For each mask in *rle_list*, return the list of COLMAP 3D point IDs
    whose 2D projection (xys) falls inside that mask.

    Returns a list of length len(rle_list); each element is a sorted list of
    unique 3D point IDs.
    """
    num_masks = len(rle_list)
    if num_masks == 0 or len(xys) == 0:
        return [[] for _ in range(num_masks)]

    areas = _get_mask_bbox_areas(rle_list)
    id_map = _build_id_map(rle_list, areas, H, W)

    # Filter out unmatched keypoints (point3D_id == -1)
    valid_mask = point3D_ids != -1
    valid_xys = xys[valid_mask].astype(int)
    valid_p3d = point3D_ids[valid_mask]

    if len(valid_xys) == 0:
        return [[] for _ in range(num_masks)]

    x, y = valid_xys[:, 0], valid_xys[:, 1]
    bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    final_x = x[bounds]
    final_y = y[bounds]
    final_p3d = valid_p3d[bounds]

    # Vectorised lookup into the painter's-algorithm id_map
    associated_mask_indices = id_map[final_y, final_x]

    mask_point_sets: List[Set[int]] = [set() for _ in range(num_masks)]
    for i, mask_idx in enumerate(associated_mask_indices):
        if mask_idx != -1:
            mask_point_sets[mask_idx].add(int(final_p3d[i]))

    return [sorted(s) for s in mask_point_sets]


# ---------------------------------------------------------------------------
# Main association logic
# ---------------------------------------------------------------------------


def associate_per_object(dataset_name: str) -> None:
    config = load_config(dataset_name)
    outputs_dir = get_outputs_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    masks_base_dir = outputs_dir / "object_level_masks" / "masks"
    if not masks_base_dir.is_dir():
        raise NotADirectoryError(
            f"Masks directory not found: {masks_base_dir}\n"
            "Run sam3_runner.py first to generate per-object mask files."
        )

    # ----- Load COLMAP model ------------------------------------------------
    print("Loading COLMAP model…")
    cameras, images, _ = load_colmap_model(str(colmap_model_dir))
    image_metadata = index_image_metadata(images)

    # Build stem → (xys, point3D_ids, H, W) for fast per-frame lookup
    stem_to_meta: Dict[str, Dict[str, Any]] = {}
    for image_id, meta in image_metadata.items():
        stem = Path(meta["name"]).stem
        camera = cameras[images[image_id].camera_id]
        stem_to_meta[stem] = {
            "xys": meta["xys"],
            "point3D_ids": meta["point3D_ids"],
            "H": int(camera.height),
            "W": int(camera.width),
        }

    # ----- Discover (obj_slug, seq_idx) pairs --------------------------------
    # Directory layout: masks_base_dir/<obj_slug>/seq_<i>/<frame_name>.json
    work_items = []  # (obj_slug, seq_idx, seq_dir)
    for obj_dir in sorted(masks_base_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        for seq_dir in sorted(obj_dir.iterdir()):
            if not seq_dir.is_dir() or not seq_dir.name.startswith("seq_"):
                continue
            try:
                seq_idx = int(seq_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            work_items.append((obj_dir.name, seq_idx, seq_dir))

    if not work_items:
        print("No mask sequences found – nothing to do.")
        return

    print(f"Found {len(work_items)} (object, sequence) pair(s) to process.")

    # ----- Associate ---------------------------------------------------------
    # Nested: result[obj_slug][seq_idx][obj_id] → accumulated Set[int]
    result: Dict[str, Dict[int, Dict[int, Set[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )

    for obj_slug, seq_idx, seq_dir in tqdm(work_items, unit="seq"):
        frame_files = sorted(seq_dir.glob("*.json"))
        for frame_path in frame_files:
            frame_stem = frame_path.stem  # e.g. "frame_00002"

            colmap_meta = stem_to_meta.get(frame_stem)
            if colmap_meta is None:
                # This frame has no COLMAP data – skip silently
                continue

            with frame_path.open("r", encoding="utf-8") as fh:
                annotations: List[Dict[str, Any]] = json.load(fh)

            if not annotations:
                continue

            H = colmap_meta["H"]
            W = colmap_meta["W"]
            xys = colmap_meta["xys"]
            point3D_ids = colmap_meta["point3D_ids"]

            # points_in_masks returns one list per element of annotations
            per_mask_points = _points_in_masks(xys, point3D_ids, annotations, H, W)

            for ann, pts in zip(annotations, per_mask_points):
                if not pts:
                    continue
                obj_id = int(ann["obj_id"])
                result[obj_slug][seq_idx][obj_id].update(pts)

    # ----- Serialise ---------------------------------------------------------
    output = {
        obj_slug: {
            f"seq_{seq_idx}": {
                str(obj_id): sorted(pts) for obj_id, pts in sorted(seq_data.items())
            }
            for seq_idx, seq_data in sorted(seq_map.items())
        }
        for obj_slug, seq_map in sorted(result.items())
    }

    out_path = outputs_dir / "object_level_masks" / "object_3d_associations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    total_instances = sum(
        len(obj_data) for seq_map in output.values() for obj_data in seq_map.values()
    )
    total_pts = sum(
        len(pts)
        for seq_map in output.values()
        for obj_data in seq_map.values()
        for pts in obj_data.values()
    )
    print(
        f"\nDone.  {total_instances} instance(s) written, "
        f"{total_pts} total 3D-point associations."
    )
    print(f"Output → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Associate per-object SAM3 masks with COLMAP 3D points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a folder under data/ and outputs/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    associate_per_object(args.dataset)
