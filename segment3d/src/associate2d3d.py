"""Associate 2D mask observations with 3D point IDs from COLMAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from pycocotools import mask as mask_utils


def points_in_masks(
    xys: np.ndarray, point3D_ids: np.ndarray, masks: List[np.ndarray], H: int, W: int
) -> List[Set[int]]:
    """
    Assign 3D point IDs to masks based on 2D observations.

    For each 2D observation (x, y), add its point3D_id to the single
    smallest-bbox mask that contains (x, y) if multiple masks contain it.

    Args:
        xys: Integer pixel coordinates of shape (N, 2) in format [x, y]
        point3D_ids: Array of 3D point IDs of length N
        masks: List of boolean H×W arrays (one per mask)
        H: Image height
        W: Image width

    Returns:
        List of sets, one set of 3D point IDs per mask
    """
    # Initialize result: one set per mask
    mask_point_sets: List[Set[int]] = [set() for _ in masks]

    if len(masks) == 0:
        return mask_point_sets

    # Precompute bounding box areas for each mask
    mask_areas = []
    for mask in masks:
        pos = np.where(mask)
        if len(pos[0]) == 0:
            mask_areas.append(float("inf"))  # Empty mask
        else:
            y_min, y_max = pos[0].min(), pos[0].max()
            x_min, x_max = pos[1].min(), pos[1].max()
            area = (y_max - y_min + 1) * (x_max - x_min + 1)
            mask_areas.append(area)

    # Process each observation
    for i in range(len(xys)):
        point3D_id = int(point3D_ids[i])

        # Ignore invalid 3D point IDs
        if point3D_id == -1:
            continue

        x, y = int(xys[i, 0]), int(xys[i, 1])

        # Check bounds
        if not (0 <= x < W and 0 <= y < H):
            continue

        # Find all masks containing this point
        containing_mask_indices = []
        for mask_idx, mask in enumerate(masks):
            if mask[y, x]:  # Note: mask indexing is [y, x]
                containing_mask_indices.append(mask_idx)

        # Assign to smallest-bbox mask if any contain this point
        if containing_mask_indices:
            # Find the mask with smallest area
            smallest_idx = min(containing_mask_indices, key=lambda idx: mask_areas[idx])
            mask_point_sets[smallest_idx].add(point3D_id)

    return mask_point_sets


def decode_to_bool(rle_list: List[Dict[str, Any]], H: int, W: int) -> List[np.ndarray]:
    """
    Decode COCO RLE masks to boolean arrays.

    Args:
        rle_list: List of mask annotations with 'segmentation' field in COCO RLE format
        H: Image height
        W: Image width

    Returns:
        List of boolean numpy arrays of shape (H, W)
    """
    bool_masks = []

    for annotation in rle_list:
        # Decode RLE to binary mask
        rle = annotation["segmentation"]
        mask = mask_utils.decode(rle)

        # Convert to boolean
        bool_mask = mask.astype(bool)

        # Verify dimensions
        if bool_mask.shape != (H, W):
            raise ValueError(
                f"Mask shape {bool_mask.shape} does not match expected ({H}, {W})"
            )

        bool_masks.append(bool_mask)

    return bool_masks


def load_masks_rle(path: Path) -> List[Dict[str, Any]]:
    """
    Load masks in COCO RLE format from a JSON file.

    Args:
        path: Path to JSON file containing masks in COCO RLE format

    Returns:
        List of mask annotation dictionaries
    """
    with path.open("r", encoding="utf-8") as f:
        masks_data = json.load(f)

    # Handle both list of annotations and dict with 'annotations' key
    if isinstance(masks_data, list):
        return masks_data
    elif isinstance(masks_data, dict) and "annotations" in masks_data:
        return masks_data["annotations"]
    else:
        raise ValueError(f"Unexpected masks data format in {path}")
