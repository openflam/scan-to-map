"""Associate 2D mask observations with 3D point IDs from COLMAP."""

from __future__ import annotations

import argparse
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


def associate_all_images(dataset_name: str) -> None:
    """
    Associate 2D masks with 3D points for all COLMAP images.

    For each COLMAP image:
    1. Load the image's 2D observations (xys) and 3D point IDs
    2. Load and decode the corresponding masks
    3. Associate 3D point IDs to masks
    4. Save the results as JSON
    """
    import cv2
    from .colmap_io import index_image_metadata, load_colmap_model
    from .io_paths import (
        get_colmap_model_dir,
        get_images_dir,
        get_associations_dir,
        get_masks_dir,
        load_config,
    )

    # Load configuration
    config = load_config(dataset_name)

    images_dir = get_images_dir(config)
    masks_dir = get_masks_dir(config)
    associations_dir = get_associations_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Associations directory: {associations_dir}")
    print(f"COLMAP model directory: {colmap_model_dir}")

    # Load COLMAP model
    print("\nLoading COLMAP model...")
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))
    image_metadata = index_image_metadata(images)

    print(f"Loaded {len(images)} images from COLMAP model")

    # Process each COLMAP image
    processed_count = 0
    skipped_count = 0

    for image_id, metadata in image_metadata.items():
        image_name = metadata["name"]
        xys = metadata["xys"]
        point3D_ids = metadata["point3D_ids"]

        # Check if mask file exists
        image_stem = Path(image_name).stem
        mask_path = masks_dir / f"{image_stem}_masks.json"

        if not mask_path.exists():
            print(
                f"[{image_id}] Skipping {image_name}: mask file not found at {mask_path}"
            )
            skipped_count += 1
            continue

        # Get image dimensions
        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"[{image_id}] Skipping {image_name}: image file not found")
            skipped_count += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{image_id}] Skipping {image_name}: could not read image")
            skipped_count += 1
            continue

        H, W = image.shape[:2]

        # Load and decode masks
        try:
            masks_rle = load_masks_rle(mask_path)
            masks_bool = decode_to_bool(masks_rle, H, W)
        except Exception as e:
            print(f"[{image_id}] Error loading masks for {image_name}: {e}")
            skipped_count += 1
            continue

        # Associate 3D points to masks
        mask_point_sets = points_in_masks(xys, point3D_ids, masks_bool, H, W)

        # Convert sets to lists for JSON serialization
        mask_point3d_lists = [sorted(list(s)) for s in mask_point_sets]

        # Prepare output data
        output_data = {
            "image_id": int(image_id),
            "image_name": image_name,
            "mask_point3d_sets": mask_point3d_lists,
            "H": H,
            "W": W,
        }

        # Save to associations directory
        output_path = associations_dir / f"imageId_{image_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        # Print progress
        total_points = sum(len(s) for s in mask_point_sets)
        print(
            f"[{image_id}] {image_name}: {len(masks_bool)} masks, {total_points} point associations → {output_path.name}"
        )
        processed_count += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Results saved to: {associations_dir}")


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Associate 2D masks with 3D points")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to process"
    )

    args = parser.parse_args()

    associate_all_images(args.dataset_name)
    parser = argparse.ArgumentParser(description="Associate 2D masks with 3D points")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to process"
    )

    args = parser.parse_args()

    associate_all_images(args.dataset_name)


if __name__ == "__main__":
    main()
