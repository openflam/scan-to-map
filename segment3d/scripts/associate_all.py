"""CLI script to associate 2D masks with 3D points for all COLMAP images."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.associate2d3d import decode_to_bool, load_masks_rle, points_in_masks
from src.colmap_io import index_image_metadata, load_colmap_model
from src.io_paths import (
    get_colmap_model_dir,
    get_images_dir,
    get_labels_dir,
    get_masks_dir,
    load_config,
)


def associate_all_images() -> None:
    """
    Associate 2D masks with 3D points for all COLMAP images.

    For each COLMAP image:
    1. Load the image's 2D observations (xys) and 3D point IDs
    2. Load and decode the corresponding masks
    3. Associate 3D point IDs to masks
    4. Save the results as JSON
    """
    # Load configuration
    config = load_config()

    images_dir = get_images_dir(config)
    masks_dir = get_masks_dir(config)
    labels_dir = get_labels_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Labels directory: {labels_dir}")
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

        # Save to labels directory
        output_path = labels_dir / f"imageId_{image_id}.json"
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
    print(f"Results saved to: {labels_dir}")


def main() -> None:
    """Main entry point."""
    associate_all_images()


if __name__ == "__main__":
    main()
