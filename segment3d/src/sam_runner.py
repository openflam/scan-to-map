"""Run SAM on images and save masks in COCO format."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from .io_paths import (
    get_device,
    get_images_dir,
    get_masks_dir,
    get_sam_checkpoint,
    get_sam_model_type,
    load_config,
)


def run_sam_on_images(config_path: str | Path | None = None) -> None:
    """Run SAM on all images and save masks in COCO format.

    Args:
        config_path: Path to config file (uses default if None)
    """
    # Load configuration
    config = load_config() if config_path is None else load_config(config_path)

    images_dir = get_images_dir(config)
    masks_dir = get_masks_dir(config)
    sam_checkpoint = get_sam_checkpoint(config)
    sam_model_type = get_sam_model_type(config)
    device = get_device(config)

    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"SAM checkpoint: {sam_checkpoint}")
    print(f"SAM model type: {sam_model_type}")
    print(f"Device: {device}")

    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)

    # Create mask generator with specified parameters and COCO RLE output
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.94,
        stability_score_thresh=0.98,
        stability_score_offset=1.0,
        box_nms_thresh=0.5,
        crop_n_layers=0,
        crop_nms_thresh=0.5,
        crop_overlap_ratio=0.2,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=10000,
        output_mode="coco_rle",
    )

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
    )

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_paths)} images")

    # Process each image
    for image_id, image_path in enumerate(image_paths, start=1):
        print(f"Processing {image_id}/{len(image_paths)}: {image_path.name}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: Could not read image, skipping")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks - output is already in COCO RLE format
        try:
            masks = mask_generator.generate(image_rgb)
            print(f"  Generated {len(masks)} masks")

            # Save masks for this image
            output_path = masks_dir / f"{image_path.stem}_masks.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(masks, f)
            print(f"  Saved to {output_path.name}")

        except Exception as e:
            print(f"  Error processing image: {e}")
            continue

    print(f"\nCompleted processing all images. Masks saved to {masks_dir}")


def main() -> None:
    """Main entry point."""
    run_sam_on_images()


if __name__ == "__main__":
    main()
