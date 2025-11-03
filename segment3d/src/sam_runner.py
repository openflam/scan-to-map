"""Run SAM on images and save masks in COCO format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from .io_paths import (
    get_device,
    get_images_dir,
    get_masks_dir,
    get_masks_images_dir,
    get_sam_checkpoint,
    get_sam_model_type,
    load_config,
)


def show_anns_side_by_side(image, anns, output_path, figsize=(20, 20)):
    """
    Displays the original image on the left and the mask on the right.
    Saves the result to output_path.

    Args:
        image: (H, W, 3) numpy array
        anns: list of dicts with 'segmentation' (COCO RLE format) and 'area' fields
        output_path: Path to save the figure
        figsize: Figure size tuple
    """
    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ----- LEFT: Original image -----
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    # ----- RIGHT: Mask -----
    if anns:
        # Sort by area, largest first
        sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)

        # Get image dimensions from first annotation
        # Decode the RLE format to get dimensions
        first_mask = mask_utils.decode(sorted_anns[0]["segmentation"])
        h, w = first_mask.shape

        # Create transparent RGBA overlay
        overlay = np.zeros((h, w, 4))
        overlay[..., :] = 0  # fully transparent initially

        # Add masks with random colors
        for ann in sorted_anns:
            # Decode COCO RLE format to binary mask
            if isinstance(ann["segmentation"], dict):
                m = mask_utils.decode(ann["segmentation"])
            else:
                # In case it's already a numpy array
                m = ann["segmentation"]

            color_mask = np.concatenate([np.random.random(3), [0.6]])  # RGB + alpha
            overlay[m > 0] = color_mask

        # Plot masks
        axes[1].imshow(overlay)
    axes[1].axis("off")
    axes[1].set_title(f"Masks ({len(anns)} segments)")

    plt.tight_layout()
    plt.savefig(output_path, format="jpg", dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_sam_on_images(dataset_name: str) -> None:
    """Run SAM on all images and save masks in COCO format.

    Args:
        config_path: Path to config file (uses default if None)
    """
    # Load configuration
    config = load_config(dataset_name)

    images_dir = get_images_dir(config)
    masks_dir = get_masks_dir(config)
    masks_images_dir = get_masks_images_dir(config)
    sam_checkpoint = get_sam_checkpoint(config)
    sam_model_type = get_sam_model_type(config)
    device = get_device(config)

    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Masks images directory: {masks_images_dir}")
    print(f"SAM checkpoint: {sam_checkpoint}")
    print(f"SAM model type: {sam_model_type}")
    print(f"Device: {device}")

    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)

    # Create mask generator with specified parameters and COCO RLE output
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     points_per_batch=64,
    #     pred_iou_thresh=0.94,
    #     stability_score_thresh=0.98,
    #     stability_score_offset=1.0,
    #     box_nms_thresh=0.5,
    #     crop_n_layers=0,
    #     crop_nms_thresh=0.5,
    #     crop_overlap_ratio=0.2,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=10000,
    #     output_mode="coco_rle",
    # )
    mask_generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")

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
            print(f"  Saved masks to {output_path.name}")

            # Save visualization (side-by-side image and mask)
            viz_output_path = masks_images_dir / f"{image_path.stem}_visualization.jpg"
            show_anns_side_by_side(image_rgb, masks, viz_output_path)
            print(f"  Saved visualization to {viz_output_path.name}")

        except Exception as e:
            print(f"  Error processing image: {e}")
            continue

    print(f"\nCompleted processing all images.")
    print(f"Masks saved to: {masks_dir}")
    print(f"Visualizations saved to: {masks_images_dir}")


def main() -> None:
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Run SAM on images")
    parser.add_argument(
        "--dataset-name", type=str, required=True, help="Name of the dataset to process"
    )
    args = parser.parse_args()

    run_sam_on_images(dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
