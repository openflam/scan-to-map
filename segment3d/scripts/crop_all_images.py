"""CLI script to crop all images based on projection coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.io_paths import get_images_dir, get_outputs_dir, load_config


def crop_all_images() -> None:
    """
    Crop images based on projected bounding box coordinates.

    Reads image_crop_coordinates.json and saves cropped images organized by component ID.
    """
    # Load configuration
    config = load_config()

    outputs_dir = get_outputs_dir(config)
    images_dir = get_images_dir(config)

    print(f"Outputs directory: {outputs_dir}")
    print(f"Images directory: {images_dir}")

    # Load crop coordinates
    crop_coords_path = outputs_dir / "image_crop_coordinates.json"
    if not crop_coords_path.exists():
        raise FileNotFoundError(
            f"Crop coordinates file not found: {crop_coords_path}\n"
            "Please run scripts.project_all_bboxes first."
        )

    print(f"\nLoading crop coordinates from: {crop_coords_path}")
    with crop_coords_path.open("r", encoding="utf-8") as f:
        crop_coordinates = json.load(f)

    print(f"Found {len(crop_coordinates)} components with crops")

    # Create main crops directory
    crops_dir = outputs_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    print(f"Crops will be saved to: {crops_dir}")

    # Statistics
    total_crops = 0
    failed_crops = 0

    # Process each component
    for comp_id, crop_list in crop_coordinates.items():
        print(f"\nProcessing component {comp_id} ({len(crop_list)} crops)...")

        # Create directory for this component
        component_dir = crops_dir / f"component_{comp_id}"
        component_dir.mkdir(exist_ok=True)

        # Process each crop for this component
        for idx, crop_info in enumerate(crop_list):
            image_name = crop_info["image_name"]
            crop_coords = crop_info["crop_coordinates"]

            # Get crop coordinates
            x_min, y_min, x_max, y_max = crop_coords

            # Convert to integers
            x_min = int(round(x_min))
            y_min = int(round(y_min))
            x_max = int(round(x_max))
            y_max = int(round(y_max))

            # Validate crop coordinates
            if x_max <= x_min or y_max <= y_min:
                print(
                    f"  Warning: Invalid crop coordinates for {image_name}: [{x_min}, {y_min}, {x_max}, {y_max}]"
                )
                failed_crops += 1
                continue

            # Load image
            image_path = images_dir / image_name
            if not image_path.exists():
                print(f"  Warning: Image not found: {image_path}")
                failed_crops += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  Warning: Could not read image: {image_path}")
                failed_crops += 1
                continue

            # Verify crop bounds are within image
            img_height, img_width = image.shape[:2]
            x_min = max(0, min(x_min, img_width - 1))
            y_min = max(0, min(y_min, img_height - 1))
            x_max = max(x_min + 1, min(x_max, img_width))
            y_max = max(y_min + 1, min(y_max, img_height))

            # Crop image
            cropped_image = image[y_min:y_max, x_min:x_max]

            if cropped_image.size == 0:
                print(f"  Warning: Empty crop for {image_name}")
                failed_crops += 1
                continue

            # Create output filename
            # Format: component_<id>_<image_stem>_crop<idx>.jpg
            image_stem = Path(image_name).stem
            output_filename = f"{image_stem}_crop{idx:03d}.jpg"
            output_path = component_dir / output_filename

            # Save cropped image
            cv2.imwrite(str(output_path), cropped_image)
            total_crops += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == len(crop_list):
                print(f"  Processed {idx + 1}/{len(crop_list)} crops")

    print(f"\n{'='*60}")
    print(f"Cropping complete!")
    print(f"Successfully saved: {total_crops} crops")
    print(f"Failed: {failed_crops} crops")
    print(f"Output directory: {crops_dir}")

    # Save manifest file
    manifest = {
        "total_crops": total_crops,
        "failed_crops": failed_crops,
        "components": {},
    }

    for comp_id in crop_coordinates.keys():
        component_dir = crops_dir / f"component_{comp_id}"
        if component_dir.exists():
            crop_files = list(component_dir.glob("*.jpg"))
            manifest["components"][comp_id] = {
                "num_crops": len(crop_files),
                "directory": f"component_{comp_id}",
            }

    manifest_path = crops_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to: {manifest_path}")


def main() -> None:
    """Main entry point."""
    crop_all_images()


if __name__ == "__main__":
    main()
