"""
Crop images based on projected 3D bounding boxes.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np


def crop_single_image(
    image_path: Path, crop_coords: List[int], output_path: Path
) -> bool:
    """
    Crop a single image and save the result.

    Args:
        image_path: Path to the source image
        crop_coords: [x_min, y_min, x_max, y_max]
        output_path: Path to save the cropped image

    Returns:
        True if successful, False otherwise
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read image: {image_path}")
        return False

    # Get image dimensions
    img_height, img_width = img.shape[:2]

    # Extract and convert coordinates to integers
    x_min, y_min, x_max, y_max = crop_coords
    x_min = int(round(x_min))
    y_min = int(round(y_min))
    x_max = int(round(x_max))
    y_max = int(round(y_max))

    # Validate crop coordinates
    if x_max <= x_min or y_max <= y_min:
        print(
            f"  Warning: Invalid crop coordinates for {image_path.name}: [{x_min}, {y_min}, {x_max}, {y_max}]"
        )
        return False

    # Clamp coordinates to image bounds
    x_min = max(0, min(x_min, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    x_max = max(x_min + 1, min(x_max, img_width))
    y_max = max(y_min + 1, min(y_max, img_height))

    # Crop
    cropped = img[y_min:y_max, x_min:x_max]

    # Verify crop is not empty
    if cropped.size == 0:
        print(f"  Warning: Empty crop for {image_path.name}")
        return False

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), cropped)

    if not success:
        print(f"  Warning: Failed to write crop to {output_path}")
        return False

    return True


def crop_all_images_cli(dataset_name: str) -> None:
    """
    Crop all images based on projected bounding box coordinates.
    """
    from .io_paths import load_config, get_images_dir, get_outputs_dir

    # Load configuration
    config = load_config(dataset_name)

    images_dir = get_images_dir(config)
    outputs_dir = get_outputs_dir(config)

    print(f"Images directory: {images_dir}")
    print(f"Outputs directory: {outputs_dir}")

    # Load crop coordinates
    crop_coords_path = outputs_dir / "image_crop_coordinates.json"
    if not crop_coords_path.exists():
        raise FileNotFoundError(
            f"Crop coordinates file not found: {crop_coords_path}\n"
            "Please run src.project_bbox first."
        )

    print(f"\nLoading crop coordinates from: {crop_coords_path}")
    with crop_coords_path.open("r", encoding="utf-8") as f:
        all_crop_data = json.load(f)

    # Create crops directory
    crops_dir = outputs_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Crops will be saved to: {crops_dir}")

    # Process each component
    manifest_data = {}
    total_crops = 0
    failed_crops = 0

    for comp_id, crop_list in all_crop_data.items():
        print(f"\nProcessing component {comp_id} ({len(crop_list)} crops)...")

        # Create component directory
        comp_dir = crops_dir / f"component_{comp_id}"
        comp_dir.mkdir(parents=True, exist_ok=True)

        component_crops = []

        for idx, crop_info in enumerate(crop_list):
            image_name = crop_info["image_name"]
            crop_coords = crop_info["crop_coordinates"]

            # Find source image
            image_path = images_dir / image_name
            if not image_path.exists():
                print(f"  Warning: Image not found: {image_path}, skipping")
                failed_crops += 1
                continue

            # Generate output filename
            image_stem = image_path.stem
            output_name = f"{image_stem}_crop{idx:03d}.jpg"
            output_path = comp_dir / output_name

            try:
                success = crop_single_image(image_path, crop_coords, output_path)

                if success:
                    # Add to manifest
                    component_crops.append(
                        {
                            "crop_filename": output_name,
                            "source_image": image_name,
                            "crop_index": idx,
                            "crop_coordinates": crop_coords,
                            "image_id": crop_info["image_id"],
                            "fraction_visible": crop_info["fraction_visible"],
                            "visible_points": crop_info["visible_points"],
                            "total_points": crop_info["total_points"],
                        }
                    )
                    total_crops += 1
                else:
                    failed_crops += 1

                if (idx + 1) % 10 == 0 or (idx + 1) == len(crop_list):
                    print(f"  Processed {idx + 1}/{len(crop_list)} crops")

            except Exception as e:
                print(f"  Error cropping {image_name}: {e}")
                failed_crops += 1
                continue

        manifest_data[comp_id] = {
            "component_id": int(comp_id),
            "total_crops": len(component_crops),
            "crops": component_crops,
        }

        print(f"  Saved {len(component_crops)} crops to {comp_dir}")

    # Save manifest
    manifest_path = crops_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Cropping complete!")
    print(f"Total components processed: {len(manifest_data)}")
    print(f"Successfully saved: {total_crops} crops")
    print(f"Failed: {failed_crops} crops")
    print(f"Manifest saved to: {manifest_path}")

    # Print statistics
    if manifest_data:
        crops_per_component = [
            comp_data["total_crops"] for comp_data in manifest_data.values()
        ]
        print(f"\nStatistics:")
        print(
            f"  Average crops per component: {sum(crops_per_component) / len(crops_per_component):.2f}"
        )
        print(f"  Min crops in component: {min(crops_per_component)}")
        print(f"  Max crops in component: {max(crops_per_component)}")


def main() -> None:
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop images based on projected 3D bounding boxes"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to process"
    )

    args = parser.parse_args()

    crop_all_images_cli(args.dataset_name)


if __name__ == "__main__":
    main()
