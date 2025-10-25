"""CLI script to project all bounding boxes onto images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.io_paths import get_colmap_model_dir, get_outputs_dir, load_config
from src.project_bbox import bbox_3d_to_2d


def project_all_bboxes(min_fraction: float = 0.3) -> None:
    """
    Project all 3D bounding boxes onto images and save crop coordinates.

    Args:
        min_fraction: Minimum fraction of 3D points that must be visible in an image
    """
    # Load configuration
    config = load_config()

    outputs_dir = get_outputs_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    print(f"Outputs directory: {outputs_dir}")
    print(f"COLMAP model directory: {colmap_model_dir}")
    print(f"Minimum fraction threshold: {min_fraction}")

    # Load connected components
    components_path = outputs_dir / "connected_components.json"
    if not components_path.exists():
        raise FileNotFoundError(
            f"Connected components file not found: {components_path}\n"
            "Please run scripts.build_mask_graph first."
        )

    print(f"\nLoading connected components from: {components_path}")
    with components_path.open("r", encoding="utf-8") as f:
        connected_components = json.load(f)

    # Load bounding box corners
    bbox_corners_path = outputs_dir / "bbox_corners.json"
    if not bbox_corners_path.exists():
        raise FileNotFoundError(
            f"Bounding box corners file not found: {bbox_corners_path}\n"
            "Please run scripts.get_all_bbox_corners first."
        )

    print(f"Loading bounding box corners from: {bbox_corners_path}")
    with bbox_corners_path.open("r", encoding="utf-8") as f:
        bbox_corners_data = json.load(f)

    print(f"Found {len(connected_components)} connected components")
    print(f"Found {len(bbox_corners_data)} bounding boxes")

    # Create lookup for bbox corners by component ID
    bbox_lookup = {
        bbox["connected_comp_id"]: bbox["bbox"]["corners"] for bbox in bbox_corners_data
    }

    # Project each component's bounding box
    all_image_crops = {}

    for component in connected_components:
        comp_id = component["connected_comp_id"]
        point3D_ids = component["set_of_point3DIds"]

        print(f"\nProcessing component {comp_id} ({len(point3D_ids)} points)...")

        # Get bbox corners for this component
        if comp_id not in bbox_lookup:
            print(f"  Warning: No bounding box found for component {comp_id}, skipping")
            continue

        bbox_corners_3d = np.array(bbox_lookup[comp_id])

        try:
            # Project to images
            image_projections = bbox_3d_to_2d(
                bbox_corners_3d,
                point3D_ids,
                colmap_model_dir,
                min_fraction=min_fraction,
            )

            print(f"  Projected to {len(image_projections)} images")

            # Structure results by component ID
            component_crops = []
            for image_name, projection_data in image_projections.items():
                component_crops.append(
                    {
                        "image_name": image_name,
                        "crop_coordinates": projection_data["bbox_2d"],
                        "image_id": projection_data["image_id"],
                        "corners_2d": projection_data["corners_2d"],
                        "visible_points": projection_data["visible_points"],
                        "total_points": projection_data["total_points"],
                        "fraction_visible": projection_data["fraction_visible"],
                        "image_width": projection_data["image_width"],
                        "image_height": projection_data["image_height"],
                    }
                )

            all_image_crops[str(comp_id)] = component_crops

        except Exception as e:
            print(f"  Error projecting component {comp_id}: {e}")
            continue

    # Save results
    output_path = outputs_dir / "image_crop_coordinates.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_image_crops, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved crop coordinates to: {output_path}")
    print(f"Total components with crops: {len(all_image_crops)}")

    # Print statistics
    total_crops = sum(len(crops) for crops in all_image_crops.values())
    print(f"Total crop regions: {total_crops}")

    if all_image_crops:
        crops_per_component = [len(crops) for crops in all_image_crops.values()]
        print(
            f"Average crops per component: {sum(crops_per_component) / len(crops_per_component):.2f}"
        )
        print(f"Max crops in single component: {max(crops_per_component)}")

    # Save summary statistics
    stats = {
        "total_components": len(all_image_crops),
        "total_crops": total_crops,
        "min_fraction_threshold": min_fraction,
    }

    if all_image_crops:
        crops_per_component = [len(crops) for crops in all_image_crops.values()]
        stats["crops_per_component"] = {
            "min": min(crops_per_component),
            "max": max(crops_per_component),
            "mean": sum(crops_per_component) / len(crops_per_component),
        }

    stats_path = outputs_dir / "crop_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Project 3D bounding boxes onto images"
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.3,
        help="Minimum fraction of 3D points that must be visible (default: 0.3)",
    )

    args = parser.parse_args()

    if args.min_fraction <= 0 or args.min_fraction > 1:
        parser.error("min-fraction must be between 0 and 1")

    project_all_bboxes(min_fraction=args.min_fraction)


if __name__ == "__main__":
    main()
