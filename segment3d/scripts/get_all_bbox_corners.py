"""CLI script to compute bounding box corners for all connected components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.bbox_corners import get_bbox
from src.io_paths import get_colmap_model_dir, get_outputs_dir, load_config


def get_all_bbox_corners(percentile: float = 95.0) -> None:
    """
    Compute bounding box corners for all connected components.

    Args:
        percentile: Percentile threshold for outlier removal (default: 95.0)
    """
    # Load configuration
    config = load_config()

    outputs_dir = get_outputs_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    print(f"Outputs directory: {outputs_dir}")
    print(f"COLMAP model directory: {colmap_model_dir}")
    print(f"Outlier filter percentile: {percentile}%")

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

    print(f"Found {len(connected_components)} connected components")

    # Compute bounding boxes for each component
    bbox_results = []

    for component in connected_components:
        comp_id = component["connected_comp_id"]
        point3D_ids = component["set_of_point3DIds"]

        print(f"\nProcessing component {comp_id} ({len(point3D_ids)} points)...")

        try:
            bbox_info = get_bbox(point3D_ids, colmap_model_dir, percentile=percentile)

            # Convert numpy arrays to lists for JSON serialization
            bbox_result = {
                "connected_comp_id": comp_id,
                "num_point3d_ids": len(point3D_ids),
                "num_points_used": int(bbox_info["num_points"]),
                "num_filtered": int(bbox_info["num_filtered"]),
                "bbox": {
                    "corners": bbox_info["corners"].tolist(),
                    "min": bbox_info["min"].tolist(),
                    "max": bbox_info["max"].tolist(),
                    "center": bbox_info["center"].tolist(),
                    "size": bbox_info["size"].tolist(),
                },
            }

            bbox_results.append(bbox_result)

            print(f"  Bounding box computed:")
            print(f"    Points used: {bbox_info['num_points']}/{len(point3D_ids)}")
            print(f"    Filtered out: {bbox_info['num_filtered']}")
            print(
                f"    Center: [{bbox_info['center'][0]:.3f}, {bbox_info['center'][1]:.3f}, {bbox_info['center'][2]:.3f}]"
            )
            print(
                f"    Size: [{bbox_info['size'][0]:.3f}, {bbox_info['size'][1]:.3f}, {bbox_info['size'][2]:.3f}]"
            )

        except Exception as e:
            print(f"  Error computing bounding box for component {comp_id}: {e}")
            continue

    # Save results
    output_path = outputs_dir / "bbox_corners.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(bbox_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved bounding box corners to: {output_path}")
    print(
        f"Successfully processed {len(bbox_results)}/{len(connected_components)} components"
    )

    # Save summary statistics
    if bbox_results:
        sizes = [np.array(r["bbox"]["size"]) for r in bbox_results]
        volumes = [np.prod(s) for s in sizes]

        stats = {
            "total_components": len(bbox_results),
            "percentile_filter": percentile,
            "volume_statistics": {
                "min": float(min(volumes)),
                "max": float(max(volumes)),
                "mean": float(np.mean(volumes)),
                "median": float(np.median(volumes)),
            },
            "size_statistics": {
                "min_size": np.min(sizes, axis=0).tolist(),
                "max_size": np.max(sizes, axis=0).tolist(),
                "mean_size": np.mean(sizes, axis=0).tolist(),
            },
        }

        stats_path = outputs_dir / "bbox_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved bounding box statistics to: {stats_path}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bounding box corners for all connected components"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for outlier removal (default: 95.0)",
    )

    args = parser.parse_args()

    if args.percentile <= 0 or args.percentile > 100:
        parser.error("Percentile must be between 0 and 100")

    get_all_bbox_corners(percentile=args.percentile)


if __name__ == "__main__":
    main()
