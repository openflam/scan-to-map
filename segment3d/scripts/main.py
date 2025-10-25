#!/usr/bin/env python3
"""
Main pipeline script that runs all steps sequentially.

This script orchestrates the complete scan-to-map pipeline:
1. Run SAM segmentation
2. Associate 2D masks with 3D points
3. Build mask connectivity graph
4. Compute 3D bounding boxes
5. Project bounding boxes to images
6. Crop images based on projections
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sam_runner import main as run_sam
from src.associate2d3d import associate_all_images
from src.mask_graph import build_mask_graph_cli
from src.bbox_corners import get_all_bbox_corners_cli
from src.project_bbox import project_all_bboxes_cli
from src.crop_images import crop_all_images_cli
from src.io_paths import load_config


def print_step_header(step_num: int, total_steps: int, title: str) -> None:
    """Print a formatted step header."""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 80 + "\n")


def print_step_complete(elapsed_time: float) -> None:
    """Print step completion message."""
    print(f"\n✓ Step completed in {elapsed_time:.2f} seconds")


def main() -> None:
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete scan-to-map pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Run SAM segmentation on all images
  2. Associate 2D masks with 3D COLMAP points
  3. Build mask connectivity graph and extract connected components
  4. Compute 3D bounding boxes for each component
  5. Project 3D bounding boxes onto images
  6. Crop images based on projected coordinates

Configuration:
  All settings are read from segment3d/config.yaml
        """,
    )

    # Pipeline parameters
    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Skip SAM segmentation step (use existing masks)",
    )
    parser.add_argument(
        "--skip-association",
        action="store_true",
        help="Skip 2D-3D association step (use existing associations)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Number of nearest neighbors for mask graph (default: 5)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="Jaccard similarity threshold for mask graph (default: 0.2)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for bbox outlier removal (default: 95.0)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.3,
        help="Minimum fraction of visible points for projection (default: 0.3)",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.tau <= 0 or args.tau > 1:
        parser.error("--tau must be between 0 and 1")
    if args.percentile <= 0 or args.percentile > 100:
        parser.error("--percentile must be between 0 and 100")
    if args.min_fraction <= 0 or args.min_fraction > 1:
        parser.error("--min-fraction must be between 0 and 1")

    # Load and display configuration
    print("\n" + "=" * 80)
    print("SCAN-TO-MAP PIPELINE")
    print("=" * 80)

    try:
        config = load_config()
        print("\nConfiguration loaded from: segment3d/config.yaml")
        print(f"  Images directory: {config.get('images_dir')}")
        print(f"  COLMAP model: {config.get('colmap_model_dir')}")
        print(f"  SAM checkpoint: {config.get('sam_ckpt')}")
        print(f"  Device: {config.get('device')}")
        print(f"  Outputs directory: {config.get('outputs_dir')}")
    except Exception as e:
        print(f"\nError loading configuration: {e}")
        sys.exit(1)

    print("\nPipeline Parameters:")
    if args.skip_sam:
        print("  SAM segmentation: SKIPPED")
    if args.skip_association:
        print("  2D-3D association: SKIPPED")
    print(f"  Mask graph K: {args.K}")
    print(f"  Mask graph tau: {args.tau}")
    print(f"  Bbox percentile: {args.percentile}")
    print(f"  Min fraction: {args.min_fraction}")

    # Track overall timing
    pipeline_start = time.time()
    total_steps = 6
    current_step = 0

    try:
        # Step 1: Run SAM
        if not args.skip_sam:
            current_step += 1
            print_step_header(current_step, total_steps, "Run SAM Segmentation")
            step_start = time.time()

            # SAM runner's main() uses sys.argv, so we need to clear it
            old_argv = sys.argv
            sys.argv = [sys.argv[0]]
            try:
                run_sam()
            finally:
                sys.argv = old_argv

            print_step_complete(time.time() - step_start)
        else:
            print("\nSkipping Step 1: SAM Segmentation (using existing masks)")

        # Step 2: Associate 2D-3D
        if not args.skip_association:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Associate 2D Masks with 3D Points"
            )
            step_start = time.time()

            associate_all_images()

            print_step_complete(time.time() - step_start)
        else:
            print("\nSkipping Step 2: 2D-3D Association (using existing associations)")

        # Step 3: Build mask graph
        current_step += 1
        print_step_header(current_step, total_steps, "Build Mask Connectivity Graph")
        step_start = time.time()

        build_mask_graph_cli(K=args.K, tau=args.tau)

        print_step_complete(time.time() - step_start)

        # Step 4: Compute bounding boxes
        current_step += 1
        print_step_header(current_step, total_steps, "Compute 3D Bounding Boxes")
        step_start = time.time()

        get_all_bbox_corners_cli(percentile=args.percentile)

        print_step_complete(time.time() - step_start)

        # Step 5: Project to 2D
        current_step += 1
        print_step_header(current_step, total_steps, "Project Bounding Boxes to Images")
        step_start = time.time()

        project_all_bboxes_cli(min_fraction=args.min_fraction)

        print_step_complete(time.time() - step_start)

        # Step 6: Crop images
        current_step += 1
        print_step_header(current_step, total_steps, "Crop Images")
        step_start = time.time()

        crop_all_images_cli()

        print_step_complete(time.time() - step_start)

        # Pipeline complete
        total_time = time.time() - pipeline_start
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(
            f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        print(f"\nResults saved to: {config.get('outputs_dir')}")
        print("\nOutput structure:")
        print("  ├── masks/              - SAM segmentation masks")
        print("  ├── associations/       - 2D-3D point associations")
        print("  ├── mask_graph.gpickle  - Mask connectivity graph")
        print("  ├── connected_components.json")
        print("  ├── bbox_corners.json   - 3D bounding box corners")
        print("  ├── image_crop_coordinates.json")
        print("  ├── crop_stats.json")
        print("  └── crops/              - Cropped image regions")
        print("      ├── component_0/")
        print("      ├── component_1/")
        print("      └── manifest.json")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed at step {current_step}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
