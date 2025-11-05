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
7. Generate captions with VLM
8. Generate CLIP embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.sam_runner import run_sam_on_images
from src.associate2d3d import associate_all_images
from src.mask_graph import build_mask_graph_cli
from src.bbox_corners import get_all_bbox_corners_cli
from src.project_bbox import project_all_bboxes_cli
from src.crop_images import crop_all_images_cli
from src.captioning import caption_all_components_cli
from src.clip_embed import generate_clip_embeddings_cli
from src.io_paths import load_config
from config import list_datasets
import os


def print_step_header(step_num: int, total_steps: int, title: str) -> None:
    """Print a formatted step header."""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 80 + "\n")


def print_step_complete(elapsed_time: float) -> None:
    """Print step completion message."""
    print(f"\n✓ Step completed in {elapsed_time:.2f} seconds")


def run_pipeline(
    dataset_name: str,
    skip_sam: bool = False,
    skip_association: bool = False,
    skip_caption: bool = False,
    skip_clip: bool = False,
    K: int = 5,
    tau: float = 0.2,
    min_points_in_3D_segment: int = 100,
    percentile: float = 95.0,
    min_fraction: float = 0.3,
    caption_n_images: int = 5,
    captioner_type: str = "vllm",
    caption_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    caption_device: int = 0,
    caption_batch_size: int = 4,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    clip_batch_size: int = 32,
    clip_device: int = 0,
) -> None:
    """
    Run the complete pipeline.

    Args:
        dataset_name: Name of the dataset to process
        skip_sam: Skip SAM segmentation step
        skip_association: Skip 2D-3D association step
        skip_caption: Skip VLM captioning step
        skip_clip: Skip CLIP embedding generation step
        K: Number of nearest neighbors for mask graph
        tau: Jaccard similarity threshold for mask graph
        percentile: Percentile threshold for bbox outlier removal
        min_fraction: Minimum fraction of visible points for projection
        caption_n_images: Number of top images to use for captioning
        captioner_type: Type of captioner to use (e.g., "vllm")
        caption_model: VLM model to use for captioning
        caption_device: GPU device ID for captioning
        caption_batch_size: Batch size for captioning inference
        clip_model: OpenCLIP model name for embeddings
        clip_pretrained: Pretrained weights for CLIP model
        clip_batch_size: Batch size for CLIP embedding generation
        clip_device: GPU device ID for CLIP embeddings
    """
    # Load and display configuration
    print("\n" + "=" * 80)
    print("SCAN-TO-MAP PIPELINE")
    print("=" * 80)

    # Set environment variable for dataset name so all modules can access it
    os.environ["SCAN_TO_MAP_DATASET"] = dataset_name

    try:
        config = load_config(dataset_name=dataset_name)
        print(f"\nConfiguration loaded for dataset: {dataset_name}")
        print(f"  Images directory: {config.get('images_dir')}")
        print(f"  COLMAP model: {config.get('colmap_model_dir')}")
        print(f"  SAM checkpoint: {config.get('sam_ckpt')}")
        print(f"  Device: {config.get('device')}")
        print(f"  Outputs directory: {config.get('outputs_dir')}")
    except Exception as e:
        print(f"\nError loading configuration: {e}")
        sys.exit(1)

    print("\nPipeline Parameters:")
    if skip_sam:
        print("  SAM segmentation: SKIPPED")
    if skip_association:
        print("  2D-3D association: SKIPPED")
    if skip_caption:
        print("  VLM captioning: SKIPPED")
    if skip_clip:
        print("  CLIP embedding generation: SKIPPED")
    print(f"  Mask graph K: {K}")
    print(f"  Mask graph tau: {tau}")
    print(f"  Bbox percentile: {percentile}")
    print(f"  Min fraction: {min_fraction}")
    if not skip_caption:
        print(f"  Captioner type: {captioner_type}")
        print(f"  Caption model: {caption_model}")
        print(f"  Caption n_images: {caption_n_images}")
        print(f"  Caption device: {caption_device}")
        print(f"  Caption batch size: {caption_batch_size}")
    if not skip_clip:
        print(f"  CLIP model: {clip_model}")
        print(f"  CLIP pretrained: {clip_pretrained}")
        print(f"  CLIP batch size: {clip_batch_size}")
        print(f"  CLIP device: {clip_device}")

    # Track overall timing
    pipeline_start = time.time()
    total_steps = 8
    current_step = 0

    # Dictionary to store runtime statistics
    runtime_stats = {
        "dataset_name": dataset_name,
        "pipeline_start_time": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(pipeline_start)
        ),
        "steps": {},
        "parameters": {
            "K": K,
            "tau": tau,
            "min_points_in_3D_segment": min_points_in_3D_segment,
            "percentile": percentile,
            "min_fraction": min_fraction,
            "caption_n_images": caption_n_images,
            "captioner_type": captioner_type,
            "caption_model": caption_model,
            "caption_device": caption_device,
            "caption_batch_size": caption_batch_size,
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "clip_batch_size": clip_batch_size,
            "clip_device": clip_device,
            "skip_sam": skip_sam,
            "skip_association": skip_association,
            "skip_caption": skip_caption,
            "skip_clip": skip_clip,
        },
    }

    try:
        # Step 1: Run SAM
        if not skip_sam:
            current_step += 1
            print_step_header(current_step, total_steps, "Run SAM Segmentation")
            step_start = time.time()

            run_sam_on_images(dataset_name=dataset_name)

            step_time = time.time() - step_start
            runtime_stats["steps"]["1_sam_segmentation"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 1: SAM Segmentation (using existing masks)")
            runtime_stats["steps"]["1_sam_segmentation"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 2: Associate 2D-3D
        if not skip_association:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Associate 2D Masks with 3D Points"
            )
            step_start = time.time()

            associate_all_images(dataset_name=dataset_name)

            step_time = time.time() - step_start
            runtime_stats["steps"]["2_associate_2d_3d"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 2: 2D-3D Association (using existing associations)")
            runtime_stats["steps"]["2_associate_2d_3d"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 3: Build mask graph
        current_step += 1
        print_step_header(current_step, total_steps, "Build Mask Connectivity Graph")
        step_start = time.time()

        build_mask_graph_cli(
            dataset_name=dataset_name,
            K=K,
            tau=tau,
            min_points_in_3D_segment=min_points_in_3D_segment,
        )

        step_time = time.time() - step_start
        runtime_stats["steps"]["3_build_mask_graph"] = {
            "duration_seconds": step_time,
            "status": "completed",
        }
        print_step_complete(step_time)

        # Step 4: Compute bounding boxes
        current_step += 1
        print_step_header(current_step, total_steps, "Compute 3D Bounding Boxes")
        step_start = time.time()

        get_all_bbox_corners_cli(dataset_name=dataset_name, percentile=percentile)

        step_time = time.time() - step_start
        runtime_stats["steps"]["4_compute_bboxes"] = {
            "duration_seconds": step_time,
            "status": "completed",
        }
        print_step_complete(step_time)

        # Step 5: Project to 2D
        current_step += 1
        print_step_header(current_step, total_steps, "Project Bounding Boxes to Images")
        step_start = time.time()

        project_all_bboxes_cli(dataset_name=dataset_name, min_fraction=min_fraction)

        step_time = time.time() - step_start
        runtime_stats["steps"]["5_project_bboxes"] = {
            "duration_seconds": step_time,
            "status": "completed",
        }
        print_step_complete(step_time)

        # Step 6: Crop images
        current_step += 1
        print_step_header(current_step, total_steps, "Crop Images")
        step_start = time.time()

        crop_all_images_cli(dataset_name=dataset_name)

        step_time = time.time() - step_start
        runtime_stats["steps"]["6_crop_images"] = {
            "duration_seconds": step_time,
            "status": "completed",
        }
        print_step_complete(step_time)

        # Step 7: Caption components
        if not skip_caption:
            current_step += 1
            print_step_header(current_step, total_steps, "Generate Captions with VLM")
            step_start = time.time()

            caption_all_components_cli(
                dataset_name=dataset_name,
                n_images=caption_n_images,
                captioner_type=captioner_type,
                model=caption_model,
                device=caption_device,
                batch_size=caption_batch_size,
            )

            step_time = time.time() - step_start
            runtime_stats["steps"]["7_caption_components"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 7: VLM Captioning")
            runtime_stats["steps"]["7_caption_components"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 8: Generate CLIP embeddings
        if not skip_clip:
            current_step += 1
            print_step_header(current_step, total_steps, "Generate CLIP Embeddings")
            step_start = time.time()

            generate_clip_embeddings_cli(
                dataset_name=dataset_name,
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=clip_device,
                batch_size=clip_batch_size,
            )

            step_time = time.time() - step_start
            runtime_stats["steps"]["8_clip_embeddings"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 8: CLIP Embedding Generation")
            runtime_stats["steps"]["8_clip_embeddings"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Pipeline complete
        total_time = time.time() - pipeline_start

        # Add total time to runtime stats
        runtime_stats["total_duration_seconds"] = total_time
        runtime_stats["total_duration_minutes"] = total_time / 60
        runtime_stats["pipeline_end_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )

        # Save runtime statistics to file
        outputs_dir = Path(config.get("outputs_dir"))
        runtime_stats_path = outputs_dir / "runtime_stats.json"
        with runtime_stats_path.open("w", encoding="utf-8") as f:
            json.dump(runtime_stats, f, indent=2)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(
            f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        print(f"\nResults saved to: {config.get('outputs_dir')}")
        print(f"Runtime statistics saved to: {runtime_stats_path}")
        print("\nOutput structure:")
        print("  ├── masks/              - SAM segmentation masks")
        print("  ├── masks_images/       - SAM visualization images")
        print("  ├── associations/       - 2D-3D point associations")
        print("  ├── mask_graph.gpickle  - Mask connectivity graph")
        print("  ├── connected_components.json")
        print("  ├── bbox_corners.json   - 3D bounding box corners")
        print("  ├── image_crop_coordinates.json")
        print("  ├── crop_stats.json")
        print("  ├── runtime_stats.json  - Pipeline execution times")
        print("  ├── crops/              - Cropped image regions")
        print("  │   ├── component_0/")
        print("  │   ├── component_1/")
        print("  │   └── manifest.json")
        print("  ├── component_captions.json - VLM-generated captions")
        print("  ├── clip_embeddings.json    - CLIP embeddings (JSON)")
        print("  ├── clip_embeddings.npz     - CLIP embeddings (numpy)")
        print("  └── clip_embedding_stats.json")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed at step {current_step}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
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
  7. Generate captions for each component using VLM
  8. Generate CLIP embeddings for each component

Configuration:
  Dataset configurations are defined in segment3d/config.py
        """,
    )

    # Dataset selection
    available_datasets = list_datasets()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=available_datasets,
        help=f"Dataset to process (required). Available: {', '.join(available_datasets)}",
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
        "--skip-caption",
        action="store_true",
        help="Skip VLM captioning step",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP embedding generation step",
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
        "--min-points-in-3D-segment",
        type=int,
        default=5,
        help="Minimum points in 3D segment for mask graph (default: 5)",
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
    parser.add_argument(
        "--caption-n-images",
        type=int,
        default=1,
        help="Number of top images to use for captioning (default: 1)",
    )
    parser.add_argument(
        "--captioner-type",
        type=str,
        default="vllm",
        help="Type of captioner to use (default: vllm)",
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model to use for captioning (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--caption-device",
        type=int,
        default=0,
        help="GPU device ID for captioning (default: 0)",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=1024,
        help="Batch size for captioning inference (default: 1024)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name for embeddings (default: ViT-B-32)",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="Pretrained weights for CLIP model (default: laion2b_s34b_b79k)",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP embedding generation (default: 32)",
    )
    parser.add_argument(
        "--clip-device",
        type=int,
        default=0,
        help="GPU device ID for CLIP embeddings (default: 0)",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.tau <= 0 or args.tau > 1:
        parser.error("--tau must be between 0 and 1")
    if args.percentile <= 0 or args.percentile > 100:
        parser.error("--percentile must be between 0 and 100")
    if args.min_fraction <= 0 or args.min_fraction > 1:
        parser.error("--min-fraction must be between 0 and 1")
    if args.caption_n_images < 1:
        parser.error("--caption-n-images must be at least 1")
    if args.caption_batch_size < 1:
        parser.error("--caption-batch-size must be at least 1")
    if args.clip_batch_size < 1:
        parser.error("--clip-batch-size must be at least 1")

    # Run the pipeline with parsed arguments
    run_pipeline(
        dataset_name=args.dataset,
        skip_sam=args.skip_sam,
        skip_association=args.skip_association,
        skip_caption=args.skip_caption,
        skip_clip=args.skip_clip,
        K=args.K,
        tau=args.tau,
        min_points_in_3D_segment=args.min_points_in_3D_segment,
        percentile=args.percentile,
        min_fraction=args.min_fraction,
        caption_n_images=args.caption_n_images,
        captioner_type=args.captioner_type,
        caption_model=args.caption_model,
        caption_device=args.caption_device,
        caption_batch_size=args.caption_batch_size,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_batch_size=args.clip_batch_size,
        clip_device=args.clip_device,
    )
