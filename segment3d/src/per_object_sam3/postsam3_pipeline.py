#!/usr/bin/env python3
"""
Post-SAM3 pipeline script that runs all steps after SAM3 segmentation.

This script orchestrates the per-object pipeline starting from 2D-3D association:
1. Associate per-object SAM3 masks with COLMAP 3D points
2. Build object mask connectivity graph
3. Clean connected components (DBSCAN noise removal + splitting)
4. Compute 3D bounding boxes for each connected component
5. Segment (crop) images using connected component masks
6. Generate captions with VLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .associate2d3d import associate_per_object
from .clean_components import clean_connected_components
from .mask_graph import build_object_mask_graph
from .segment_crops import segment_crops_cli
from ..bbox_corners import get_all_bbox_corners_cli
from ..captioning import caption_all_components_cli
from ..clip_embed import generate_clip_embeddings_cli
from ..io_paths import load_config
from config import list_datasets, DEFAULT_PARAMETERS


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
    skip_association: bool = False,
    skip_graph: bool = False,
    skip_clean: bool = False,
    skip_bbox: bool = False,
    skip_segment_crops: bool = False,
    skip_caption: bool = False,
    skip_clip: bool = False,
    # Mask graph parameters
    K: int = 30,
    tau: float = 0.8,
    min_points: int = 1,
    min_points_in_3d_segment: int = 10,
    # Clean components parameters
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 5,
    dbscan_min_points: int = 20,
    # Bounding box parameters
    percentile: float = 95.0,
    # Segment crops parameters
    top_n: int = 5,
    min_fraction: float = 0.05,
    # Captioning parameters
    caption_n_images: int = 1,
    captioner_type: str = "vllm",
    caption_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    caption_device: int = 0,
    caption_batch_size: int = 4,
    # CLIP embedding parameters
    clip_model: str = "ViT-H-14",
    clip_pretrained: str = "laion2B-s32B-b79K",
    clip_batch_size: int = 32,
    clip_device: int = 0,
) -> None:
    """
    Run the post-SAM3 pipeline (all steps after SAM3 segmentation).

    Args:
        dataset_name: Name of the dataset to process
        skip_association: Skip 2D-3D association step
        skip_graph: Skip object mask graph building step
        skip_clean: Skip DBSCAN component cleaning step
        skip_bbox: Skip 3D bounding box computation step
        skip_segment_crops: Skip image cropping step
        skip_caption: Skip VLM captioning step
        skip_clip: Skip CLIP embedding generation step
        K: Min overlap count threshold for mask graph edges
        tau: Min Jaccard similarity threshold for mask graph edges
        min_points: Min 3D points for a node to be included in the graph
        min_points_in_3d_segment: Min 3D points in a component to be reported
        dbscan_eps: DBSCAN neighbourhood radius in world units
        dbscan_min_samples: DBSCAN minimum samples per core point
        dbscan_min_points: Drop components with fewer than this many points after cleaning
        percentile: Percentile threshold for bbox outlier removal
        top_n: Number of top frames to crop per component
        min_fraction: Minimum visibility fraction to consider a frame for cropping
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
    print("\n" + "=" * 80)
    print("POST-SAM3 PIPELINE (per-object)")
    print("=" * 80)

    os.environ["SCAN_TO_MAP_DATASET"] = dataset_name

    try:
        config = load_config(dataset_name=dataset_name)
        print(f"\nConfiguration loaded for dataset: {dataset_name}")
        print(f"  Images directory: {config.get('images_dir')}")
        print(f"  COLMAP model: {config.get('colmap_model_dir')}")
        print(f"  Outputs directory: {config.get('outputs_dir')}")
    except Exception as e:
        print(f"\nError loading configuration: {e}")
        sys.exit(1)

    print("\nPipeline Parameters:")
    if skip_association:
        print("  2D-3D association: SKIPPED")
    if skip_graph:
        print("  Mask graph building: SKIPPED")
    if skip_clean:
        print("  Component cleaning: SKIPPED")
    if skip_bbox:
        print("  3D bounding boxes: SKIPPED")
    if skip_segment_crops:
        print("  Image cropping: SKIPPED")
    if skip_caption:
        print("  VLM captioning: SKIPPED")
    if skip_clip:
        print("  CLIP embedding generation: SKIPPED")
    print(f"  Mask graph K: {K}")
    print(f"  Mask graph tau: {tau}")
    print(f"  Mask graph min_points: {min_points}")
    print(f"  Mask graph min_points_in_3d_segment: {min_points_in_3d_segment}")
    if not skip_clean:
        print(f"  DBSCAN eps: {dbscan_eps}")
        print(f"  DBSCAN min_samples: {dbscan_min_samples}")
        print(f"  DBSCAN min_points: {dbscan_min_points}")
    if not skip_bbox:
        print(f"  Bbox percentile: {percentile}")
    print(f"  Segment crops top_n: {top_n}")
    print(f"  Segment crops min_fraction: {min_fraction}")
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

    pipeline_start = time.time()
    total_steps = 7
    current_step = 0

    runtime_stats = {
        "dataset_name": dataset_name,
        "pipeline_start_time": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(pipeline_start)
        ),
        "steps": {},
        "parameters": {
            "skip_association": skip_association,
            "skip_graph": skip_graph,
            "skip_clean": skip_clean,
            "skip_bbox": skip_bbox,
            "skip_segment_crops": skip_segment_crops,
            "skip_caption": skip_caption,
            "skip_clip": skip_clip,
            "K": K,
            "tau": tau,
            "min_points": min_points,
            "min_points_in_3d_segment": min_points_in_3d_segment,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "dbscan_min_points": dbscan_min_points,
            "percentile": percentile,
            "top_n": top_n,
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
        },
    }

    try:
        # Step 1: Associate 2D-3D
        if not skip_association:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Associate 2D Masks with 3D Points"
            )
            step_start = time.time()

            associate_per_object(dataset_name=dataset_name)

            step_time = time.time() - step_start
            runtime_stats["steps"]["1_associate_2d_3d"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 1: 2D-3D Association (using existing associations)")
            runtime_stats["steps"]["1_associate_2d_3d"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 2: Build object mask graph
        if not skip_graph:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Build Object Mask Connectivity Graph"
            )
            step_start = time.time()

            build_object_mask_graph(
                dataset_name=dataset_name,
                K=K,
                tau=tau,
                min_points=min_points,
                min_points_in_3d_segment=min_points_in_3d_segment,
            )

            step_time = time.time() - step_start
            runtime_stats["steps"]["2_build_mask_graph"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 2: Mask Graph Building (using existing graph)")
            runtime_stats["steps"]["2_build_mask_graph"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 3: Clean connected components
        if not skip_clean:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Clean Connected Components (DBSCAN)"
            )
            step_start = time.time()

            clean_connected_components(
                dataset_name=dataset_name,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                min_points=dbscan_min_points,
            )

            step_time = time.time() - step_start
            runtime_stats["steps"]["3_clean_components"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print(
                "\nSkipping Step 3: Component Cleaning (using existing connected_components.json)"
            )
            runtime_stats["steps"]["3_clean_components"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 4: Compute 3D bounding boxes
        if not skip_bbox:
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
        else:
            print(
                "\nSkipping Step 4: 3D Bounding Box Computation (using existing bbox_corners.json)"
            )
            runtime_stats["steps"]["4_compute_bboxes"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 5: Segment crops
        if not skip_segment_crops:
            current_step += 1
            print_step_header(current_step, total_steps, "Segment and Crop Images")
            step_start = time.time()

            segment_crops_cli(
                dataset_name=dataset_name,
                top_n=top_n,
                min_fraction=min_fraction,
            )

            step_time = time.time() - step_start
            runtime_stats["steps"]["5_segment_crops"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 5: Image Cropping (using existing crops)")
            runtime_stats["steps"]["5_segment_crops"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 6: Caption components
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
            runtime_stats["steps"]["6_caption_components"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 6: VLM Captioning")
            runtime_stats["steps"]["6_caption_components"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Step 7: Generate CLIP embeddings
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
            runtime_stats["steps"]["7_clip_embeddings"] = {
                "duration_seconds": step_time,
                "status": "completed",
            }
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 7: CLIP Embedding Generation")
            runtime_stats["steps"]["7_clip_embeddings"] = {
                "duration_seconds": 0,
                "status": "skipped",
            }

        # Pipeline complete
        total_time = time.time() - pipeline_start

        runtime_stats["total_duration_seconds"] = total_time
        runtime_stats["total_duration_minutes"] = total_time / 60
        runtime_stats["pipeline_end_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )

        outputs_dir = Path(config.get("outputs_dir"))
        runtime_stats_path = outputs_dir / "runtime_stats_postsam3.json"
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
        print("  ├── object_level_masks/")
        print("  │   ├── masks/              - SAM3 per-object mask JSON files (input)")
        print("  │   └── object_3d_associations.json")
        print("  ├── connected_components.json")
        print("  ├── mask_graph_stats.json")
        print("  ├── bbox_corners.json       - 3D bounding box corners")
        print("  ├── bbox_stats.json")
        print("  ├── crops/                  - Masked & cropped image regions")
        print("  │   ├── component_0/")
        print("  │   ├── component_1/")
        print("  │   └── manifest.json")
        print("  ├── component_captions.json - VLM-generated captions")
        print("  ├── clip_embeddings.json    - CLIP embeddings (JSON)")
        print("  ├── clip_embeddings.npz     - CLIP embeddings (numpy)")
        print("  ├── clip_embeddings.faiss   - FAISS HNSW index")
        print("  └── runtime_stats_postsam3.json")

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
        description="Run the post-SAM3 per-object pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Associate per-object SAM3 masks with COLMAP 3D points
  2. Build object mask connectivity graph and extract connected components
  3. Clean connected components (DBSCAN noise removal + multi-cluster splitting)
  4. Compute 3D bounding boxes for each connected component
  5. Segment and crop images using connected component masks
  6. Generate captions for each component using VLM

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

    # Skip flags
    parser.add_argument(
        "--skip-association",
        action="store_true",
        help="Skip 2D-3D association step (use existing associations)",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip mask graph building step (use existing graph)",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip DBSCAN component cleaning step (use existing connected_components.json)",
    )
    parser.add_argument(
        "--skip-bbox",
        action="store_true",
        help="Skip 3D bounding box computation step (use existing bbox_corners.json)",
    )
    parser.add_argument(
        "--skip-segment-crops",
        action="store_true",
        help="Skip image cropping step (use existing crops)",
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

    # Mask graph parameters
    parser.add_argument(
        "--K",
        type=int,
        default=30,
        help=f"Min overlap count threshold for mask graph edges (default: 30)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.8,
        help=f"Min Jaccard similarity threshold for mask graph edges (default: 0.8)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Min 3D points for a node to be included in the graph (default: 1)",
    )
    parser.add_argument(
        "--min-points-in-3d-segment",
        type=int,
        default=10,
        help="Min 3D points in a connected component to be reported (default: 10)",
    )

    # Clean components parameters
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.1,
        help="DBSCAN neighbourhood radius in world units (default: 0.1)",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples per core point (default: 5)",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=20,
        help="Drop components with fewer than this many points after cleaning (default: 20)",
    )

    # Bounding box parameters
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PARAMETERS["percentile"],
        help=f"Percentile threshold for bbox outlier removal (default: {DEFAULT_PARAMETERS['percentile']})",
    )

    # Segment crops parameters
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top frames to crop per component (default: 5)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.05,
        help="Minimum visibility fraction to consider a frame for cropping (default: 0.05)",
    )

    # Captioning parameters
    parser.add_argument(
        "--caption-n-images",
        type=int,
        default=DEFAULT_PARAMETERS["caption_n_images"],
        help=f"Number of top images to use for captioning (default: {DEFAULT_PARAMETERS['caption_n_images']})",
    )
    parser.add_argument(
        "--captioner-type",
        type=str,
        default=DEFAULT_PARAMETERS["captioner_type"],
        help=f"Type of captioner to use (default: {DEFAULT_PARAMETERS['captioner_type']})",
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default=DEFAULT_PARAMETERS["caption_model"],
        help=f"VLM model to use for captioning (default: {DEFAULT_PARAMETERS['caption_model']})",
    )
    parser.add_argument(
        "--caption-device",
        type=int,
        default=DEFAULT_PARAMETERS["caption_device"],
        help=f"GPU device ID for captioning (default: {DEFAULT_PARAMETERS['caption_device']})",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=DEFAULT_PARAMETERS["caption_batch_size"],
        help=f"Batch size for captioning inference (default: {DEFAULT_PARAMETERS['caption_batch_size']})",
    )

    # CLIP embedding parameters
    parser.add_argument(
        "--clip-model",
        type=str,
        default=DEFAULT_PARAMETERS["clip_model"],
        help=f"OpenCLIP model name for embeddings (default: {DEFAULT_PARAMETERS['clip_model']})",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=DEFAULT_PARAMETERS["clip_pretrained"],
        help=f"Pretrained weights for CLIP model (default: {DEFAULT_PARAMETERS['clip_pretrained']})",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=DEFAULT_PARAMETERS["clip_batch_size"],
        help=f"Batch size for CLIP embedding generation (default: {DEFAULT_PARAMETERS['clip_batch_size']})",
    )
    parser.add_argument(
        "--clip-device",
        type=int,
        default=DEFAULT_PARAMETERS["clip_device"],
        help=f"GPU device ID for CLIP embeddings (default: {DEFAULT_PARAMETERS['clip_device']})",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.tau <= 0 or args.tau > 1:
        parser.error("--tau must be between 0 and 1")
    if args.min_fraction <= 0 or args.min_fraction > 1:
        parser.error("--min-fraction must be between 0 and 1")
    if args.K < 1:
        parser.error("--K must be at least 1")
    if args.min_points < 1:
        parser.error("--min-points must be at least 1")
    if args.min_points_in_3d_segment < 1:
        parser.error("--min-points-in-3d-segment must be at least 1")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.caption_n_images < 1:
        parser.error("--caption-n-images must be at least 1")
    if args.caption_batch_size < 1:
        parser.error("--caption-batch-size must be at least 1")
    if args.clip_batch_size < 1:
        parser.error("--clip-batch-size must be at least 1")
    if args.percentile <= 0 or args.percentile > 100:
        parser.error("--percentile must be between 0 and 100")
    if args.dbscan_eps <= 0:
        parser.error("--dbscan-eps must be positive")
    if args.dbscan_min_samples < 1:
        parser.error("--dbscan-min-samples must be at least 1")
    if args.dbscan_min_points < 1:
        parser.error("--dbscan-min-points must be at least 1")

    run_pipeline(
        dataset_name=args.dataset,
        skip_association=args.skip_association,
        skip_graph=args.skip_graph,
        skip_clean=args.skip_clean,
        skip_bbox=args.skip_bbox,
        skip_segment_crops=args.skip_segment_crops,
        skip_caption=args.skip_caption,
        skip_clip=args.skip_clip,
        K=args.K,
        tau=args.tau,
        min_points=args.min_points,
        min_points_in_3d_segment=args.min_points_in_3d_segment,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_min_points=args.dbscan_min_points,
        percentile=args.percentile,
        top_n=args.top_n,
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
