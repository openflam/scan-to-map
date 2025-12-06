"""Run SAM 3 for floor detection on images."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor

from ..colmap_io import load_colmap_model, index_image_metadata, index_point3d
from ..io_paths import (
    get_colmap_model_dir,
    get_images_dir,
    get_outputs_dir,
    load_config,
)


def get_all_floor_masks(images_dir: Path) -> dict:
    """Run SAM 3 floor detection on all images and return outputs per frame.

    Args:
        dataset_name: Name of the dataset to process

    Returns:
        Dictionary mapping frame indices to outputs
    """

    # Set up SAM 3 predictor
    # Use all available GPUs on the machine
    gpus_to_use = range(torch.cuda.device_count())
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Load video frames from images directory
    video_path = str(images_dir)
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    video_frames_for_vis.sort()

    print(f"Found {len(video_frames_for_vis)} frames")

    # Start inference session
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"Started session: {session_id}")

    # Add text prompt for floor detection on frame 0
    prompt_text_str = "floor"
    frame_idx = 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    print(f"Added text prompt '{prompt_text_str}' on frame {frame_idx}")

    # Propagate from frame 0 to the end of the video
    print("Propagating floor detection across all frames...")
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    print(f"Completed propagation for {len(outputs_per_frame)} frames")

    # Close the inference session to free GPU resources
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    # Shutdown the predictor to free up the multi-GPU process group
    predictor.shutdown()

    return outputs_per_frame


def get_floor_3D_points(outputs_per_frame: dict, dataset_name: str) -> List[Dict]:
    """Extract 3D points that fall within floor masks from SAM 3 outputs.

    Args:
        outputs_per_frame: Dictionary mapping frame indices to SAM 3 outputs
                          Each output contains 'out_binary_masks', 'out_obj_ids', etc.
        dataset_name: Name of the dataset to load COLMAP model from

    Returns:
        List of dictionaries, each containing:
        - 'point_id': int - 3D point ID
        - 'coords': list of 3 floats - XYZ coordinates
    """
    # Load configuration and COLMAP model
    config = load_config(dataset_name)
    colmap_model_dir = get_colmap_model_dir(config)
    images_dir = get_images_dir(config)

    print(f"Loading COLMAP model from: {colmap_model_dir}")
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))
    print(f"Loaded {len(images)} images and {len(points3D)} 3D points")

    # Index the data for efficient access
    image_metadata = index_image_metadata(images)
    point3d_index = index_point3d(points3D)

    # Create mapping from image filename to image_id
    filename_to_id = {meta["name"]: img_id for img_id, meta in image_metadata.items()}

    # Get sorted list of image filenames
    image_files = sorted(glob.glob(os.path.join(str(images_dir), "*.jpg")))
    print(f"Found {len(image_files)} image files")

    # Collect all 3D points that fall within floor masks
    floor_point_ids_set = set()

    for frame_idx, frame_outputs in outputs_per_frame.items():
        if frame_idx >= len(image_files):
            print(f"Warning: frame_idx {frame_idx} exceeds number of images")
            continue

        # Get the image filename for this frame
        image_path = Path(image_files[frame_idx])
        image_name = image_path.name

        # Find corresponding COLMAP image_id
        if image_name not in filename_to_id:
            print(f"Warning: {image_name} not found in COLMAP model, skipping")
            continue

        image_id = filename_to_id[image_name]
        meta = image_metadata[image_id]

        # Get 2D keypoints and their corresponding 3D point IDs
        xys = meta["xys"]  # Shape: (N, 2)
        point3D_ids = meta["point3D_ids"]  # Shape: (N,)

        # Get floor masks for this frame
        out_binary_masks = frame_outputs["out_binary_masks"]  # Shape: (M, H, W)
        out_obj_ids = frame_outputs["out_obj_ids"]  # Shape: (M,)

        if len(out_binary_masks) == 0:
            continue

        # Combine all floor masks (union of all detected floor objects)
        H, W = out_binary_masks[0].shape
        combined_floor_mask = np.zeros((H, W), dtype=bool)
        for mask in out_binary_masks:
            combined_floor_mask = np.logical_or(combined_floor_mask, mask > 0.5)

        # Check which 2D points fall within the floor mask
        for i, (xy, p3d_id) in enumerate(zip(xys, point3D_ids)):
            if p3d_id == -1:  # Invalid 3D point
                continue

            x, y = int(xy[0]), int(xy[1])

            # Check bounds
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            # Check if point falls within floor mask
            if combined_floor_mask[y, x]:
                floor_point_ids_set.add(p3d_id)

    print(f"Found {len(floor_point_ids_set)} unique 3D points within floor masks")

    # Convert to list of dictionaries
    floor_points = []
    for pid in sorted(floor_point_ids_set):
        coords = point3d_index[pid]["xyz"]
        floor_points.append(
            {
                "point_id": int(pid),
                "coords": coords.tolist(),
            }
        )

    print(f"Extracted {len(floor_points)} floor points")

    # Save to JSON file
    outputs_dir = Path(get_outputs_dir(config))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = outputs_dir / "floor_3d_points.json"
    with open(output_file, "w") as f:
        json.dump(floor_points, f, indent=2)

    print(f"Saved floor 3D points to: {output_file}")

    return floor_points


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SAM 3 for floor detection")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to process"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.dataset)
    images_dir = get_images_dir(config)

    # Get floor masks for all frames
    outputs_per_frame = get_all_floor_masks(images_dir=images_dir)
    print(f"\nReturned outputs for {len(outputs_per_frame)} frames")

    # Extract 3D points
    print("\nExtracting 3D points within floor masks...")
    floor_points = get_floor_3D_points(outputs_per_frame, args.dataset)

    print(f"\nFloor 3D points summary:")
    print(f"  Total points: {len(floor_points)}")


if __name__ == "__main__":
    main()
