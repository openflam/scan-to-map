"""Associate 2D mask observations with 3D point IDs from COLMAP in parallel."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm  # Requires: pip install tqdm

# Type aliases
RLE = Dict[str, Any]


def get_mask_bbox_areas(rle_list: List[RLE]) -> np.ndarray:
    """Calculate the area of bounding boxes for a list of RLEs efficiently."""
    bboxes = mask_utils.toBbox([ann["segmentation"] for ann in rle_list])
    return bboxes[:, 2] * bboxes[:, 3]


def build_id_map(rle_list: List[RLE], areas: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Painter's Algorithm: Paint mask indices onto the canvas (Large -> Small).
    """
    id_map = np.full((H, W), -1, dtype=np.int32)
    sorted_indices = np.argsort(areas)[::-1]

    for mask_idx in sorted_indices:
        rle = rle_list[mask_idx]["segmentation"]
        mask_bool = mask_utils.decode(rle).astype(bool)

        if mask_bool.shape[:2] != (H, W):
            continue

        id_map[mask_bool] = mask_idx

    return id_map


def points_in_masks_vectorized(
    xys: np.ndarray, point3D_ids: np.ndarray, rle_list: List[RLE], H: int, W: int
) -> List[List[int]]:
    """Association of 3D points to masks."""
    num_masks = len(rle_list)
    mask_point_sets = [set() for _ in range(num_masks)]

    if num_masks == 0 or len(xys) == 0:
        return [[] for _ in range(num_masks)]

    areas = get_mask_bbox_areas(rle_list)
    id_map = build_id_map(rle_list, areas, H, W)

    valid_mask = point3D_ids != -1
    valid_xys = xys[valid_mask].astype(int)
    valid_p3d = point3D_ids[valid_mask]

    x, y = valid_xys[:, 0], valid_xys[:, 1]

    # Vectorized bounds check
    bounds_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    final_x = x[bounds_mask]
    final_y = y[bounds_mask]
    final_p3d = valid_p3d[bounds_mask]

    # Advanced indexing to sample the ID map
    associated_mask_indices = id_map[final_y, final_x]

    for i, mask_idx in enumerate(associated_mask_indices):
        if mask_idx != -1:
            mask_point_sets[mask_idx].add(int(final_p3d[i]))

    return [sorted(list(s)) for s in mask_point_sets]


def load_masks_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        masks_data = json.load(f)
    if isinstance(masks_data, list):
        return masks_data
    elif isinstance(masks_data, dict) and "annotations" in masks_data:
        return masks_data["annotations"]
    else:
        raise ValueError(f"Unexpected masks data format in {path}")


# --- WORKER FUNCTION ---
def process_single_image(
    image_id: int,
    image_name: str,
    xys: np.ndarray,
    point3D_ids: np.ndarray,
    H: int,
    W: int,
    mask_path: Path,
    output_path: Path,
) -> Optional[str]:
    """
    Worker function to process a single image.
    Returns the image name if successful, None if skipped/failed.
    """
    if not mask_path.exists():
        return None  # Skip

    try:
        masks_rle = load_masks_file(mask_path)

        # Heavy lifting happens here
        mask_point_lists = points_in_masks_vectorized(xys, point3D_ids, masks_rle, H, W)

        output_data = {
            "image_id": image_id,
            "image_name": image_name,
            "mask_point3d_sets": mask_point_lists,
            "H": H,
            "W": W,
        }

        # Write inside the worker to distribute I/O load
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        return image_name

    except Exception as e:
        print(f"[{image_id}] Error processing {image_name}: {e}")
        return None


def associate_all_images(dataset_name: str) -> None:
    # Local imports to avoid global scope pollution in workers
    from .colmap_io import index_image_metadata, load_colmap_model
    from .io_paths import (
        get_colmap_model_dir,
        get_associations_dir,
        get_masks_dir,
        load_config,
    )

    config = load_config(dataset_name)
    masks_dir = get_masks_dir(config)
    associations_dir = get_associations_dir(config)
    colmap_model_dir = get_colmap_model_dir(config)

    associations_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COLMAP model...")
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))
    image_metadata = index_image_metadata(images)

    # Prepare tasks
    tasks = []

    print("Preparing tasks...")
    for image_id, metadata in image_metadata.items():
        image_name = metadata["name"]
        camera_id = images[image_id].camera_id
        camera = cameras[camera_id]

        # Extract primitives
        W, H = int(camera.width), int(camera.height)
        image_stem = Path(image_name).stem
        mask_path = masks_dir / f"{image_stem}_masks.json"
        output_path = associations_dir / f"imageId_{image_id}.json"

        # Pack arguments
        tasks.append(
            (
                int(image_id),
                image_name,
                metadata["xys"],
                metadata["point3D_ids"],
                H,
                W,
                mask_path,
                output_path,
            )
        )

    # Determine workers - leave 1 or 2 cores free for OS responsiveness
    max_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Starting processing on {max_workers} cores...")

    processed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_image, *t) for t in tasks]

        # Monitor with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), unit="img"):
            result = future.result()
            if result:
                processed_count += 1
            else:
                skipped_count += 1

    print(f"Done. Processed {processed_count}, Skipped {skipped_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Associate 2D masks with 3D points")
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    associate_all_images(args.dataset_name)


if __name__ == "__main__":
    # Ensure safe multiprocessing on Windows/macOS
    multiprocessing.freeze_support()
    main()
