"""Run FastSAM on images and save masks in COCO format (Optimized)."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from ultralytics import FastSAM

from .io_paths import (
    get_device,
    get_fastsam_checkpoint,
    get_images_dir,
    get_masks_dir,
    get_masks_images_dir,
    load_config,
)


def save_visualization_cv2(
    image: np.ndarray,
    masks_np: np.ndarray | None,
    output_path: Path,
) -> None:
    """
    Generates and saves a side-by-side visualization using OpenCV.
    Thread-safe.
    """
    # Create the mask overlay
    vis_mask = np.zeros_like(image)

    if masks_np is not None and len(masks_np) > 0:
        # Sort masks by area so smaller masks are drawn on top of larger ones
        areas = masks_np.sum(axis=(1, 2))
        sorted_indices = np.argsort(areas)[::-1]

        # Pre-generate random colors for all masks (N, 3)
        rng = np.random.default_rng()
        colors = rng.integers(0, 255, (len(masks_np), 3), dtype=np.uint8)

        for i, idx in enumerate(sorted_indices):
            m = masks_np[idx].astype(bool)
            if not np.any(m):
                continue
            # Apply color directly to the visualization array
            vis_mask[m] = colors[i]

    # Stack images horizontally: [Original | Masks]
    combined = np.hstack((image, vis_mask))

    # Add minimal text overlay
    count = len(masks_np) if masks_np is not None else 0
    text = f"Original | Masks ({count} segments)"
    cv2.putText(
        combined,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write using OpenCV (expects BGR if loaded via cv2, which it is)
    cv2.imwrite(str(output_path), combined)


def postprocess_and_save(
    image_path: Path,
    masks_np: np.ndarray | None,
    masks_dir: Path,
    masks_images_dir: Path,
    image_idx: int,
    total_images: int,
) -> tuple[str, bool, str]:
    """
    Loads image, resizes raw masks (CPU), encodes RLE, and saves results.
    """
    try:
        # 1. Load Image (I/O bound, done in thread)
        # OpenCV loads as BGR
        img_np = cv2.imread(str(image_path))
        if img_np is None:
            raise ValueError(f"Failed to load image from {image_path}")

        H, W = img_np.shape[:2]
        anns = []

        # We need to collect the resized masks to pass them to the visualization function
        viz_masks_list = []

        if masks_np is not None and len(masks_np) > 0:
            # Check if resize is needed.
            # masks_np comes from GPU as small prototypes (e.g., 160x160) to save VRAM.
            _, mh, mw = masks_np.shape
            do_resize = (mh != H) or (mw != W)

            for m in masks_np:
                # 2. Resize on CPU (Fast & Memory Safe)
                if do_resize:
                    # cv2.resize expects (Width, Height)
                    # Use NEAREST interpolation to keep the mask binary (0 or 1)
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                # Add to list for visualization later
                viz_masks_list.append(m)

                # 3. Encode to COCO RLE
                # Ensure Fortran-contiguousness for the C-based RLE encoding function
                m_fortran = np.asfortranarray(m)
                rle = mask_utils.encode(m_fortran)

                # Decode bytes to utf-8 string so JSON can serialize it
                rle["counts"] = rle["counts"].decode("utf-8")

                anns.append(
                    {
                        "segmentation": rle,
                        "area": float(mask_utils.area(rle)),
                    }
                )

        # Stack the resized masks back into a numpy array for the visualization function
        if viz_masks_list:
            masks_for_viz = np.stack(viz_masks_list)
        else:
            masks_for_viz = None

        num_masks = len(anns)

        # 4. Save Masks JSON using orjson
        json_path = masks_dir / f"{image_path.stem}_masks.json"
        with open(json_path, "wb") as f:
            f.write(orjson.dumps(anns))

        # 5. Save Visualization
        viz_output_path = masks_images_dir / f"{image_path.stem}_visualization.jpg"
        save_visualization_cv2(img_np, masks_for_viz, viz_output_path)

        message = (
            f"  [{image_idx}/{total_images}] {image_path.name}: {num_masks} masks saved"
        )
        return (image_path.name, True, message)

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = f"  [{image_idx}/{total_images}] {image_path.name}: Error - {e}"
        return (image_path.name, False, error_msg)


def run_fastsam_on_images(
    dataset_name: str,
    imgsz: int = 1024,
    conf: float = 0.4,
    iou: float = 0.7,
    batch_size: int = 64,
) -> None:
    import time

    start_time = time.time()

    total_failures = 0

    # Load configuration
    config = load_config(dataset_name)
    images_dir = get_images_dir(config)
    masks_dir = get_masks_dir(config)
    masks_images_dir = get_masks_images_dir(config)
    fastsam_checkpoint = get_fastsam_checkpoint(config)
    device = get_device(config)

    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Output Device: {device}")

    # Load Model
    print("Loading FastSAM model...")
    model = FastSAM(str(fastsam_checkpoint))
    model.to(device)

    # Scan for images
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

    # Batch Processing Loop
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    # Determine number of workers for I/O parallelism
    # Leave 1 or 2 cores free for OS responsiveness, GPU communication
    num_workers = max(1, os.cpu_count() - 2)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        batch_str_paths = [str(p) for p in batch_paths]

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}...")

        try:
            # 1. Inference
            results = model(
                batch_str_paths,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                retina_masks=False,
                device=device,
                verbose=False,
            )

            # 2. Prepare Data for Threads (GPU -> CPU)
            tasks = []
            for i, res in enumerate(results):
                orig_h, orig_w = res.orig_shape

                masks_data = res.masks.data if res.masks is not None else None
                masks_np = None

                if masks_data is not None:
                    # Convert to uint8 on CPU immediately
                    masks_np = masks_data.cpu().numpy().astype(np.uint8)

                tasks.append((batch_paths[i], masks_np))

            # 3. Clean GPU memory for next batch
            del results
            torch.cuda.empty_cache()

            # 4. Parallel Post-processing (I/O Bound)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []

                for idx, (p_path, p_masks) in enumerate(tasks):

                    future = executor.submit(
                        postprocess_and_save,
                        p_path,
                        p_masks,
                        masks_dir,
                        masks_images_dir,
                        start_idx + idx + 1,
                        len(image_paths),
                    )
                    futures.append(future)

                # Wait for completion
                for future in as_completed(futures):
                    _, success, msg = future.result()
                    if not success:
                        print(msg)
                        total_failures += 1
                    else:
                        pass

        except Exception as e:
            print(f"  CRITICAL BATCH ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Final Cleanup
    del model
    torch.cuda.empty_cache()

    # Calculate and print stats
    end_time = time.time()
    total_time = end_time - start_time
    images_per_second = len(image_paths) / total_time if total_time > 0 else 0

    stats = {
        "total_images_processed": len(image_paths),
        "number_of_failures": total_failures,
        "total_time_seconds": round(total_time, 2),
        "images_per_second": round(images_per_second, 2),
    }

    # Save stats to JSON
    stats_path = Path(masks_dir).parent / "fast_sam_stats.json"
    with open(stats_path, "wb") as f:
        f.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))

    print(f"\n{'='*60}")
    print(f"SEGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(orjson.dumps(stats, option=orjson.OPT_INDENT_2).decode())
    print(f"{'='*60}")
    print(f"\nResults saved to {masks_dir}")
    print(f"Stats saved to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastSAM on images (Optimized)")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_fastsam_on_images(
        dataset_name=args.dataset_name,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
