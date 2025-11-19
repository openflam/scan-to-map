"""Run FastSAM on images and save masks in COCO format."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

_MPL_LOCK = Lock()  # guard Matplotlib rendering in multi-threaded code
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


def show_anns_side_by_side(
    image: np.ndarray,
    masks_np: np.ndarray | None,
    output_path: str | Path,
    figsize: tuple[float, float] = (20, 20),
) -> None:
    """
    Displays the original image on the left and the mask on the right.
    Saves the result to output_path.

    Args:
        image: (H, W, 3) numpy array
        masks_np: numpy array of binary masks (N, H, W) or None
        output_path: Path to save the figure
        figsize: Figure size tuple
    """
    # ---- Validate and prep IO path ----
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("`image` must be HxWx3 RGB array")

    H, W = image.shape[:2]

    # ---- Build overlay outside the Matplotlib lock ----
    overlay = None
    title_right = "Masks (0 segments)"
    if masks_np is not None and len(masks_np) > 0:
        # Ensure masks are HxW and aligned
        if masks_np.ndim != 3 or masks_np.shape[1:] != (H, W):
            raise ValueError("`masks_np` must be (N, H, W) matching image size")

        # Sort masks by area (desc)
        areas = [int(m.sum()) for m in masks_np]
        order = np.argsort(areas)[::-1]

        # Create transparent RGBA overlay [0,1]
        overlay = np.zeros((H, W, 4), dtype=np.float32)

        # Thread-local RNG (don’t share global np.random state across threads)
        rng = np.random.default_rng()

        for idx in order:
            m = masks_np[idx].astype(bool, copy=False)
            if not m.any():
                continue
            # Random RGB + fixed alpha
            rgb = rng.random(3, dtype=np.float32)
            color = np.concatenate([rgb, np.array([0.6], dtype=np.float32)])
            overlay[m] = color

        title_right = f"Masks ({len(masks_np)} segments)"

    # ---- Create figure (OO API), render within a lock ----
    fig = Figure(figsize=figsize, constrained_layout=True)
    ax0, ax1 = fig.subplots(1, 2)

    # Set figure size proportional to image dimensions
    dpi = 100
    fig.set_size_inches((2 * W) / dpi, H / dpi)  # width = 2 images side-by-side

    # LEFT: original image
    ax0.imshow(image)
    ax0.axis("off")
    ax0.set_title("Original Image")

    # RIGHT: overlay (or empty)
    if overlay is not None:
        ax1.imshow(overlay)
    else:
        # Show an empty axes with a title if no masks
        ax1.imshow(np.zeros((H, W, 4), dtype=np.float32))
    ax1.axis("off")
    ax1.set_title(title_right)

    canvas = FigureCanvas(fig)

    # Matplotlib rendering/saving is not fully thread-safe → guard it
    with _MPL_LOCK:
        # Save as JPEG with bbox tightening similar to bbox_inches="tight"
        canvas.print_jpg(str(output_path))

    # Help GC
    fig.clear()


def postprocess_and_save(
    image_path: Path,
    img_np: np.ndarray,
    masks_np: np.ndarray | None,
    masks_dir: Path,
    masks_images_dir: Path,
    image_idx: int,
    total_images: int,
) -> tuple[str, bool, str]:
    """
    Post-process masks and save results for a single image.

    Args:
        image_path: Path to the image file
        img_np: Numpy array of the image (H, W, 3)
        masks_np: Numpy array of binary masks (N, H, W) or None
        masks_dir: Directory to save mask JSON files
        masks_images_dir: Directory to save visualization images
        image_idx: Index of current image (1-based)
        total_images: Total number of images

    Returns:
        Tuple of (image_name, success, message)
    """
    try:
        if masks_np is None or len(masks_np) == 0:
            anns = []
            num_masks = 0
        else:
            # Convert to fortran-contiguous and encode to COCO RLE format
            anns = []
            for m in masks_np:
                m_fortran = np.asfortranarray(m)
                rle = mask_utils.encode(m_fortran)
                # Decode bytes to string for JSON compatibility
                rle["counts"] = rle["counts"].decode("utf-8")
                anns.append(
                    {
                        "segmentation": rle,
                        "area": float(mask_utils.area(rle)),
                    }
                )
            num_masks = len(anns)

        # Save masks JSON
        output_path = masks_dir / f"{image_path.stem}_masks.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(anns, f)

        # Save visualization
        viz_output_path = masks_images_dir / f"{image_path.stem}_visualization.jpg"
        show_anns_side_by_side(img_np, masks_np, viz_output_path)

        message = (
            f"  [{image_idx}/{total_images}] {image_path.name}: {num_masks} masks saved"
        )
        return (image_path.name, True, message)

    except Exception as e:
        import traceback

        error_msg = f"  [{image_idx}/{total_images}] {image_path.name}: Error - {e}"
        traceback.print_exc()
        return (image_path.name, False, error_msg)


def run_fastsam_on_images(
    dataset_name: str,
    imgsz: int = 1024,
    conf: float = 0.4,
    iou: float = 0.7,
    batch_size: int = 64,
    num_workers: int = 4,
) -> None:
    """Run FastSAM on all images and save masks in COCO format.

    Args:
        dataset_name: Name of the dataset to process
        imgsz: Input image size for FastSAM
        conf: Confidence threshold for FastSAM
        iou: IoU threshold for NMS in FastSAM
        batch_size: Number of images to process in a batch
        num_workers: Number of worker threads for parallel I/O operations
    """
    # Load configuration
    config = load_config(dataset_name)

    images_dir = get_images_dir(config)
    masks_dir = get_masks_dir(config)
    masks_images_dir = get_masks_images_dir(config)
    fastsam_checkpoint = get_fastsam_checkpoint(config)
    device = get_device(config)

    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Masks images directory: {masks_images_dir}")
    print(f"FastSAM checkpoint: {fastsam_checkpoint}")
    print(f"Device: {device}")
    print(
        f"FastSAM parameters: imgsz={imgsz}, conf={conf}, iou={iou}, batch_size={batch_size}, num_workers={num_workers}"
    )

    # Load FastSAM model
    print("Loading FastSAM model...")
    model = FastSAM(str(fastsam_checkpoint)).to(device)

    # Get all image files
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

    # Process images in batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]

        print(
            f"\nProcessing batch {batch_idx + 1}/{num_batches} (images {start_idx + 1}-{end_idx}/{len(image_paths)})"
        )

        try:
            # Run batch inference
            batch_str_paths = [str(p) for p in batch_paths]
            results = model(
                batch_str_paths,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                retina_masks=True,
                device=device,
            )

            # Move all masks to CPU first to free GPU memory
            cpu_masks_list = []
            for res in results:
                masks = res.masks.data if res.masks is not None else None
                if masks is None or len(masks) == 0:
                    cpu_masks_list.append(None)
                else:
                    # Move to CPU as numpy array
                    masks_np = masks.cpu().numpy()
                    cpu_masks_list.append(masks_np)

            # Free GPU memory immediately after moving to CPU
            del results
            torch.cuda.empty_cache()

            # Process each result in the batch on CPU
            # Use ThreadPoolExecutor for parallel I/O operations
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []

                for idx, (image_path, masks_cpu) in enumerate(
                    zip(batch_paths, cpu_masks_list)
                ):
                    try:
                        # Load image to get dimensions
                        img_pil = Image.open(str(image_path)).convert("RGB")
                        img_np = np.array(img_pil)
                        H, W = img_np.shape[:2]

                        if masks_cpu is None:
                            masks_np = None
                        else:
                            # Resize masks to original image size if needed
                            _, h_prime, w_prime = masks_cpu.shape
                            if (h_prime, w_prime) != (H, W):
                                # Convert to torch tensor for interpolation
                                masks_tensor = torch.from_numpy(masks_cpu).float()
                                masks_tensor = F.interpolate(
                                    masks_tensor.unsqueeze(1),
                                    size=(H, W),
                                    mode="nearest",
                                ).squeeze(1)
                                masks_np = masks_tensor.numpy().astype(np.uint8)
                                del masks_tensor
                            else:
                                masks_np = masks_cpu.astype(np.uint8)

                        # Submit post-processing and saving to thread pool
                        future = executor.submit(
                            postprocess_and_save,
                            image_path,
                            img_np,
                            masks_np,
                            masks_dir,
                            masks_images_dir,
                            start_idx + idx + 1,
                            len(image_paths),
                        )
                        futures.append(future)

                    except Exception as e:
                        print(f"  Error preparing image {image_path.name}: {e}")
                        import traceback

                        traceback.print_exc()
                        continue

                # Wait for all tasks in this batch to complete and print results
                for future in as_completed(futures):
                    image_name, success, message = future.result()
                    print(message)

        except Exception as e:
            print(f"  Error processing batch: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Clean up the model to free GPU memory
    del model
    torch.cuda.empty_cache()

    print(f"\nCompleted processing all images.")
    print(f"Masks saved to: {masks_dir}")
    print(f"Visualizations saved to: {masks_images_dir}")


def main() -> None:
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Run FastSAM on images")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Input image size for FastSAM (default: 1024)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for FastSAM (default: 0.4)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS in FastSAM (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for FastSAM inference (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel I/O (default: 4)",
    )
    args = parser.parse_args()

    run_fastsam_on_images(
        dataset_name=args.dataset_name,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
