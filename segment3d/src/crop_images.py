import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import concurrent.futures
import cv2
import numpy as np

# ---------------------------------------------------------
# Core Processing Logic
# ---------------------------------------------------------


def process_image_batch(
    image_path: Path, crop_tasks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Loads an image ONCE and performs multiple crops.

    Args:
        image_path: Path to the source image
        crop_tasks: List of dicts containing 'coords', 'output_path', 'meta'

    Returns:
        List of successful manifest entries.
    """
    results = []

    # 1. Load Image (High I/O Cost - performed only once per batch)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image: {image_path}")
        return results

    h, w = img.shape[:2]

    for task in crop_tasks:
        try:
            # 2. Parse Coordinates
            coords = task["coords"]
            # Use numpy clip for faster, cleaner clamping
            x_min, y_min, x_max, y_max = np.round(coords).astype(int)

            # Validate basics
            if x_max <= x_min or y_max <= y_min:
                continue

            # Clamp
            x_min = np.clip(x_min, 0, w - 1)
            y_min = np.clip(y_min, 0, h - 1)
            x_max = np.clip(x_max, x_min + 1, w)
            y_max = np.clip(y_max, y_min + 1, h)

            # 3. Slicing (Zero-copy view usually, very fast)
            crop_img = img[y_min:y_max, x_min:x_max]

            if crop_img.size == 0:
                continue

            # 4. Save (High I/O Cost)
            out_path = task["output_path"]

            # Ensure directory exists (cached check is faster, but straightforward here)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            success = cv2.imwrite(str(out_path), crop_img)

            if success:
                # Append the pre-packaged manifest info
                results.append(task["manifest_data"])

        except Exception as e:
            print(f"Error processing crop for {image_path.name}: {e}")
            continue

    return results


# ---------------------------------------------------------
# Orchestration
# ---------------------------------------------------------


def crop_all_images_cli(dataset_name: str) -> None:
    # Mock imports for standalone functionality
    try:
        from .io_paths import load_config, get_images_dir, get_outputs_dir
    except ImportError:
        # Fallback for testing optimization without project structure
        print("Using dummy paths for standalone execution...")
        load_config = lambda x: {}
        get_images_dir = lambda x: Path("data/images")
        get_outputs_dir = lambda x: Path("data/outputs")

    config = load_config(dataset_name)
    images_dir = get_images_dir(config)
    outputs_dir = get_outputs_dir(config)

    crop_coords_path = outputs_dir / "image_crop_coordinates.json"

    if not crop_coords_path.exists():
        print(f"Error: {crop_coords_path} not found.")
        return

    print(f"Loading coordinates from {crop_coords_path}...")
    with crop_coords_path.open("r", encoding="utf-8") as f:
        all_crop_data = json.load(f)

    crops_dir = outputs_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # --- PRE-PROCESSING: Group by Image ---
    # We reshape the data structure from Component -> Crops
    # to Image -> [List of Crops across all components]

    image_tasks = defaultdict(list)
    total_tasks_count = 0

    print("Regrouping data by source image...")

    for comp_id, crop_list in all_crop_data.items():
        comp_dir = crops_dir / f"component_{comp_id}"

        # Pre-create component directories to avoid race conditions in threads
        comp_dir.mkdir(parents=True, exist_ok=True)

        for idx, crop_info in enumerate(crop_list):
            image_name = crop_info["image_name"]
            image_path = images_dir / image_name

            output_name = f"{image_path.stem}_crop{idx:03d}.jpg"
            output_path = comp_dir / output_name

            # Prepare the data needed for the worker
            task_payload = {
                "coords": crop_info["crop_coordinates"],
                "output_path": output_path,
                "manifest_data": {
                    # Data needed for the final manifest
                    "comp_id": comp_id,
                    "data": {
                        "crop_filename": output_name,
                        "source_image": image_name,
                        "crop_index": idx,
                        "crop_coordinates": crop_info["crop_coordinates"],
                        "image_id": crop_info.get("image_id"),
                        "fraction_visible": crop_info.get("fraction_visible"),
                        "visible_points": crop_info.get("visible_points"),
                        "total_points": crop_info.get("total_points"),
                    },
                },
            }

            image_tasks[image_path].append(task_payload)
            total_tasks_count += 1

    print(
        f"Optimization: Grouped {total_tasks_count} crops into {len(image_tasks)} source images."
    )
    print(f"Starting parallel processing with ProcessPoolExecutor...")

    # --- PARALLEL EXECUTION ---

    final_manifest = defaultdict(
        lambda: {"component_id": None, "total_crops": 0, "crops": []}
    )

    # ProcessPoolExecutor is preferred for CPU-heavy CV tasks (decoding/encoding)
    # Adjust max_workers based on RAM available.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a future for each image batch
        future_to_img = {
            executor.submit(process_image_batch, img_path, tasks): img_path
            for img_path, tasks in image_tasks.items()
        }

        completed_crops = 0

        for future in concurrent.futures.as_completed(future_to_img):
            img_path = future_to_img[future]
            try:
                batch_results = future.result()

                # Reconstruct the Manifest structure
                for res in batch_results:
                    c_id = res["comp_id"]
                    c_data = res["data"]

                    if final_manifest[c_id]["component_id"] is None:
                        final_manifest[c_id]["component_id"] = int(c_id)

                    final_manifest[c_id]["crops"].append(c_data)
                    final_manifest[c_id]["total_crops"] += 1
                    completed_crops += 1

            except Exception as exc:
                print(f"Image {img_path.name} generated an exception: {exc}")

            if completed_crops % 100 == 0:
                print(
                    f"Progress: {completed_crops}/{total_tasks_count} crops processed...",
                    end="\r",
                )

    # Save Manifest
    manifest_path = crops_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)

    print(f"\nDone! Processed {completed_crops} crops.")
    print(f"Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    crop_all_images_cli(args.dataset_name)


if __name__ == "__main__":
    main()
