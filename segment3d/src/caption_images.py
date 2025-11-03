"""
Caption images for each connected component using a Vision Language Model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_top_images(
    component_id: str | int, manifest_data: Dict[str, Any], n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get the top n images with the most visible points for a component.

    Args:
        component_id: ID of the component
        manifest_data: Dictionary loaded from manifest.json
        n: Number of top images to return (default: 5)

    Returns:
        List of dictionaries containing crop information, sorted by visible points
    """
    comp_id_str = str(component_id)

    if comp_id_str not in manifest_data:
        raise ValueError(f"Component {comp_id_str} not found in manifest")

    component_data = manifest_data[comp_id_str]
    crops = component_data.get("crops", [])

    # Sort by visible_points descending
    sorted_crops = sorted(crops, key=lambda x: x.get("visible_points", 0), reverse=True)

    # Return top n
    return sorted_crops[:n]


def prepare_component_messages(
    component_id: str | int,
    top_images: List[Dict[str, Any]],
    crops_dir: Path,
) -> tuple[List[str], Optional[List[Dict]]]:
    """
    Prepare messages for a component without calling the model.

    Args:
        component_id: ID of the component
        top_images: List of top crop information from get_top_images
        crops_dir: Directory containing the cropped images

    Returns:
        Tuple of (image_paths, messages) or ([], None) if no valid images
    """
    # Prepare image paths from crop filenames
    image_paths = []
    for crop_info in top_images:
        crop_filename = crop_info["crop_filename"]
        image_path = crops_dir / f"component_{component_id}" / crop_filename

        if not image_path.exists():
            print(f"  Warning: Crop image not found: {image_path}, skipping")
            continue

        image_paths.append(str(image_path))

    if not image_paths:
        return [], None

    # Create the prompt
    # prompt_text = (
    #     "These images show different views of the same object or region in a 3D scene. "
    #     "Analyze all the images together and provide a concise, descriptive caption "
    #     "that captures what this object or region is. Focus on:\n"
    #     "1. What the main object/region is\n"
    #     "2. Its key visual characteristics (color, shape, texture)\n"
    #     "3. Any notable features or context\n\n"
    #     "Keep the caption clear and factual, suitable for 3D semantic search."
    # )
    prompt_text = (
        "Caption images. Focus on:\n"
        "The main object,\n"
        "Its key visual characteristics (color, shape, texture)\n"
        "Any notable features or context\n\n"
        "Keep the caption clear and suitable for 3D semantic search."
    )

    # Build messages with images and text prompt
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": path} for path in image_paths]
                + [{"type": "text", "text": prompt_text}]
            ),
        },
    ]

    return image_paths, messages


def caption_components_batch_qwen(
    batch_data: List[tuple[str | int, List[Dict[str, Any]]]],
    crops_dir: Path,
    pipe: Any,
    model: str,
) -> List[tuple[str | int, str, List[str]]]:
    """
    Generate captions for multiple components in batch using HuggingFace Qwen model.

    Args:
        batch_data: List of (component_id, top_images) tuples
        crops_dir: Directory containing the cropped images
        pipe: HuggingFace pipeline for image-text-to-text
        model: Model name (default: "Qwen/Qwen2.5-VL-7B-Instruct")

    Returns:
        List of (component_id, caption, image_paths) tuples
    """
    results = []
    batch_messages = []
    batch_component_ids = []
    batch_image_paths = []

    # Prepare all messages
    for component_id, top_images in batch_data:
        image_paths, messages = prepare_component_messages(
            component_id, top_images, crops_dir
        )

        if messages is None:
            # No valid images for this component
            results.append(
                (
                    component_id,
                    f"[No valid crop images found for component {component_id}]",
                    [],
                )
            )
        else:
            batch_messages.append(messages)
            batch_component_ids.append(component_id)
            batch_image_paths.append(image_paths)

    # Process batch
    if batch_messages:
        try:
            # Call pipeline with batch of messages
            responses = pipe(
                text=batch_messages,
                max_new_tokens=300,
                return_full_text=False,
                batch_size=len(batch_messages),
                do_sample=False,  # Ensure deterministic output
            )

            # Extract captions from responses
            for component_id, response, image_paths in zip(
                batch_component_ids, responses, batch_image_paths
            ):
                caption = response[0]["generated_text"].strip()
                # Qwen/Qwen2.5-VL-7B-Instruct generated "addCriterion" for some reason.
                # Just remove that if it appears.
                if model == "Qwen/Qwen2.5-VL-7B-Instruct":
                    caption = caption.replace("addCriterion:", "").strip()
                    caption = caption.replace("addCriterion\n", "").strip()
                    caption = caption.replace("addCriterion", "").strip()
                    caption = caption.strip('"\\"')
                results.append((component_id, caption, image_paths))

        except Exception as e:
            print(f"  Error calling Qwen model for batch: {e}")
            # Add error results for all components in batch
            for component_id, image_paths in zip(
                batch_component_ids, batch_image_paths
            ):
                results.append(
                    (component_id, f"[Error generating caption: {str(e)}]", image_paths)
                )

    return results


def caption_all_components_cli(
    dataset_name: str,
    n_images: int = 5,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: int = 0,
    max_components: Optional[int] = None,
    batch_size: int = 4,
) -> None:
    """
    Caption all components by reading manifest.json and bbox_corners.json.

    Uses cropped images from the crops directory for captioning.

    Args:
        dataset_name: Name of the dataset to process
        n_images: Number of top images to use for each component
        model: Name of the VLM model to use
        device: GPU device ID to use for inference
        max_components: Maximum number of components to process (None for all)
        batch_size: Number of components to process in each batch (default: 4)
    """
    import os
    import torch
    from transformers import pipeline
    from .io_paths import load_config, get_outputs_dir

    import time as time_module

    # Load configuration
    config = load_config(dataset_name=dataset_name)

    outputs_dir = get_outputs_dir(config)
    crops_dir = outputs_dir / "crops"

    print(f"Outputs directory: {outputs_dir}")
    print(f"Crops directory: {crops_dir}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Images per component: {n_images}")
    print(f"Batch size: {batch_size}")

    # Initialize the pipeline
    print(f"\nInitializing {model}...")
    pipe = pipeline(
        task="image-text-to-text",
        model=model,
        device=device,
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully!")

    # Load manifest
    manifest_path = crops_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Please run the crop generation step first."
        )

    print(f"\nLoading manifest from: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Get list of component IDs to process
    component_ids = sorted(manifest_data.keys(), key=int)

    # Limit the number of components if specified
    if max_components is not None:
        component_ids = component_ids[:max_components]
        print(
            f"\nLimiting to {max_components} components (out of {len(manifest_data)} total)"
        )
    else:
        print(f"\nFound {len(component_ids)} components to caption")

    # Caption components in batches
    all_captions = []
    total_components = len(component_ids)

    # Start timing
    start_time = time_module.time()

    for batch_start in range(0, total_components, batch_size):
        batch_end = min(batch_start + batch_size, total_components)
        batch_component_ids = component_ids[batch_start:batch_end]

        print(
            f"\n{'='*60}\nProcessing batch [{batch_start + 1}-{batch_end}] of {total_components} components\n{'='*60}"
        )

        # Prepare batch data and store top_images_info for later use
        batch_data = []
        component_metadata = {}  # Store top_images_info for each component
        for component_id in batch_component_ids:
            try:
                top_images_info = get_top_images(
                    component_id, manifest_data, n=n_images
                )
                batch_data.append((component_id, top_images_info))
                component_metadata[component_id] = top_images_info
            except Exception as e:
                print(f"Error preparing component {component_id}: {e}")
                continue

        if not batch_data:
            print("No valid components in this batch, skipping")
            continue

        # Process batch
        try:
            batch_results = caption_components_batch_qwen(
                batch_data, crops_dir, pipe, model
            )

            # Process results
            for component_id, caption, image_paths in batch_results:
                comp_id_int = int(component_id)

                # Get the top images info from metadata (already retrieved)
                top_images_info = component_metadata.get(component_id, [])
                crop_filenames = [img["crop_filename"] for img in top_images_info]

                print(
                    f"\nComponent {component_id}:\n  Images: {', '.join(crop_filenames[:3])}{'...' if len(crop_filenames) > 3 else ''}"
                )
                print(
                    f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}"
                )

                component_result = {
                    "component_id": comp_id_int,
                    "caption": caption,
                    "num_images_used": len(top_images_info),
                    "crop_filenames": crop_filenames,
                }

                all_captions.append(component_result)

        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate timing statistics
    end_time = time_module.time()
    total_runtime = end_time - start_time
    total_captions_generated = len(all_captions)

    # Calculate rates
    captions_per_second = (
        total_captions_generated / total_runtime if total_runtime > 0 else 0
    )
    time_per_caption = (
        total_runtime / total_captions_generated if total_captions_generated > 0 else 0
    )

    # Save captions
    output_path = outputs_dir / "component_captions.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2)

    # Save caption statistics
    caption_stats = {
        "total_captions_generated": total_captions_generated,
        "total_runtime_seconds": total_runtime,
        "captions_per_second": captions_per_second,
        "time_per_caption_seconds": time_per_caption,
        "batch_size": batch_size,
        "n_images_per_component": n_images,
        "model": model,
        "device": device,
    }

    stats_output_path = outputs_dir / "caption_stats.json"
    with stats_output_path.open("w", encoding="utf-8") as f:
        json.dump(caption_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Captioning complete!")
    print(f"Successfully captioned {len(all_captions)}/{len(manifest_data)} components")
    print(f"Captions saved to: {output_path}")
    print(f"Statistics saved to: {stats_output_path}")

    # Print timing statistics
    print(f"\nTiming statistics:")
    print(
        f"  Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
    )
    print(f"  Total captions generated: {total_captions_generated}")
    print(f"  Captions per second: {captions_per_second:.2f}")
    print(f"  Time per caption: {time_per_caption:.2f} seconds")

    # Print caption length statistics
    caption_lengths = [len(c["caption"]) for c in all_captions]
    if caption_lengths:
        print(f"\nCaption length statistics:")
        print(
            f"  Average length: {sum(caption_lengths) / len(caption_lengths):.1f} characters"
        )
        print(f"  Min length: {min(caption_lengths)} characters")
        print(f"  Max length: {max(caption_lengths)} characters")


def main() -> None:
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Caption components using a Vision Language Model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=2,
        help="Number of top images to use per component (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model to use (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Maximum number of components to process (default: process all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of components to process in each batch (default: 4)",
    )

    args = parser.parse_args()

    if args.n_images < 1:
        parser.error("--n-images must be at least 1")

    if args.max_components is not None and args.max_components < 1:
        parser.error("--max-components must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    caption_all_components_cli(
        dataset_name=args.dataset,
        n_images=args.n_images,
        model=args.model,
        device=args.device,
        max_components=args.max_components,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
