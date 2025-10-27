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


def caption_component_qwen(
    component_id: str | int,
    top_images: List[Dict[str, Any]],
    crops_dir: Path,
    pipe,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> str:
    """
    Generate a caption for a component using HuggingFace Qwen model.

    Args:
        component_id: ID of the component
        top_images: List of top crop information from get_top_images
        crops_dir: Directory containing the cropped images
        pipe: HuggingFace pipeline for image-text-to-text
        model: Model name (default: "Qwen/Qwen2.5-VL-7B-Instruct")

    Returns:
        Generated caption string
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
        return f"[No valid crop images found for component {component_id}]"

    # Create the prompt
    prompt_text = (
        "These images show different views of the same object or region in a 3D scene. "
        "Analyze all the images together and provide a concise, descriptive caption "
        "that captures what this object or region is. Focus on:\n"
        "1. What the main object/region is\n"
        "2. Its key visual characteristics (color, shape, texture)\n"
        "3. Any notable features or context\n\n"
        "Keep the caption clear and factual, suitable for 3D semantic search."
    )

    # Build messages with images and text prompt
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": path} for path in image_paths]
                + [{"type": "text", "text": prompt_text}]
            ),
        }
    ]

    try:
        response = pipe(text=messages, max_new_tokens=300, return_full_text=False)
        caption = response[0]["generated_text"].strip()
        return caption

    except Exception as e:
        print(f"  Error calling Qwen model: {e}")
        return f"[Error generating caption: {str(e)}]"


def caption_component(
    component_id: str | int,
    manifest_data: Dict[str, Any],
    crops_dir: Path,
    pipe,
    n_images: int = 10,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> str:
    """
    Caption a single component by getting top images and calling VLM.

    Args:
        component_id: ID of the component
        manifest_data: Dictionary loaded from manifest.json
        crops_dir: Directory containing the cropped images
        pipe: HuggingFace pipeline for image-text-to-text
        n_images: Number of top images to use (default: 10)
        model: Model name (default: "Qwen/Qwen2.5-VL-7B-Instruct")

    Returns:
        Generated caption string
    """
    # Get top images
    top_images = get_top_images(component_id, manifest_data, n=n_images)

    print(f"  Using {len(top_images)} images for component {component_id}")

    # Generate caption
    caption = caption_component_qwen(component_id, top_images, crops_dir, pipe, model)

    return caption


def caption_all_components_cli(
    dataset_name: str,
    n_images: int = 5,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: int = 0,
    max_components: Optional[int] = None,
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
    """
    import os
    import torch
    from transformers import pipeline
    from .io_paths import load_config, get_outputs_dir

    # Load configuration
    config = load_config(dataset_name=dataset_name)

    outputs_dir = get_outputs_dir(config)
    crops_dir = outputs_dir / "crops"

    print(f"Outputs directory: {outputs_dir}")
    print(f"Crops directory: {crops_dir}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Images per component: {n_images}")

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

    # Load bounding box corners
    bbox_corners_path = outputs_dir / "bbox_corners.json"
    if not bbox_corners_path.exists():
        raise FileNotFoundError(
            f"Bounding box corners not found: {bbox_corners_path}\n"
            "Please run src.bbox_corners first."
        )

    print(f"Loading bounding box corners from: {bbox_corners_path}")
    with bbox_corners_path.open("r", encoding="utf-8") as f:
        bbox_corners_data = json.load(f)

    # Create lookup for bbox corners by component ID
    bbox_lookup = {
        bbox["connected_comp_id"]: bbox["bbox"] for bbox in bbox_corners_data
    }

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

    # Caption each component
    all_captions = []

    for idx, component_id in enumerate(component_ids, 1):
        print(f"\n[{idx}/{len(component_ids)}] Processing component {component_id}...")

        try:
            # Get the top images info to log image names
            top_images_info = get_top_images(component_id, manifest_data, n=n_images)
            crop_filenames = [img["crop_filename"] for img in top_images_info]
            print(f"  Crop images used: {', '.join(crop_filenames)}")

            caption = caption_component(
                component_id,
                manifest_data,
                crops_dir,
                pipe,
                n_images=n_images,
                model=model,
            )

            print(f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")

            # Get bbox for this component
            comp_id_int = int(component_id)
            bbox = bbox_lookup.get(comp_id_int, None)

            component_result = {
                "component_id": comp_id_int,
                "caption": caption,
                "bbox": bbox,
                "num_images_used": len(top_images_info),
                "crop_filenames": crop_filenames,
            }

            all_captions.append(component_result)

        except Exception as e:
            print(f"  Error processing component {component_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save captions
    output_path = outputs_dir / "component_captions.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Captioning complete!")
    print(f"Successfully captioned {len(all_captions)}/{len(manifest_data)} components")
    print(f"Captions saved to: {output_path}")

    # Print some statistics
    caption_lengths = [len(c["caption"]) for c in all_captions]
    if caption_lengths:
        print(f"\nCaption statistics:")
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

    args = parser.parse_args()

    if args.n_images < 1:
        parser.error("--n-images must be at least 1")

    if args.max_components is not None and args.max_components < 1:
        parser.error("--max-components must be at least 1")

    caption_all_components_cli(
        dataset_name=args.dataset,
        n_images=args.n_images,
        model=args.model,
        device=args.device,
        max_components=args.max_components,
    )


if __name__ == "__main__":
    main()
