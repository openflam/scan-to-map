"""
CLIP embedding generation for components.

This module generates CLIP embeddings for each component using the top image
from the manifest. Uses OpenCLIP for embedding generation.
"""

from __future__ import annotations

import json
import time as time_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import faiss


class ComponentImageDataset(Dataset):
    """PyTorch dataset for loading component top images."""

    def __init__(
        self,
        component_ids: List[str],
        image_paths: List[Path],
        preprocess,
    ):
        """
        Initialize the dataset.

        Args:
            component_ids: List of component IDs
            image_paths: List of paths to the top images
            preprocess: OpenCLIP preprocessing function
        """
        self.component_ids = component_ids
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.component_ids)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (component_id, preprocessed_image_tensor)
        """
        component_id = self.component_ids[idx]
        image_path = self.image_paths[idx]

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image tensor if loading fails
            image_tensor = torch.zeros((3, 224, 224))

        return component_id, image_tensor


def get_top_image(
    component_id: str | int, manifest_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Get the top image with the most visible points for a component.

    Args:
        component_id: ID of the component
        manifest_data: Dictionary loaded from manifest.json

    Returns:
        Dictionary containing crop information, or None if not found
    """
    comp_id_str = str(component_id)

    if comp_id_str not in manifest_data:
        print(f"Warning: Component {comp_id_str} not found in manifest")
        return None

    component_data = manifest_data[comp_id_str]
    crops = component_data.get("crops", [])

    if not crops:
        print(f"Warning: No crops found for component {comp_id_str}")
        return None

    # Sort by visible_points descending and return the top one
    sorted_crops = sorted(crops, key=lambda x: x.get("visible_points", 0), reverse=True)
    return sorted_crops[0]


def generate_clip_embeddings_cli(
    dataset_name: str,
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2B-s32B-b79K",
    device: Optional[int] = None,
    batch_size: int = 32,
    max_components: Optional[int] = None,
) -> None:
    """
    Generate CLIP embeddings for all components using their top images.

    Args:
        dataset_name: Name of the dataset to process
        model_name: OpenCLIP model name (default: ViT-B-32)
        pretrained: Pretrained weights to use (default: laion2b_s34b_b79k)
        device: GPU device ID to use for inference (None for auto-detect)
        batch_size: Number of images to process in each batch
        max_components: Maximum number of components to process (None for all)
    """
    from .io_paths import load_config, get_outputs_dir

    # Load configuration
    config = load_config(dataset_name=dataset_name)
    outputs_dir = get_outputs_dir(config)
    crops_dir = outputs_dir / "crops"

    print(f"\n{'='*80}")
    print("CLIP EMBEDDING GENERATION")
    print(f"{'='*80}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Crops directory: {crops_dir}")
    print(f"Model: {model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Batch size: {batch_size}")

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
        print(f"\nFound {len(component_ids)} components to process")

    # Prepare data: get top image for each component
    print("\nPreparing component images...")
    valid_component_ids = []
    valid_image_paths = []

    for component_id in tqdm(component_ids, desc="Loading top images"):
        top_image_info = get_top_image(component_id, manifest_data)
        if top_image_info is None:
            continue

        crop_filename = top_image_info.get("crop_filename")
        if not crop_filename:
            print(f"Warning: No crop_filename for component {component_id}")
            continue

        # Construct path to the crop image
        component_dir = crops_dir / f"component_{component_id}"
        image_path = component_dir / crop_filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        valid_component_ids.append(component_id)
        valid_image_paths.append(image_path)

    print(f"Found {len(valid_component_ids)} valid components with images")

    if not valid_component_ids:
        print("No valid components to process. Exiting.")
        return

    # Set up device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = f"cuda:{device}"

    print(f"\nUsing device: {device_str}")

    # Load CLIP model
    print(f"\nLoading CLIP model: {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device_str)
    model.eval()

    # Create dataset and dataloader
    dataset = ComponentImageDataset(
        component_ids=valid_component_ids,
        image_paths=valid_image_paths,
        preprocess=preprocess,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device_str.startswith("cuda") else False,
    )

    # Generate embeddings
    print(f"\nGenerating CLIP embeddings...")
    start_time = time_module.time()

    all_embeddings = {}
    all_component_ids_list = []

    with torch.no_grad():
        for batch_component_ids, batch_images in tqdm(
            dataloader, desc="Processing batches"
        ):
            # Move images to device
            batch_images = batch_images.to(device_str)

            # Generate embeddings
            image_features = model.encode_image(batch_images)

            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Move to CPU and convert to numpy
            image_features_np = image_features.cpu().numpy()

            # Store embeddings
            for i, component_id in enumerate(batch_component_ids):
                embedding = image_features_np[i]
                all_embeddings[component_id] = embedding.tolist()
                all_component_ids_list.append(component_id)

    end_time = time_module.time()
    total_runtime = end_time - start_time

    # Calculate statistics
    embeddings_per_second = (
        len(all_embeddings) / total_runtime if total_runtime > 0 else 0
    )
    time_per_embedding = (
        total_runtime / len(all_embeddings) if len(all_embeddings) > 0 else 0
    )

    print(f"\n{'='*80}")
    print(f"Embedding generation complete!")
    print(f"{'='*80}")
    print(f"Successfully generated embeddings for {len(all_embeddings)} components")

    # Save embeddings as JSON
    embeddings_output_path = outputs_dir / "clip_embeddings.json"
    with embeddings_output_path.open("w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, indent=2)

    print(f"\nEmbeddings saved to: {embeddings_output_path}")

    # Also save as numpy array for efficient loading
    embeddings_np_path = outputs_dir / "clip_embeddings.npz"
    embeddings_array = np.array([all_embeddings[cid] for cid in all_component_ids_list])
    np.savez(
        embeddings_np_path,
        embeddings=embeddings_array,
        component_ids=np.array(all_component_ids_list),
    )

    print(f"Embeddings (numpy) saved to: {embeddings_np_path}")

    # Save FAISS index
    print(f"\nBuilding FAISS HNSW index...")
    embedding_dimension = embeddings_array.shape[1]

    # Create HNSW index
    # M = 32 is a good default for HNSW (number of connections per layer)
    # ef_construction = 200 controls index build time vs accuracy tradeoff
    index = faiss.IndexHNSWFlat(embedding_dimension, 32)
    index.hnsw.efConstruction = 200

    # Add embeddings to index
    index.add(embeddings_array.astype(np.float32))

    # Save FAISS index
    faiss_index_path = outputs_dir / "clip_embeddings.faiss"
    faiss.write_index(index, str(faiss_index_path))

    print(f"FAISS index saved to: {faiss_index_path}")
    print(f"  Index type: HNSW")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {embedding_dimension}")

    # Save statistics
    embedding_stats = {
        "total_embeddings_generated": len(all_embeddings),
        "embedding_dimension": (
            len(list(all_embeddings.values())[0]) if all_embeddings else 0
        ),
        "model_name": model_name,
        "pretrained": pretrained,
        "batch_size": batch_size,
        "device": device_str,
        "total_runtime_seconds": total_runtime,
        "embeddings_per_second": embeddings_per_second,
        "time_per_embedding_seconds": time_per_embedding,
        "faiss_index_type": "HNSW",
        "faiss_index_path": str(faiss_index_path),
    }

    stats_output_path = outputs_dir / "clip_embedding_stats.json"
    with stats_output_path.open("w", encoding="utf-8") as f:
        json.dump(embedding_stats, f, indent=2)

    print(f"Statistics saved to: {stats_output_path}")

    # Print timing statistics
    print(f"\nTiming statistics:")
    print(
        f"  Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
    )
    print(f"  Embeddings per second: {embeddings_per_second:.2f}")
    print(f"  Time per embedding: {time_per_embedding:.4f} seconds")

    # Print embedding info
    if all_embeddings:
        print(f"\nEmbedding information:")
        print(f"  Dimension: {embedding_stats['embedding_dimension']}")
        print(f"  Total components: {len(all_embeddings)}")


def main() -> None:
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for components"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-H-14",
        help="OpenCLIP model name (default: ViT-H-14)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2B-s32B-b79K",
        help="Pretrained weights to use (default: laion2B-s32B-b79K)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU device ID to use (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to process in each batch (default: 32)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Maximum number of components to process (default: process all)",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.max_components is not None and args.max_components < 1:
        parser.error("--max-components must be at least 1")

    generate_clip_embeddings_cli(
        dataset_name=args.dataset,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        max_components=args.max_components,
    )


if __name__ == "__main__":
    main()
