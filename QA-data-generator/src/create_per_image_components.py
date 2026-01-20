import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def create_per_image_components(dataset_name: str, pixel_threshold: int = 5000) -> None:
    """
    Create per-image component JSON files containing component IDs and captions.

    Args:
        dataset_name: Name of the dataset (e.g., 'ProjectLabStudio_NoNeg')
        pixel_threshold: Minimum number of pixels to consider a crop valid (default: 5000)
    """
    # Define paths - go up to project root (scan-to-map)
    base_path = Path(__file__).parent.parent.parent / "outputs" / dataset_name
    manifest_path = base_path / "crops" / "manifest.json"
    captions_path = base_path / "component_captions.json"
    per_image_dir = base_path / "per_image_components"

    # Validate that required files exist
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    # Read manifest.json
    print(f"Loading manifest from {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    # Read component_captions.json
    print(f"Loading captions from {captions_path}")
    with open(captions_path, "r") as f:
        captions_data = json.load(f)

    # Create a lookup dictionary for captions by component_id
    component_captions_lookup = {
        item["component_id"]: item["caption"] for item in captions_data
    }

    print(f"Loaded manifest with {len(manifest_data)} components")
    print(f"Loaded captions for {len(component_captions_lookup)} components")
    print(f"Using pixel threshold: {pixel_threshold}")

    # Create inverse mapping: source_image -> list of {component_id, crop_filename, num_pixels} objects
    source_image_to_components: Dict[str, List[Dict[str, Any]]] = {}

    for component_id, component_data in manifest_data.items():
        if "crops" in component_data:
            for crop in component_data["crops"]:
                source_image = crop["source_image"]
                crop_filename = crop["crop_filename"]
                crop_coordinates = crop["crop_coordinates"]

                # Calculate number of pixels from crop_coordinates [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = crop_coordinates
                width = x_max - x_min
                height = y_max - y_min
                num_pixels = int(width * height)

                # Add object with component_id, crop_filename, and num_pixels
                if num_pixels > pixel_threshold:
                    if source_image not in source_image_to_components:
                        source_image_to_components[source_image] = []
                    source_image_to_components[source_image].append(
                        {
                            "component_id": component_id,
                            "crop_filename": crop_filename,
                            "num_pixels": num_pixels,
                        }
                    )

    print(f"Found {len(source_image_to_components)} unique source images")

    # Create directory for per-image component files
    per_image_dir.mkdir(parents=True, exist_ok=True)

    # Process each image and save components to individual JSON files
    files_saved = 0
    for image_name, components_list in source_image_to_components.items():
        # Track which component_ids we've already added to avoid duplicates
        added_component_ids = set()
        components_for_image = []

        for comp in components_list:
            component_id = comp["component_id"]

            # Only add each component_id once per image
            if component_id not in added_component_ids:
                # Get caption from lookup, default to empty string if not found
                caption = component_captions_lookup.get(int(component_id), "")

                components_for_image.append(
                    {"component_id": component_id, "component_caption": caption}
                )

                added_component_ids.add(component_id)

        # Save to JSON file (remove extension from image name)
        image_name_without_ext = Path(image_name).stem
        output_file = per_image_dir / f"{image_name_without_ext}_components.json"
        with open(output_file, "w") as f:
            json.dump(components_for_image, f, indent=2)

        files_saved += 1

    print(f"Saved {files_saved} component files to {per_image_dir}")


def main():
    """CLI entry point for creating per-image component files."""
    parser = argparse.ArgumentParser(
        description="Create per-image component JSON files with component IDs and captions."
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=5000,
        help="Minimum number of pixels to consider a crop valid (default: 5000)",
    )

    args = parser.parse_args()

    try:
        create_per_image_components(args.dataset_name, args.pixel_threshold)
        print("\nSuccessfully created per-image component files!")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
