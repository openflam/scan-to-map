"""
Main entry point for QA data generation pipeline.
"""

import argparse
from src.create_per_image_components import create_per_image_components


def main():
    """Run the QA data generation pipeline."""
    parser = argparse.ArgumentParser(
        description="QA Data Generator - Create per-image component files"
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=5000,
        help="Minimum number of pixels to consider a crop valid (default: 5000)",
    )

    args = parser.parse_args()

    # Create per-image component files
    print(f"Processing dataset: {args.dataset_name}")
    print(f"Pixel threshold: {args.pixel_threshold}")
    print()

    try:
        create_per_image_components(args.dataset_name, args.pixel_threshold)

        print()
        print("=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
    except Exception as e:
        print()
        print("=" * 80)
        print(f"Error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
