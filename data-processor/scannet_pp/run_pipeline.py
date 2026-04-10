from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from create_bboxes import get_bounding_boxes
from create_crops import create_crops
from create_captions import create_captions
from create_gltf import create_gltf


DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def _resolve_data_dir(scene_dir: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidate = Path(scene_dir).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (data_root / scene_dir).resolve()


def run_pipeline(
    data_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_name: str | None = None,
    min_fraction: float = 0.1,
    max_crops_per_component: int = 5,
) -> None:
    data_dir = Path(data_dir).resolve()
    
    if output_name is None:
        if not data_dir.name.startswith("scannetpp_"):
            output_name = f"scannetpp_{data_dir.name}"
        else:
            output_name = data_dir.name

    print(f"\n--- Running create_gltf on {data_dir} ---")
    create_gltf(
        data_dir=data_dir,
        output_root=output_root,
        output_name=output_name,
    )

    print(f"\n--- Running create_bboxes on {data_dir} ---")
    get_bounding_boxes(
        data_dir=data_dir,
        output_root=output_root,
        output_name=output_name,
    )

    print(f"\n--- Running create_crops on {data_dir} ---")
    create_crops(
        data_dir=data_dir,
        output_root=output_root,
        output_name=output_name,
        min_fraction=min_fraction,
        max_crops_per_component=max_crops_per_component,
    )

    print(f"\n--- Running create_captions on {data_dir} ---")
    create_captions(
        data_dir=data_dir,
        output_root=output_root,
        output_name=output_name,
    )

    print("\n--- Pipeline finished successfully! ---")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full data processing pipeline for a ScanNet++ scene"
    )
    parser.add_argument(
        "scene_dir",
        help=(
            "ScanNet++ scene directory name under data/ "
            "(for example: scannetpp_7b6477cb95) or an absolute path"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base data directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base outputs directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Output directory name under output-root. Defaults to the scene "
            "directory name."
        ),
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.1,
        help="Minimum visible 3D-point fraction for projecting a bbox to an image.",
    )
    parser.add_argument(
        "--max-crops-per-component",
        type=int,
        default=5,
        help="Maximum number of crops to generate per component.",
    )

    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.scene_dir, data_root=args.data_root)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {data_dir}")

    run_pipeline(
        data_dir=data_dir,
        output_root=args.output_root,
        output_name=args.output_name,
        min_fraction=args.min_fraction,
        max_crops_per_component=args.max_crops_per_component,
    )


if __name__ == "__main__":
    main()
