from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment3d.src import io_paths as segment3d_io_paths
from segment3d.src.crop_images import crop_all_images_cli
from segment3d.src.project_bbox import project_all_bboxes_cli
import segment3d.src.project_bbox as project_bbox_module


DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def _resolve_data_dir(scene_dir: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidate = Path(scene_dir).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (data_root / scene_dir).resolve()


def _build_scannetpp_config(
    dataset_name: str,
    images_dir: Path,
    colmap_model_dir: Path,
    outputs_dir: Path,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "images_dir": str(images_dir),
        "colmap_model_dir": str(colmap_model_dir),
        "outputs_dir": str(outputs_dir),
        "_config_dir": str(REPO_ROOT / "segment3d"),
    }


@contextmanager
def _patched_segment3d_paths(config: dict[str, Any]) -> Iterator[None]:
    outputs_dir = Path(config["outputs_dir"])
    images_dir = Path(config["images_dir"])
    colmap_model_dir = Path(config["colmap_model_dir"])

    original_io_paths = {
        "load_config": segment3d_io_paths.load_config,
        "get_images_dir": segment3d_io_paths.get_images_dir,
        "get_outputs_dir": segment3d_io_paths.get_outputs_dir,
        "get_colmap_model_dir": segment3d_io_paths.get_colmap_model_dir,
    }
    original_project_bbox = {
        "load_config": project_bbox_module.load_config,
        "get_outputs_dir": project_bbox_module.get_outputs_dir,
        "get_colmap_model_dir": project_bbox_module.get_colmap_model_dir,
    }

    def _load_config(_: str) -> dict[str, Any]:
        return config

    def _get_images_dir(_: dict[str, Any]) -> Path:
        return images_dir

    def _get_outputs_dir(_: dict[str, Any]) -> Path:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return outputs_dir

    def _get_colmap_model_dir(_: dict[str, Any]) -> Path:
        return colmap_model_dir

    segment3d_io_paths.load_config = _load_config
    segment3d_io_paths.get_images_dir = _get_images_dir
    segment3d_io_paths.get_outputs_dir = _get_outputs_dir
    segment3d_io_paths.get_colmap_model_dir = _get_colmap_model_dir

    project_bbox_module.load_config = _load_config
    project_bbox_module.get_outputs_dir = _get_outputs_dir
    project_bbox_module.get_colmap_model_dir = _get_colmap_model_dir

    try:
        yield
    finally:
        for name, value in original_io_paths.items():
            setattr(segment3d_io_paths, name, value)
        for name, value in original_project_bbox.items():
            setattr(project_bbox_module, name, value)


def create_crops(
    data_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_name: str | None = None,
    min_fraction: float = 0.3,
) -> Path:
    data_dir = Path(data_dir).resolve()
    dataset_name = output_name or data_dir.name
    images_dir = data_dir / "dslr" / "resized_undistorted_images"
    colmap_model_dir = data_dir / "dslr" / "colmap"
    outputs_dir = Path(output_root).resolve() / dataset_name

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {data_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not colmap_model_dir.is_dir():
        raise FileNotFoundError(f"COLMAP model directory not found: {colmap_model_dir}")

    connected_components_path = outputs_dir / "connected_components.json"
    bbox_corners_path = outputs_dir / "bbox_corners.json"
    if not connected_components_path.is_file():
        raise FileNotFoundError(
            f"Missing connected components file: {connected_components_path}"
        )
    if not bbox_corners_path.is_file():
        raise FileNotFoundError(f"Missing bbox corners file: {bbox_corners_path}")

    config = _build_scannetpp_config(
        dataset_name=dataset_name,
        images_dir=images_dir,
        colmap_model_dir=colmap_model_dir,
        outputs_dir=outputs_dir,
    )

    with _patched_segment3d_paths(config):
        print(f"Using images directory: {images_dir}")
        print(f"Using COLMAP model directory: {colmap_model_dir}")
        print(f"Using outputs directory: {outputs_dir}")
        print("Projecting 3D bounding boxes into image crops...")
        project_all_bboxes_cli(dataset_name=dataset_name, min_fraction=min_fraction)
        print("Cropping projected image regions...")
        crop_all_images_cli(dataset_name=dataset_name)

    return outputs_dir / "crops"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create image crops for a ScanNet++ scene by running the segment3d "
            "project_bbox and crop_images stages."
        )
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

    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.scene_dir, data_root=args.data_root)
    crops_dir = create_crops(
        data_dir=data_dir,
        output_root=args.output_root,
        output_name=args.output_name,
        min_fraction=args.min_fraction,
    )
    print(f"Saved crops to {crops_dir}")


if __name__ == "__main__":
    main()
