from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def _resolve_data_dir(scene_dir: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidate = Path(scene_dir).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (data_root / scene_dir).resolve()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_crop_filename(component_manifest: dict[str, Any]) -> str:
    crops = component_manifest.get("crops", [])
    if not crops:
        raise ValueError("Component has no crops in manifest.json")

    best_crop = max(
        crops,
        key=lambda crop: (
            float(crop.get("fraction_visible") or 0.0),
            -int(crop.get("crop_index") or 0),
            str(crop.get("crop_filename") or ""),
        ),
    )
    crop_filename = best_crop.get("crop_filename")
    if not crop_filename:
        raise ValueError("Best crop entry is missing crop_filename")
    return str(crop_filename)


def create_captions(
    data_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_name: str | None = None,
) -> Path:
    data_dir = Path(data_dir).resolve()
    dataset_name = output_name or data_dir.name
    outputs_dir = Path(output_root).resolve() / dataset_name

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {data_dir}")

    bbox_path = outputs_dir / "bbox_corners.json"
    manifest_path = outputs_dir / "crops" / "manifest.json"
    output_path = outputs_dir / "component_captions.json"

    if not bbox_path.is_file():
        raise FileNotFoundError(f"Missing bbox corners file: {bbox_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing crops manifest file: {manifest_path}")

    bbox_entries = _load_json(bbox_path)
    manifest = _load_json(manifest_path)
    bbox_lookup = {
        int(entry["connected_comp_id"]): entry for entry in bbox_entries
    }

    captions: list[dict[str, Any]] = []
    for component_key in sorted(manifest.keys(), key=int):
        component_id = int(component_key)
        manifest_entry = manifest[component_key]
        bbox_entry = bbox_lookup.get(component_id)
        if bbox_entry is None:
            continue

        captions.append(
            {
                "component_id": component_id,
                "caption": str(bbox_entry.get("label") or ""),
                "num_images_used": 1,
                "crop_filenames": [_best_crop_filename(manifest_entry)],
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)

    print(f"Saved {len(captions)} captions to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create component_captions.json for a ScanNet++ scene using bbox labels "
            "and the highest-fraction crop from manifest.json."
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

    args = parser.parse_args()
    data_dir = _resolve_data_dir(args.scene_dir, data_root=args.data_root)
    create_captions(
        data_dir=data_dir,
        output_root=args.output_root,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
