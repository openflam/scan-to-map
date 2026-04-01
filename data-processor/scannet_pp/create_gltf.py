from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def _resolve_data_dir(scene_dir: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidate = Path(scene_dir).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (data_root / scene_dir).resolve()


def create_gltf(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir).resolve()
    mesh_path = data_dir / "scans" / "mesh_aligned_0.05.ply"
    output_dir = data_dir / "polycam_data"
    output_path = output_dir / "raw.glb"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {data_dir}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = trimesh.load(str(mesh_path), force='mesh')
    if mesh.is_empty:
        raise ValueError(f"Trimesh could not read any mesh data from {mesh_path}")

    # --- FIX: Rotate -90 degrees around X-axis to convert Z-up to Y-up ---
    # This aligns the ScanNet++ data with the GLTF standard.
    R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh.apply_transform(R)
    # --------------------------------------------------------------------

    # --- FIX: Rotate -90 degrees around Y-axis for some tansformations that happen
    # in the viewer ---
    R = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])
    mesh.apply_transform(R)
    # --------------------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    print(f"Saved GLB mesh to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ScanNet++ PLY mesh into polycam_data/raw.glb. "
            "Run this script with the segment3d-env interpreter."
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

    args = parser.parse_args()
    data_dir = _resolve_data_dir(args.scene_dir, data_root=args.data_root)
    create_gltf(data_dir)


if __name__ == "__main__":
    main()
