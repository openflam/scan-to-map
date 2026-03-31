from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


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

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Open3D could not read any mesh data from {mesh_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_triangle_mesh(str(output_path), mesh)
    if not success:
        raise RuntimeError(f"Open3D failed to write GLB mesh to {output_path}")

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
