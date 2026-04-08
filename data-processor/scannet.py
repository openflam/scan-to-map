"""
ScanNet → scan-to-map data preparation.

Converts a raw ScanNet scene directory into the layout expected by the
segment3d pipeline's config.py:

    data/<dataset>/ns_data/images/   ← RGB frames (copied here)
    data/<dataset>/hloc_data/sfm_reconstruction/
        cameras.txt
        images.txt   ← poses + POINTS2D (projected mesh vertices)
        points3D.txt ← empty header (not required by associate2d3d)

The key fix over the previous version: images.txt now contains real POINTS2D
entries (mesh vertex projections). The associate2d3d step reads these pixel
coordinates + their 3D point IDs to link SAM masks to 3D space.  Without them
the whole segment3d pipeline produces empty associations.

Usage (run from any directory):
    python data-processor/scannet.py <scannet_scene_dir>
                                     [--output-root /path/to/scan-to-map/data]
                                     [--num-image-limit 500]
                                     [--max-mesh-points 50000]

Requires: numpy, scipy, Pillow, open3d  (open3d is already used by polycam.py)
"""

import argparse
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_scene_paths(scannet_scene_dir: str, output_root: str = "../data") -> dict:
    """
    Centralise all input + output paths for one ScanNet scene.

    Output layout is chosen to match what segment3d/config.py expects:
      data/<dataset>/ns_data/images/
      data/<dataset>/hloc_data/sfm_reconstruction/   ← COLMAP text model
    """
    scene_path = Path(scannet_scene_dir).resolve()
    scene_id = scene_path.name                        # e.g. "scene0000_00"
    dataset_name = f"scannet_{scene_id}"              # e.g. "scannet_scene0000_00"

    out = Path(output_root).resolve()
    dataset_dir = out / dataset_name

    # COLMAP output goes directly into the sfm_reconstruction folder so that
    # config.py's `_default_colmap` path resolves correctly.
    colmap_dir = dataset_dir / "hloc_data" / "sfm_reconstruction"
    images_dir = dataset_dir / "ns_data" / "images"

    colmap_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    return {
        "scene_id":              scene_id,
        "dataset_name":          dataset_name,
        "scannet_color_dir":     str(scene_path / "sens" / "color"),
        "scannet_intrinsic_path":str(scene_path / "sens" / "intrinsic" / "intrinsic_color.txt"),
        "scannet_pose_dir":      str(scene_path / "sens" / "pose"),
        "mesh_ply_path":         str(scene_path / f"{scene_id}_vh_clean_2.ply"),
        "colmap_dir":            str(colmap_dir),
        "images_dir":            str(images_dir),
    }


# ---------------------------------------------------------------------------
# Frame selection helpers
# ---------------------------------------------------------------------------

def _sorted_color_files(color_dir: Path) -> list:
    files = sorted(color_dir.glob("*.jpg")) + sorted(color_dir.glob("*.png"))
    return sorted(files, key=lambda x: int(x.stem))


def _sorted_pose_files(pose_dir: Path) -> list:
    return sorted(pose_dir.glob("*.txt"), key=lambda x: int(x.stem))


def _uniform_subsample(pose_files, color_files, limit: int):
    """Uniformly select `limit` frames from the full sequence."""
    n = len(pose_files)
    indices = np.linspace(0, n - 1, limit, dtype=int)
    return [pose_files[i] for i in indices], [color_files[i] for i in indices]


def _load_pose(pose_file) -> Optional[np.ndarray]:
    """
    Load 4×4 camera-to-world matrix.  Returns None for invalid (inf/nan) frames
    which ScanNet occasionally includes.
    """
    c2w = np.loadtxt(pose_file)
    if not np.isfinite(c2w).all():
        return None
    return c2w


# ---------------------------------------------------------------------------
# Step 1: Copy / symlink images
# ---------------------------------------------------------------------------

def copy_images(paths: dict, color_files: list) -> None:
    """Copy RGB frames into ns_data/images/ (skip existing files)."""
    dst_dir = Path(paths["images_dir"])
    copied = 0
    for src in color_files:
        dst = dst_dir / Path(src).name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} new images to {dst_dir}  ({len(color_files)} total)")


# ---------------------------------------------------------------------------
# Step 2: Read intrinsics & image size
# ---------------------------------------------------------------------------

def read_intrinsics(paths: dict) -> tuple:
    """Return (fx, fy, cx, cy, W, H)."""
    mat = np.loadtxt(paths["scannet_intrinsic_path"])
    fx, fy = mat[0, 0], mat[1, 1]
    cx, cy = mat[0, 2], mat[1, 2]

    color_dir = Path(paths["scannet_color_dir"])
    sample = (_sorted_color_files(color_dir) or [None])[0]
    if sample is None:
        raise FileNotFoundError(f"No images found in {color_dir}")
    with Image.open(sample) as img:
        W, H = img.size
    return fx, fy, cx, cy, W, H


# ---------------------------------------------------------------------------
# Step 3: Write cameras.txt
# ---------------------------------------------------------------------------

def write_cameras_file(paths: dict, W: int, H: int,
                       fx: float, fy: float, cx: float, cy: float) -> None:
    out = Path(paths["colmap_dir"]) / "cameras.txt"
    with open(out, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    print(f"Wrote cameras.txt  → {out}")


# ---------------------------------------------------------------------------
# Step 4: Load mesh and generate 3D points
# ---------------------------------------------------------------------------

def load_mesh_points(mesh_ply_path: str, max_points: int = 50_000) -> np.ndarray:
    """
    Load ScanNet mesh vertices and subsample.

    Returns ndarray of shape (N, 6): [X, Y, Z, R, G, B].
    Requires open3d (already a dependency of polycam.py).
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for 3D point generation. "
            "Install it with:  pip install open3d"
        )

    ply = Path(mesh_ply_path)
    if not ply.exists():
        raise FileNotFoundError(
            f"ScanNet mesh not found: {ply}\n"
            "Expected file: <scene_dir>/<scene_id>_vh_clean_2.ply"
        )

    mesh = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(mesh.vertices)                     # (N, 3)

    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(float)
    else:
        colors = np.zeros((len(verts), 3), dtype=float)

    points = np.concatenate([verts, colors], axis=1)      # (N, 6)

    if len(points) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        print(f"Subsampled mesh: {len(verts)} → {max_points} points")
    else:
        print(f"Loaded {len(points)} mesh vertices")

    return points


# ---------------------------------------------------------------------------
# Step 5: Project mesh points and write images.txt + points3D.txt
# ---------------------------------------------------------------------------

def _project(pts_xyz: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray,
             fx: float, fy: float, cx: float, cy: float,
             W: int, H: int) -> tuple:
    """
    Vectorised projection of world-space points into a camera.

    Returns:
        xys          – (K, 2) float pixel coordinates of visible points
        valid_idx    – (K,) indices into pts_xyz of those points
    """
    P = (R_w2c @ pts_xyz.T).T + t_w2c      # (N, 3) camera-space

    depth = P[:, 2]
    front = depth > 0.05                    # must be in front of camera

    d = np.where(front, depth, 1.0)         # avoid div-by-zero
    x = fx * P[:, 0] / d + cx
    y = fy * P[:, 1] / d + cy

    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    visible = front & in_bounds

    valid_idx = np.where(visible)[0]
    xys = np.stack([x[valid_idx], y[valid_idx]], axis=1)
    return xys, valid_idx


def write_images_and_points3d(paths: dict,
                               pose_files: list,
                               color_files: list,
                               points_xyzrgb: np.ndarray,
                               fx: float, fy: float,
                               cx: float, cy: float,
                               W: int, H: int) -> None:
    """
    Core fix: write images.txt with real POINTS2D entries.

    Each POINTS2D line contains the projected pixel coordinates of visible
    mesh vertices together with their vertex index used as the POINT3D_ID.
    The associate2d3d step uses exactly these (x,y) / ID pairs to link SAM
    mask pixels to 3D locations — without them the pipeline produces nothing.

    points3D.txt is written as an empty file; associate2d3d never reads it.
    """
    colmap_dir = Path(paths["colmap_dir"])
    pts_xyz = points_xyzrgb[:, :3]                        # (N, 3)

    print(f"Projecting {len(pts_xyz)} mesh vertices into {len(pose_files)} frames …")

    # ---- collect per-image projections ---------------------------------
    per_image: List[dict] = []                            # ordered by image_id

    for pose_file, color_file in zip(pose_files, color_files):
        c2w = _load_pose(pose_file)
        if c2w is None:
            per_image.append(None)                        # invalid frame
            continue

        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        # store rotation as quaternion for images.txt
        quat = Rotation.from_matrix(R_w2c).as_quat()     # [x,y,z,w]
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        tx, ty, tz = t_w2c

        xys, valid_idx = _project(pts_xyz, R_w2c, t_w2c,
                                   fx, fy, cx, cy, W, H)

        per_image.append({
            "color_file": color_file,
            "qwxyz": (qw, qx, qy, qz),
            "txyz":  (tx, ty, tz),
            "xys":   xys,            # (K, 2)
            "p3d_ids": valid_idx,    # (K,) vertex indices = POINT3D_IDs
        })

    # ---- write images.txt ----------------------------------------------
    images_path = colmap_dir / "images.txt"
    valid_count = sum(1 for d in per_image if d is not None)

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {valid_count}, mean observations per image: 0.0\n")

        image_id = 1
        for frame_data in per_image:
            if frame_data is None:
                continue                                  # skip invalid poses

            qw, qx, qy, qz = frame_data["qwxyz"]
            tx, ty, tz       = frame_data["txyz"]
            name             = Path(frame_data["color_file"]).name

            f.write(
                f"{image_id} {qw:.16f} {qx:.16f} {qy:.16f} {qz:.16f} "
                f"{tx:.16f} {ty:.16f} {tz:.16f} 1 {name}\n"
            )

            # POINTS2D line — each token is: X Y POINT3D_ID
            xys     = frame_data["xys"]
            p3d_ids = frame_data["p3d_ids"]
            if len(xys) > 0:
                tokens = " ".join(
                    f"{x:.2f} {y:.2f} {int(pid)}"
                    for (x, y), pid in zip(xys, p3d_ids)
                )
                f.write(tokens + "\n")
            else:
                f.write("\n")

            image_id += 1

    print(f"Wrote images.txt  → {images_path}  ({valid_count} valid frames)")

    # ---- write points3D.txt with real mesh vertex positions ----------------
    # bbox_corners.py looks up point XYZ from this file via the COLMAP model
    # loader. We write every vertex that is visible in at least one frame so
    # the bbox step can find 3D coordinates for each component's point IDs.
    #
    # Format per line:
    #   POINT3D_ID  X  Y  Z  R  G  B  ERROR  IMAGE_ID  POINT2D_IDX  ...

    # Build reverse index: vertex_idx → list of (image_id, point2d_idx)
    vertex_to_track: Dict[int, List[tuple]] = defaultdict(list)
    for image_id, frame_data in enumerate(per_image, start=1):
        if frame_data is None:
            continue
        for pt2d_idx, vid in enumerate(frame_data["p3d_ids"]):
            vertex_to_track[int(vid)].append((image_id, pt2d_idx))

    p3d_path = colmap_dir / "points3D.txt"
    written = 0
    with open(p3d_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        n_visible = len(vertex_to_track)
        f.write(f"# Number of points: {n_visible}\n")

        for vid, tracks in vertex_to_track.items():
            x, y, z = pts_xyz[vid]
            r, g, b = points_xyzrgb[vid, 3:6]
            track_str = " ".join(f"{img_id} {p2d_idx}"
                                 for img_id, p2d_idx in tracks[:10])  # cap track len
            f.write(f"{vid} {x:.6f} {y:.6f} {z:.6f} "
                    f"{int(r)} {int(g)} {int(b)} 0.0 {track_str}\n")
            written += 1

    print(f"Wrote points3D.txt → {p3d_path}  ({written} visible vertices)")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a ScanNet scene into scan-to-map format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output layout (matches segment3d/config.py expectations):
  data/<dataset>/ns_data/images/            ← RGB frames
  data/<dataset>/hloc_data/sfm_reconstruction/
      cameras.txt
      images.txt    ← camera poses + mesh vertex projections (POINTS2D)
      points3D.txt  ← empty header

Then run the segment3d pipeline:
  cd segment3d/
  python main.py --dataset scannet_<scene_id>
""",
    )
    parser.add_argument(
        "scannet_scene_dir",
        help="Path to the ScanNet scene directory "
             "(must contain sens/ and <scene_id>_vh_clean_2.ply)",
    )
    parser.add_argument(
        "--output-root",
        default="../data",
        help="Base output directory (default: ../data, "
             "i.e. scan-to-map/data/ when run from data-processor/)",
    )
    parser.add_argument(
        "--num-image-limit",
        type=int,
        default=None,
        dest="num_images_limit",
        help="Uniformly subsample to this many frames (default: use all)",
    )
    parser.add_argument(
        "--max-mesh-points",
        type=int,
        default=50_000,
        help="Maximum mesh vertices to use as 3D points (default: 50000)",
    )
    args = parser.parse_args()

    # ---- build paths -------------------------------------------------------
    paths = build_scene_paths(args.scannet_scene_dir, args.output_root)
    print(f"\nDataset name : {paths['dataset_name']}")
    print(f"COLMAP output: {paths['colmap_dir']}")
    print(f"Images output: {paths['images_dir']}\n")

    # ---- collect + optionally subsample frames -----------------------------
    color_dir = Path(paths["scannet_color_dir"])
    pose_dir  = Path(paths["scannet_pose_dir"])

    color_files = _sorted_color_files(color_dir)
    pose_files  = _sorted_pose_files(pose_dir)

    if not color_files:
        raise FileNotFoundError(f"No color images found in {color_dir}")
    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {pose_dir}")

    if args.num_images_limit and args.num_images_limit < len(pose_files):
        pose_files, color_files = _uniform_subsample(
            pose_files, color_files, args.num_images_limit
        )
        print(f"Subsampled to {len(pose_files)} frames")
    else:
        print(f"Using all {len(pose_files)} frames")

    pose_files  = [str(p) for p in pose_files]
    color_files = [str(c) for c in color_files]

    # ---- read intrinsics + image size --------------------------------------
    fx, fy, cx, cy, W, H = read_intrinsics(paths)
    print(f"Intrinsics : fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"Image size : {W}×{H}\n")

    # ---- Step 1: copy images -----------------------------------------------
    copy_images(paths, color_files)

    # ---- Step 2: cameras.txt -----------------------------------------------
    write_cameras_file(paths, W, H, fx, fy, cx, cy)

    # ---- Step 3: load mesh vertices ----------------------------------------
    points_xyzrgb = load_mesh_points(paths["mesh_ply_path"], args.max_mesh_points)

    # ---- Step 4: images.txt + points3D.txt ---------------------------------
    write_images_and_points3d(
        paths, pose_files, color_files,
        points_xyzrgb, fx, fy, cx, cy, W, H,
    )

    # ---- summary -----------------------------------------------------------
    print(f"""
Done!  Dataset '{paths['dataset_name']}' is ready.

Next steps:
  1. Make sure segment3d checkpoints are in place:
       scan-to-map/checkpoints/FastSAM-x.pt
  2. Run the segment3d pipeline:
       cd scan-to-map/segment3d/
       python main.py --dataset {paths['dataset_name']}
  3. (Optional) Evaluate with ScanQA:
       python eval/scanqa_eval.py \\
           --scanqa-root /path/to/ScanQA/data/qa/ \\
           --outputs-root ../outputs
""")


if __name__ == "__main__":
    main()
