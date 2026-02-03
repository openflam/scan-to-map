import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation


def build_scene_paths(scannet_scene_dir: str, output_root: str = "data") -> dict:
    """
    Centralize all paths (input + output) for ScanNet scene processing.

    Args:
        scannet_scene_dir: Path to the ScanNet scene directory containing color/, intrinsic/, pose/
        output_root: Base directory for COLMAP data (default "data")

    Returns:
        Dictionary containing all relevant paths for the scene
    """
    scannet_scene_path = Path(scannet_scene_dir)

    # Extract scene identifier from path (e.g., "scene0000_00" from the directory name)
    scene_id = scannet_scene_path.name
    scene_name = f"scannet_{scene_id}"

    # Input paths
    scannet_color_dir = scannet_scene_path / "sens" / "color"
    scannet_intrinsic_path = (
        scannet_scene_path / "sens" / "intrinsic" / "intrinsic_color.txt"
    )
    scannet_pose_dir = scannet_scene_path / "sens" / "pose"

    # Output paths
    output_root_path = Path(output_root)
    colmap_root = output_root_path / scene_name / "colmap_known_poses"
    images_dir = output_root_path / scene_name / "ns_data" / "images"

    # Create output directories
    colmap_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    return {
        "scannet_color_dir": str(scannet_color_dir),
        "scannet_intrinsic_path": str(scannet_intrinsic_path),
        "scannet_pose_dir": str(scannet_pose_dir),
        "colmap_root": str(colmap_root),
        "images_dir": str(images_dir),
    }


def prepare_cameras_file(paths: dict) -> None:
    """
    Read ScanNet intrinsic matrix and convert to COLMAP cameras.txt format.

    Args:
        paths: Dictionary containing 'scannet_intrinsic_path', 'scannet_color_dir', and 'colmap_root'
    """
    # Read the 4x4 intrinsic matrix from ScanNet
    intrinsic_path = Path(paths["scannet_intrinsic_path"])
    intrinsic_matrix = np.loadtxt(intrinsic_path)

    # Extract intrinsic parameters from the matrix
    # Format: [[fx, 0, cx, 0],
    #          [0, fy, cy, 0],
    #          [0, 0, 1, 0],
    #          [0, 0, 0, 1]]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Get image dimensions from a sample color image
    color_dir = Path(paths["scannet_color_dir"])
    sample_images = list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.png"))
    if not sample_images:
        raise FileNotFoundError(f"No images found in {color_dir}")

    with Image.open(sample_images[0]) as img:
        width, height = img.size

    # Create sparse directory structure
    colmap_root = Path(paths["colmap_root"])
    sparse_dir = colmap_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Write cameras.txt in COLMAP format using PINHOLE model
    # PINHOLE model parameters: fx, fy, cx, cy
    cameras_path = sparse_dir / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    print(f"Created cameras.txt at {cameras_path}")


def create_empty_points3D_file(paths: dict) -> None:
    """
    Create an empty points3D.txt file in the COLMAP sparse directory.

    Args:
        paths: Dictionary containing 'colmap_root'
    """
    colmap_root = Path(paths["colmap_root"])
    sparse_dir = colmap_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    points3D_path = sparse_dir / "points3D.txt"
    with open(points3D_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")

    print(f"Created empty points3D.txt at {points3D_path}")


def prepare_images_file(paths: dict, num_images_limit: int = None) -> None:
    """
    Read ScanNet pose files and convert to COLMAP images.txt format.

    Args:
        paths: Dictionary containing 'scannet_pose_dir', 'scannet_color_dir', and 'colmap_root'
        num_images_limit: Maximum number of images to sample uniformly (default: None, use all images)
    """
    pose_dir = Path(paths["scannet_pose_dir"])
    color_dir = Path(paths["scannet_color_dir"])
    colmap_root = Path(paths["colmap_root"])
    sparse_dir = colmap_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Get all pose files and sort them
    pose_files = sorted(pose_dir.glob("*.txt"), key=lambda x: int(x.stem))

    if not pose_files:
        raise FileNotFoundError(f"No pose files found in {pose_dir}")

    # Get corresponding image files
    color_files = sorted(color_dir.glob("*.jpg")) + sorted(color_dir.glob("*.png"))
    color_files = sorted(color_files, key=lambda x: int(x.stem))

    # Uniformly sample if num_images_limit is specified
    if num_images_limit is not None and num_images_limit < len(pose_files):
        total_images = len(pose_files)
        indices = np.linspace(0, total_images - 1, num_images_limit, dtype=int)
        pose_files = [pose_files[i] for i in indices]
        color_files = [color_files[i] for i in indices]

    images_path = sparse_dir / "images.txt"
    with open(images_path, "w") as f:
        # Write header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(
            f"# Number of images: {len(pose_files)}, mean observations per image: 0.0\n"
        )

        for image_id, (pose_file, color_file) in enumerate(
            zip(pose_files, color_files), start=1
        ):
            # Read the 4x4 camera-to-world transformation matrix
            c2w_matrix = np.loadtxt(pose_file)

            # Extract rotation and translation from camera-to-world
            R_c2w = c2w_matrix[:3, :3]
            t_c2w = c2w_matrix[:3, 3]

            # Convert to world-to-camera (COLMAP convention)
            # R_w2c = R_c2w^T
            # t_w2c = -R_w2c @ t_c2w
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            # Convert rotation matrix to quaternion using scipy
            rotation = Rotation.from_matrix(R_w2c)
            quat = rotation.as_quat()  # Returns [x, y, z, w]

            # COLMAP uses [w, x, y, z] format (Hamilton convention)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            tx, ty, tz = t_w2c

            # Camera ID is always 1 (single camera model)
            camera_id = 1
            image_name = color_file.name

            # Write image line
            f.write(f"{image_id} {qw:.16f} {qx:.16f} {qy:.16f} {qz:.16f} ")
            f.write(f"{tx:.16f} {ty:.16f} {tz:.16f} {camera_id} {image_name}\n")

            # Write empty line for POINTS2D (no features yet)
            f.write("\n")

    print(f"Created images.txt with {len(pose_files)} images at {images_path}")


def main():
    """Command-line interface for build_scene_paths."""
    parser = argparse.ArgumentParser(
        description="Build and display paths for ScanNet scene processing"
    )
    parser.add_argument(
        "scannet_scene_dir",
        type=str,
        help="Path to the ScanNet scene directory (e.g., scans/scene0000_00)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="../data",
        help="Base directory for COLMAP data (default: ../data)",
    )
    parser.add_argument(
        "--num-image-limit",
        type=int,
        dest="num_images_limit",
        default=None,
        help="Maximum number of images to sample uniformly (default: None, use all images)",
    )

    args = parser.parse_args()

    paths = build_scene_paths(args.scannet_scene_dir, args.output_root)

    # Create initial COLMAP files with known poses
    prepare_cameras_file(paths)
    prepare_images_file(paths, num_images_limit=args.num_images_limit)
    create_empty_points3D_file(paths)


if __name__ == "__main__":
    main()
