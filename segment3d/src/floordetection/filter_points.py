import json
import argparse
import numpy as np
import sys
from pathlib import Path
from scipy.spatial import KDTree

from ..io_paths import get_outputs_dir, load_config


def load_points(filepath):
    """Loads points from the specified JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'.")
        sys.exit(1)


def save_points(points, output_dir, filename="filtered_floor_3d_points.json"):
    """Saves the filtered list of dictionaries to a JSON file."""
    output_path = Path(output_dir) / filename
    with open(output_path, "w") as f:
        json.dump(points, f, indent=2)
    print(f"Filtered points saved to {output_path}")


def distance_based_filter(data, up_axis_index, tolerance=0.3):
    """
    Filters points that are close to the lowest value in the up_axis.

    Args:
        data: List of dicts [{'point_id': int, 'coords': [x,y,z]}]
        up_axis_index: 1 for 'y', 2 for 'z'
        tolerance: Distance from the minimum floor height to include.
    """
    print(f"Running Distance Based Filter (Axis index: {up_axis_index})...")

    # Extract just the values for the relevant axis to process quickly using numpy
    coords = np.array([p["coords"] for p in data])
    height_values = coords[:, up_axis_index]

    # We assume the floor is at the bottom.
    # To be robust against outliers (one point deep underground),
    # we take the 1st percentile instead of the absolute min.
    floor_height = np.percentile(height_values, 1)

    cutoff = floor_height + tolerance

    filtered_data = []
    for point in data:
        if point["coords"][up_axis_index] <= cutoff:
            filtered_data.append(point)

    print(f"Found floor height approx: {floor_height:.4f}")
    return filtered_data


def normal_based_filter(data, up_axis_index, k_neighbors=15, angle_threshold_deg=20):
    """
    Filters points by estimating surface normals. Keeps points where the normal
    is roughly parallel to the up-vector (horizontal surfaces).

    Args:
        data: List of dicts
        up_axis_index: 1 for 'y', 2 for 'z'
        k_neighbors: Number of neighbors to use for normal estimation (PCA)
        angle_threshold_deg: Max deviation from the up-vector in degrees.
    """
    print(f"Running Normal Based Filter (calculating local geometry)...")

    coords = np.array([p["coords"] for p in data])
    n_points = len(coords)

    if n_points < k_neighbors:
        print("Not enough points to calculate normals.")
        return []

    # 1. Build a KDTree for fast neighbor lookup
    tree = KDTree(coords)

    # 2. Define the global 'Up' vector
    up_vector = np.zeros(3)
    up_vector[up_axis_index] = 1.0  # e.g., [0, 1, 0]

    filtered_data = []

    # Cosine threshold
    # Dot product of normalized vectors = cos(theta).
    # We want theta < threshold, so dot > cos(threshold)
    cos_threshold = np.cos(np.radians(angle_threshold_deg))

    # 3. Iterate points to estimate normal using PCA (Principal Component Analysis)
    # Query all neighbors at once for efficiency in smaller datasets,
    # or chunked for large ones. Here we loop for clarity and memory safety.
    for i in range(n_points):
        # Get indices of k nearest neighbors
        # (k+1 because the point includes itself)
        _, idxs = tree.query(coords[i], k=k_neighbors)

        # Get neighbor coordinates
        neighbors = coords[idxs]

        # PCA: Center the data
        centered = neighbors - np.mean(neighbors, axis=0)

        # Compute Covariance Matrix
        cov = np.cov(centered.T)

        # Eigen decomposition.
        # The normal vector is the eigenvector corresponding to the smallest eigenvalue.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # The smallest eigenvalue is at index 0 in eigh output (sorted)
        normal = eigenvectors[:, 0]

        # 4. Check alignment with Up Vector
        # We take absolute value of dot product because normals can point up or down
        # and still represent a horizontal floor.
        dot_product = abs(np.dot(normal, up_vector))

        if dot_product >= cos_threshold:
            # It is a flat surface
            filtered_data.append(data[i])

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter point cloud to find the floor."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to process"
    )
    parser.add_argument(
        "--up_axis",
        type=str,
        choices=["y", "z"],
        default="z",
        help="The vertical axis ('y' or 'z')",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["distance", "normal"],
        default="normal",
        help="Filtering method to use",
    )

    args = parser.parse_args()

    # Normalize arguments
    up_axis_map = {"y": 1, "z": 2}
    axis_idx = up_axis_map[args.up_axis]

    # Load configuration and construct filepath
    config = load_config(args.dataset)
    outputs_dir = Path(get_outputs_dir(config))
    filepath = outputs_dir / "floor_3d_points.json"

    # Load
    print(f"Loading {filepath}...")
    data = load_points(filepath)
    print(f"Total points loaded: {len(data)}")

    # Process
    if args.method == "distance":
        filtered = distance_based_filter(data, axis_idx)
    else:
        filtered = normal_based_filter(data, axis_idx)

    # Output
    print(f"Points remaining after filter: {len(filtered)}")
    save_points(filtered, outputs_dir)


if __name__ == "__main__":
    main()
