"""Build a graph connecting masks across images based on shared 3D points."""

from __future__ import annotations

import concurrent.futures
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


def load_single_association(
    file_path: Path, min_points: int
) -> List[Tuple[Tuple[int, int], Set[int]]]:
    """
    Worker function to load a single JSON file.
    Returns a list of ((image_id, mask_idx), point_set) tuples.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = data["image_id"]
    mask_point3d_sets = data["mask_point3d_sets"]
    results = []

    for mask_idx, point3d_list in enumerate(mask_point3d_sets):
        if len(point3d_list) < min_points:
            continue
        # Convert to set here to save main thread from doing it
        results.append(((image_id, mask_idx), set(point3d_list)))

    return results


def load_associations_parallel(
    associations_dir, min_points: int = 10
) -> Dict[Tuple[int, int], Set[int]]:
    """
    Load all association files in parallel using ProcessPoolExecutor.

    Args:
        associations_dir: Directory containing imageId_*.json files
        min_points: Minimum number of points required for a mask to be included

    Returns:
        Dictionary mapping (image_id, mask_idx) to set of 3D point IDs
    """
    associations_dir = Path(associations_dir)
    per_image_sets: Dict[Tuple[int, int], Set[int]] = {}

    # Find all association files
    association_files = sorted(associations_dir.glob("imageId_*.json"))

    if not association_files:
        raise ValueError(f"No association files found in {associations_dir}")

    print(f"Found {len(association_files)} association files. Loading in parallel...")

    # Use ProcessPoolExecutor to parallelize JSON parsing and set construction
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [
            executor.submit(load_single_association, f, min_points)
            for f in association_files
        ]

        # Gather results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                for node_key, point_set in results:
                    per_image_sets[node_key] = point_set
            except Exception as exc:
                print(f"File loading generated an exception: {exc}")

    return per_image_sets


def build_edges_scipy(
    per_image_sets: Dict[Tuple[int, int], Set[int]], K: int = 5, tau: float = 0.2
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Dict]]:
    """
    Build edges using Scipy Sparse Matrices.
    Performance: O(N_associations) + Matrix Multiplication (highly optimized).

    Args:
        per_image_sets: Dictionary mapping (image_id, mask_idx) to set of 3D point IDs
        K: Minimum overlap size threshold
        tau: Minimum Jaccard similarity threshold

    Returns:
        List of edges as tuples: (node1, node2, attributes_dict)
    """
    # 1. Mappings: Nodes -> Index and Points -> Index
    nodes_list = list(per_image_sets.keys())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}

    # Collect all unique points efficiently
    all_points = set()
    total_associations = 0
    for s in per_image_sets.values():
        all_points.update(s)
        total_associations += len(s)

    point_to_idx = {pt: i for i, pt in enumerate(all_points)}

    num_nodes = len(nodes_list)
    num_points = len(all_points)

    print(f"Constructing sparse matrix ({num_nodes} masks x {num_points} points)...")

    # 2. Build Coordinate Arrays for Sparse Matrix
    row_inds = np.zeros(total_associations, dtype=np.int32)
    col_inds = np.zeros(total_associations, dtype=np.int32)

    current_idx = 0
    for node_idx, node in enumerate(nodes_list):
        points = per_image_sets[node]
        for pt in points:
            row_inds[current_idx] = node_idx
            col_inds[current_idx] = point_to_idx[pt]
            current_idx += 1

    # All values are 1 (binary association)
    data = np.ones(total_associations, dtype=np.int32)

    # Create CSR Matrix
    M = sparse.csr_matrix((data, (row_inds, col_inds)), shape=(num_nodes, num_points))

    # 3. Compute Intersections via Matrix Multiplication
    # O[i, j] = dot(Row_i, Row_j) = number of shared points
    print("Computing intersection matrix (M * M.T)...")
    intersection_matrix = M.dot(M.T)

    # 4. Pre-compute Set Sizes for Jaccard
    # Row sum of M is the number of points in each mask
    set_sizes = np.array(M.sum(axis=1)).flatten()

    # 5. Extract Edges
    # We only need the upper triangle to avoid duplicates and self-loops
    print("Filtering edges and calculating Jaccard...")
    upper_tri = sparse.triu(intersection_matrix, k=1).tocoo()

    edges = []

    # Iterate strictly over non-zero overlaps
    for i, j, overlap in zip(upper_tri.row, upper_tri.col, upper_tri.data):
        node1 = nodes_list[i]
        node2 = nodes_list[j]

        # Optimization: Skip masks from the same image
        if node1[0] == node2[0]:
            continue

        # Jaccard Calculation: |A n B| / (|A| + |B| - |A n B|)
        union_size = set_sizes[i] + set_sizes[j] - overlap

        if union_size == 0:
            jaccard_sim = 0.0
        else:
            jaccard_sim = overlap / union_size

        # Threshold Check
        if overlap >= K or jaccard_sim >= tau:
            edge_attrs = {"overlap": int(overlap), "jaccard": float(jaccard_sim)}
            edges.append((node1, node2, edge_attrs))

    return edges


def to_networkx(
    nodes: List[Tuple[int, int]],
    edges: List[Tuple[Tuple[int, int], Tuple[int, int], Dict]],
) -> nx.Graph:
    """
    Convert nodes and edges to a NetworkX graph.
    """
    G = nx.Graph()

    # Add nodes with attributes
    for node in nodes:
        image_id, mask_idx = node
        G.add_node(node, image_id=image_id, mask_idx=mask_idx)

    # Add edges with attributes
    for node1, node2, attrs in edges:
        G.add_edge(node1, node2, **attrs)

    return G


def build_mask_graph_cli(
    dataset_name: str, K: int = 5, tau: float = 0.2, min_points_in_3D_segment: int = 100
) -> None:
    """
    Build a mask graph from association files.
    """
    from .io_paths import get_associations_dir, get_outputs_dir, load_config

    # Load configuration
    config = load_config(dataset_name)

    associations_dir = get_associations_dir(config)
    outputs_dir = get_outputs_dir(config)

    print(f"Associations directory: {associations_dir}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Parameters: K={K}, tau={tau}")

    # Load associations (Parallelized)
    print("\nLoading associations...")
    per_image_sets = load_associations_parallel(associations_dir)

    # Count statistics
    total_nodes = len(per_image_sets)
    total_points = sum(len(s) for s in per_image_sets.values())
    non_empty_nodes = sum(1 for s in per_image_sets.values() if len(s) > 0)

    print(f"Total nodes (masks): {total_nodes}")
    print(f"Non-empty nodes: {non_empty_nodes}")
    print(f"Total point associations: {total_points}")
    print(
        f"Average points per non-empty mask: {total_points / non_empty_nodes if non_empty_nodes > 0 else 0:.2f}"
    )

    # Build edges
    print("\nBuilding edges (Scipy Sparse)...")
    edges = build_edges_scipy(per_image_sets, K=K, tau=tau)

    print(f"Created {len(edges)} edges")

    # Convert to NetworkX graph
    print("\nCreating NetworkX graph...")
    nodes = list(per_image_sets.keys())
    G = to_networkx(nodes, edges)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute graph statistics
    if G.number_of_edges() > 0:
        connected_components = list(nx.connected_components(G))
        print(f"Connected components: {len(connected_components)}")
        largest_cc_size = max(len(cc) for cc in connected_components)
        print(f"Largest component size: {largest_cc_size}")

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        if degrees:
            print(f"Average degree: {sum(degrees) / len(degrees):.2f}")
            print(f"Max degree: {max(degrees)}")

    # Save graph
    output_path = outputs_dir / "mask_graph.gpickle"
    with output_path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved graph to: {output_path}")

    # Save connected components with their 3D point IDs
    print("\nExtracting connected components...")
    connected_components_list = []

    if G.number_of_nodes() > 0:
        connected_components = list(nx.connected_components(G))

        for comp_id, component in enumerate(connected_components):
            # Union all 3D point IDs from all nodes in this component
            point3d_union = set()
            for node in component:
                point3d_union.update(per_image_sets[node])

            if len(point3d_union) >= min_points_in_3D_segment:
                connected_components_list.append(
                    {
                        "connected_comp_id": comp_id,
                        "set_of_point3DIds": sorted(list(point3d_union)),
                    }
                )

        print(f"Found {len(connected_components_list)} connected components")

        # Print some statistics about components
        if connected_components_list:
            comp_sizes = [
                len(cc["set_of_point3DIds"]) for cc in connected_components_list
            ]
            print(f"  Largest component: {max(comp_sizes)} 3D points")
            print(f"  Smallest component: {min(comp_sizes)} 3D points")
            print(
                f"  Average points per component: {sum(comp_sizes) / len(comp_sizes):.2f}"
            )

    # Save connected components
    components_path = outputs_dir / "connected_components.json"
    with components_path.open("w", encoding="utf-8") as f:
        json.dump(connected_components_list, f, indent=2)
    print(f"Saved connected components to: {components_path}")

    # Save summary statistics
    stats = {
        "parameters": {
            "K": K,
            "tau": tau,
            "min_points_in_3D_segment": min_points_in_3D_segment,
        },
        "nodes": {"total": total_nodes, "non_empty": non_empty_nodes},
        "edges": {"total": G.number_of_edges()},
        "point_associations": {
            "total": total_points,
            "average_per_mask": (
                total_points / non_empty_nodes if non_empty_nodes > 0 else 0
            ),
        },
    }

    if G.number_of_edges() > 0:
        connected_components = list(nx.connected_components(G))
        degrees = [d for n, d in G.degree()]
        stats["graph_statistics"] = {
            "connected_components": len(connected_components),
            "largest_component_size": max(len(cc) for cc in connected_components),
            "average_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }

    stats_path = outputs_dir / "mask_graph_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")


def main() -> None:
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build mask graph from 2D-3D associations"
    )

    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to process"
    )
    parser.add_argument(
        "--K", type=int, default=5, help="Minimum overlap size threshold (default: 5)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.2,
        help="Minimum Jaccard similarity threshold (default: 0.2)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=100,
        help="Minimum points in connected component (default: 100)",
    )

    args = parser.parse_args()

    build_mask_graph_cli(
        dataset_name=args.dataset_name,
        K=args.K,
        tau=args.tau,
        min_points_in_3D_segment=args.min_points,
    )


if __name__ == "__main__":
    main()
