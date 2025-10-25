"""Build a graph connecting masks across images based on shared 3D points."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import networkx as nx


def jaccard(a: Set[int], b: Set[int]) -> float:
    """
    Compute the Jaccard similarity between two sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        a: First set
        b: Second set

    Returns:
        Jaccard similarity in range [0, 1]
    """
    if len(a) == 0 and len(b) == 0:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


def build_edges(
    per_image_sets: Dict[Tuple[int, int], Set[int]], K: int = 5, tau: float = 0.2
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Dict]]:
    """
    Build edges between mask nodes based on shared 3D points.

    For every pair of nodes from different images, compute the overlap size |S|
    and Jaccard similarity J. If |S| ≥ K or J ≥ tau, add an edge with attributes.

    Args:
        per_image_sets: Dictionary mapping (image_id, mask_idx) to set of 3D point IDs
        K: Minimum overlap size threshold (default: 5)
        tau: Minimum Jaccard similarity threshold (default: 0.2)

    Returns:
        List of edges as tuples: (node1, node2, attributes_dict)
        where attributes_dict contains "overlap" and "jaccard" keys
    """
    edges = []
    nodes = list(per_image_sets.keys())

    # Compare all pairs of nodes from different images
    for i in range(len(nodes)):
        node1 = nodes[i]
        image_id1, mask_idx1 = node1
        set1 = per_image_sets[node1]

        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            image_id2, mask_idx2 = node2

            # Only consider pairs from different images
            if image_id1 == image_id2:
                continue

            set2 = per_image_sets[node2]

            # Compute overlap and Jaccard similarity
            overlap_size = len(set1 & set2)
            jaccard_sim = jaccard(set1, set2)

            # Add edge if threshold conditions are met
            if overlap_size >= K or jaccard_sim >= tau:
                edge_attrs = {"overlap": overlap_size, "jaccard": jaccard_sim}
                edges.append((node1, node2, edge_attrs))

    return edges


def to_networkx(
    nodes: List[Tuple[int, int]],
    edges: List[Tuple[Tuple[int, int], Tuple[int, int], Dict]],
) -> nx.Graph:
    """
    Convert nodes and edges to a NetworkX graph.

    Args:
        nodes: List of node tuples (image_id, mask_idx)
        edges: List of edge tuples (node1, node2, attributes_dict)

    Returns:
        NetworkX Graph with nodes and edges
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


def load_associations(
    associations_dir, min_points: int = 10
) -> Dict[Tuple[int, int], Set[int]]:
    """
    Load all association files and build per-mask 3D point sets.

    Args:
        associations_dir: Directory containing imageId_*.json files
        min_points: Minimum number of points required for a mask to be included

    Returns:
        Dictionary mapping (image_id, mask_idx) to set of 3D point IDs
    """
    import json
    from pathlib import Path

    associations_dir = Path(associations_dir)
    per_image_sets: Dict[Tuple[int, int], Set[int]] = {}

    # Find all association files
    association_files = sorted(associations_dir.glob("imageId_*.json"))

    if not association_files:
        raise ValueError(f"No association files found in {associations_dir}")

    print(f"Found {len(association_files)} association files")

    # Load each file and build the mapping
    for assoc_file in association_files:
        with assoc_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_id = data["image_id"]
        mask_point3d_sets = data["mask_point3d_sets"]

        # Create a node for each mask
        for mask_idx, point3d_list in enumerate(mask_point3d_sets):
            if len(point3d_list) < min_points:
                continue
            node_key = (image_id, mask_idx)
            per_image_sets[node_key] = set(point3d_list)

    return per_image_sets


def build_mask_graph_cli(
    dataset_name: str, K: int = 5, tau: float = 0.2, min_points_in_3D_segment: int = 100
) -> None:
    """
    Build a mask graph from association files.

    Args:
        K: Minimum overlap size threshold (default: 5)
        tau: Minimum Jaccard similarity threshold (default: 0.2)
        min_points_in_3D_segment: Minimum points in connected component (default: 100)
    """
    import json
    import pickle
    from .io_paths import get_associations_dir, get_outputs_dir, load_config

    # Load configuration
    config = load_config(dataset_name)

    associations_dir = get_associations_dir(config)
    outputs_dir = get_outputs_dir(config)

    print(f"Associations directory: {associations_dir}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Parameters: K={K}, tau={tau}")

    # Load associations
    print("\nLoading associations...")
    per_image_sets = load_associations(associations_dir)

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
    print("\nBuilding edges...")
    edges = build_edges(per_image_sets, K=K, tau=tau)

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
        "parameters": {"K": K, "tau": tau},
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
