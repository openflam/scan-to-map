"""CLI script to build mask graph from 2D-3D associations."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple

import networkx as nx

from src.io_paths import get_associations_dir, get_outputs_dir, load_config
from src.mask_graph import build_edges, to_networkx


def load_associations(
    associations_dir: Path, min_points: int = 10
) -> Dict[Tuple[int, int], Set[int]]:
    """
    Load all association files and build per-mask 3D point sets.

    Args:
        associations_dir: Directory containing imageId_*.json files

    Returns:
        Dictionary mapping (image_id, mask_idx) to set of 3D point IDs
    """
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


def build_mask_graph(
    K: int = 5, tau: float = 0.2, min_points_in_3D_segment: int = 100
) -> None:
    """
    Build a mask graph from association files.

    Args:
        K: Minimum overlap size threshold (default: 5)
        tau: Minimum Jaccard similarity threshold (default: 0.2)
    """
    # Load configuration
    config = load_config()

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
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build mask graph from 2D-3D associations"
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

    args = parser.parse_args()

    build_mask_graph(K=args.K, tau=args.tau)


if __name__ == "__main__":
    main()
