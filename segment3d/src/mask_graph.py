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
