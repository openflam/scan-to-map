"""
mask_graph.py - Build a graph connecting per-object SAM3 instances across sequences
based on shared COLMAP 3D points.

Reads:
    <outputs_dir>/object_level_masks/object_3d_associations.json

produced by associate2d3d.py.  Each node in the graph is one
(obj_slug, seq_key, obj_id) triple.  Edges connect nodes from *different*
sequences that share enough 3D points (Jaccard ≥ tau or overlap ≥ K).
The merge is object-name-agnostic: nodes from different objects may be linked.

Outputs (written to <outputs_dir>/object_level_masks/):
    object_mask_graph.gpickle          – NetworkX graph (pickle)
    object_connected_components.json   – list of merged instances with point3D IDs
    object_mask_graph_stats.json       – summary statistics

Usage (from the segment3d/ directory):
    python -m src.per_object_sam3.mask_graph --dataset ProjectLabStudio_inv_method
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Path plumbing
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent  # src/per_object_sam3
_SRC_DIR = _SCRIPT_DIR.parent  # src/
_SEGMENT3D_DIR = _SRC_DIR.parent  # segment3d/
for _p in [str(_SRC_DIR), str(_SEGMENT3D_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ..io_paths import get_outputs_dir, load_config  # noqa: E402

# Node type: (seq_key, obj_slug, obj_id_str)
# Using seq_key as the first element so the "same-sequence" skip is node[0] == node[0],
# mirroring the "same image" guard in src/mask_graph.py.
Node = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Load associations
# ---------------------------------------------------------------------------


def load_associations(
    associations_path: Path, min_points: int = 1
) -> Tuple[Dict[Node, Set[int]], Dict[Node, str]]:
    """
    Load object_3d_associations.json and return:
      - per_instance_sets  : Node → set of 3D point IDs
      - node_to_instance_id: Node → human-readable instance key string

    Nodes with fewer than *min_points* 3D associations are dropped.
    """
    with associations_path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Dict[str, Dict[str, List[int]]]] = json.load(fh)

    per_instance_sets: Dict[Node, Set[int]] = {}
    node_to_instance_id: Dict[Node, str] = {}

    for obj_slug, seq_map in data.items():
        for seq_key, obj_map in seq_map.items():
            for obj_id_str, point_list in obj_map.items():
                if len(point_list) < min_points:
                    continue
                node: Node = (seq_key, obj_slug, obj_id_str)
                per_instance_sets[node] = set(point_list)
                node_to_instance_id[node] = f"{obj_slug}_{seq_key}_{obj_id_str}"

    return per_instance_sets, node_to_instance_id


# ---------------------------------------------------------------------------
# Edge building  (same sparse-matrix logic as src/mask_graph.py)
# ---------------------------------------------------------------------------


def build_edges_scipy(
    per_instance_sets: Dict[Node, Set[int]],
    K: int = 5,
    tau: float = 0.2,
) -> List[Tuple[Node, Node, Dict[str, Any]]]:
    """
    Build edges using a sparse incidence matrix + M·Mᵀ intersection trick.

    Two nodes are connected when:
        overlap ≥ K  OR  Jaccard(A, B) ≥ tau
    """
    nodes_list = list(per_instance_sets.keys())
    if not nodes_list:
        return []

    # --- collect all unique 3D point IDs ---
    all_points: Set[int] = set()
    total_assoc = 0
    for s in per_instance_sets.values():
        all_points.update(s)
        total_assoc += len(s)

    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    num_nodes = len(nodes_list)
    num_points = len(all_points)

    print(f"Constructing sparse matrix ({num_nodes} instances × {num_points} points)…")

    # --- build COO arrays ---
    row_inds = np.zeros(total_assoc, dtype=np.int32)
    col_inds = np.zeros(total_assoc, dtype=np.int32)
    cur = 0
    for ni, node in enumerate(nodes_list):
        for pt in per_instance_sets[node]:
            row_inds[cur] = ni
            col_inds[cur] = point_to_idx[pt]
            cur += 1

    data = np.ones(total_assoc, dtype=np.int32)
    M = sparse.csr_matrix((data, (row_inds, col_inds)), shape=(num_nodes, num_points))

    print("Computing intersection matrix (M · Mᵀ)…")
    intersection_matrix = M.dot(M.T)

    set_sizes = np.array(M.sum(axis=1)).flatten()

    print("Filtering edges and calculating Jaccard…")
    upper_tri = sparse.triu(intersection_matrix, k=1).tocoo()

    edges: List[Tuple[Node, Node, Dict[str, Any]]] = []
    for i, j, overlap in zip(upper_tri.row, upper_tri.col, upper_tri.data):
        node1 = nodes_list[i]
        node2 = nodes_list[j]

        union_size = set_sizes[i] + set_sizes[j] - overlap
        jaccard_sim = float(overlap / union_size) if union_size > 0 else 0.0

        if overlap >= K or jaccard_sim >= tau:
            edges.append(
                (node1, node2, {"overlap": int(overlap), "jaccard": jaccard_sim})
            )

    return edges


# ---------------------------------------------------------------------------
# Geometric (voxel-based) edge building
# ---------------------------------------------------------------------------


def build_edges_geometric_intersection(
    per_instance_sets: Dict[Node, Set[int]],
    points3D: Dict[int, Any],
    voxel_size_cm: float = 10.0,
    tau: float = 0.8,
) -> List[Tuple[Node, Node, Dict[str, Any]]]:
    """
    Build edges using voxel-based geometric intersection.

    Each instance's 3D point set is projected onto a voxel grid whose cell
    side length is *voxel_size_cm* centimetres (point coordinates are assumed
    to be in metres).  The grid spans the global bounding box of **all points
    in the COLMAP model** (not just the points belonging to the instances under
    comparison).  This ensures voxel indices are consistent across all instance
    pairs and that the grid is not artificially shrunk to fit only the
    instances being compared (which would inflate Jaccard similarity).

    Two instances are connected when their voxel-occupancy Jaccard similarity
    meets the threshold:

        Jaccard(A, B) ≥ tau

    The same sparse incidence matrix + M·Mᵀ trick employed in
    :func:`build_edges_scipy` is used, but the columns represent voxels
    instead of raw point IDs.

    Args:
        per_instance_sets:  Node → set of COLMAP 3D point IDs.
        points3D:           COLMAP ``points3D`` dict
                            (``point_id → Point3D`` namedtuple with ``.xyz``).
        voxel_size_cm:      Side length of each voxel in centimetres
                            (point coordinates are in metres).  The per-axis
                            grid resolution is derived as
                            ``ceil(extent_m / (voxel_size_cm / 100))``.
        tau:                Min Jaccard similarity of voxel-occupancy sets.

    Returns:
        List of ``(node1, node2, attrs)`` edge triples where *attrs* contains
        ``voxel_overlap`` (int) and ``geometric_jaccard`` (float).
    """
    nodes_list = list(per_instance_sets.keys())
    if not nodes_list:
        return []

    # ------------------------------------------------------------------
    # Global bounding box: use ALL points in the COLMAP model so that
    # voxel indices are stable and not biased toward the instances being
    # compared.
    # ------------------------------------------------------------------
    if not points3D:
        print("Warning: points3D is empty; returning no edges.")
        return []

    all_scene_xyz = np.stack(
        [np.array(pt.xyz, dtype=np.float64) for pt in points3D.values()], axis=0
    )  # (N_scene, 3)
    min_xyz = all_scene_xyz.min(axis=0)
    max_xyz = all_scene_xyz.max(axis=0)
    extent = max_xyz - min_xyz
    # Avoid division-by-zero on degenerate axes.
    extent = np.where(extent == 0, 1.0, extent)

    voxel_size_m = voxel_size_cm / 100.0
    grid_xyz = np.maximum(np.ceil(extent / voxel_size_m).astype(np.int64), 1)
    grid_x, grid_y, grid_z = int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2])

    print(
        f"Voxel grid: {grid_x}×{grid_y}×{grid_z} (voxel_size={voxel_size_cm}cm)  "
        f"bbox [{min_xyz[0]:.3f}, {min_xyz[1]:.3f}, {min_xyz[2]:.3f}] → "
        f"[{max_xyz[0]:.3f}, {max_xyz[1]:.3f}, {max_xyz[2]:.3f}]"
    )

    def _point_to_voxel(xyz: np.ndarray) -> int:
        """Map an XYZ coordinate to a flat voxel index."""
        frac = (xyz - min_xyz) / extent  # values in [0, 1]
        ix, iy, iz = (frac * grid_xyz).astype(np.int64).clip(0, grid_xyz - 1)
        return int(ix * grid_y * grid_z + iy * grid_z + iz)

    # ------------------------------------------------------------------
    # Resolve instance point IDs → xyz (skip IDs absent from the model).
    # ------------------------------------------------------------------
    all_instance_point_ids: Set[int] = set()
    for s in per_instance_sets.values():
        all_instance_point_ids.update(s)

    point_xyz: Dict[int, np.ndarray] = {
        pid: np.array(points3D[pid].xyz, dtype=np.float64)
        for pid in all_instance_point_ids
        if pid in points3D
    }

    if not point_xyz:
        print(
            "Warning: no valid 3D coordinates found for any instance; returning no edges."
        )
        return []

    # ------------------------------------------------------------------
    # Compute per-instance voxel occupancy sets.
    # ------------------------------------------------------------------
    per_instance_voxels: Dict[Node, Set[int]] = {}
    for node, point_ids in per_instance_sets.items():
        voxel_set: Set[int] = set()
        for pid in point_ids:
            if pid in point_xyz:
                voxel_set.add(_point_to_voxel(point_xyz[pid]))
        if voxel_set:
            per_instance_voxels[node] = voxel_set

    # Work only with nodes that have at least one valid voxel.
    valid_nodes = [n for n in nodes_list if n in per_instance_voxels]
    if not valid_nodes:
        return []

    # ------------------------------------------------------------------
    # Build sparse incidence matrix M  (instances × voxels).
    # ------------------------------------------------------------------
    all_voxels: Set[int] = set()
    total_assoc = 0
    for s in per_instance_voxels.values():
        all_voxels.update(s)
        total_assoc += len(s)

    voxel_to_idx = {v: i for i, v in enumerate(all_voxels)}
    num_nodes = len(valid_nodes)
    num_voxels = len(all_voxels)

    print(
        f"Constructing sparse voxel matrix "
        f"({num_nodes} instances × {num_voxels} occupied voxels)…"
    )

    row_inds = np.zeros(total_assoc, dtype=np.int32)
    col_inds = np.zeros(total_assoc, dtype=np.int32)
    cur = 0
    for ni, node in enumerate(valid_nodes):
        for v in per_instance_voxels[node]:
            row_inds[cur] = ni
            col_inds[cur] = voxel_to_idx[v]
            cur += 1

    data = np.ones(total_assoc, dtype=np.int32)
    M = sparse.csr_matrix((data, (row_inds, col_inds)), shape=(num_nodes, num_voxels))

    print("Computing voxel intersection matrix (M · Mᵀ)…")
    intersection_matrix = M.dot(M.T)

    set_sizes = np.array(M.sum(axis=1)).flatten()

    print("Filtering edges and calculating geometric Jaccard…")
    upper_tri = sparse.triu(intersection_matrix, k=1).tocoo()

    edges: List[Tuple[Node, Node, Dict[str, Any]]] = []
    for i, j, overlap in zip(upper_tri.row, upper_tri.col, upper_tri.data):
        node1 = valid_nodes[i]
        node2 = valid_nodes[j]

        union_size = set_sizes[i] + set_sizes[j] - overlap
        geo_jac = float(overlap / union_size) if union_size > 0 else 0.0

        if geo_jac >= tau:
            edges.append(
                (
                    node1,
                    node2,
                    {"voxel_overlap": int(overlap), "geometric_jaccard": geo_jac},
                )
            )

    return edges


# ---------------------------------------------------------------------------
# NetworkX conversion
# ---------------------------------------------------------------------------


def to_networkx(
    nodes: List[Node],
    edges: List[Tuple[Node, Node, Dict[str, Any]]],
    node_to_instance_id: Dict[Node, str],
) -> nx.Graph:
    G = nx.Graph()
    for node in nodes:
        seq_key, obj_slug, obj_id_str = node
        G.add_node(
            node,
            seq_key=seq_key,
            obj_slug=obj_slug,
            obj_id=obj_id_str,
            instance_id=node_to_instance_id.get(node, ""),
        )
    for node1, node2, attrs in edges:
        G.add_edge(node1, node2, **attrs)
    return G


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_object_mask_graph(
    dataset_name: str,
    K: int = 30,
    tau: float = 0.8,
    min_points: int = 1,
    min_points_in_3d_segment: int = 10,
    intersection_type: str = "geometric",
    voxel_size_cm: float = 10.0,
) -> None:
    """
    Build the object mask graph.

    Args:
        intersection_type: How to measure instance overlap.
            ``"geometric"``  – voxelise each instance's 3D point cloud and
                              compute Jaccard over the voxel-occupancy sets
                              (calls :func:`build_edges_geometric_intersection`).
            ``"id_based"``   – compute Jaccard directly over the raw COLMAP
                              point-ID sets (calls :func:`build_edges_scipy`).
        voxel_size_cm: Side length of each voxel in centimetres used when
            *intersection_type* is ``"geometric"``.
    """
    if intersection_type not in ("geometric", "id_based"):
        raise ValueError(
            f"intersection_type must be 'geometric' or 'id_based', got {intersection_type!r}"
        )

    config = load_config(dataset_name)
    outputs_dir = get_outputs_dir(config)
    obj_level_dir = outputs_dir / "object_level_masks"

    associations_path = obj_level_dir / "object_3d_associations.json"
    if not associations_path.is_file():
        raise FileNotFoundError(
            f"Associations file not found: {associations_path}\n"
            "Run associate2d3d.py first."
        )

    print(f"Loading associations from {associations_path}…")
    per_instance_sets, node_to_instance_id = load_associations(
        associations_path, min_points=min_points
    )

    total_nodes = len(per_instance_sets)
    total_pts = sum(len(s) for s in per_instance_sets.values())
    print(f"Loaded {total_nodes} instances, {total_pts} total point associations.")
    if total_nodes:
        print(f"Average points per instance: {total_pts / total_nodes:.1f}")

    # ----- Build edges ------------------------------------------------------
    print(
        f"\nBuilding edges (K={K}, tau={tau}, intersection_type={intersection_type!r})…"
    )
    if intersection_type == "geometric":
        from ..colmap_io import load_colmap_model
        from ..io_paths import get_colmap_model_dir

        colmap_model_dir = get_colmap_model_dir(config)
        print(
            f"Loading COLMAP model for geometric intersection from {colmap_model_dir}…"
        )
        _, _, points3D = load_colmap_model(str(colmap_model_dir))
        edges = build_edges_geometric_intersection(
            per_instance_sets,
            points3D=points3D,
            voxel_size_cm=voxel_size_cm,
            tau=tau,
        )
    else:
        edges = build_edges_scipy(per_instance_sets, K=K, tau=tau)
    print(f"Created {len(edges)} edges.")

    # ----- NetworkX graph ---------------------------------------------------
    print("\nCreating NetworkX graph…")
    nodes = list(per_instance_sets.keys())
    G = to_networkx(nodes, edges, node_to_instance_id)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    if G.number_of_edges() > 0:
        ccs = list(nx.connected_components(G))
        print(f"Connected components: {len(ccs)}")
        print(f"Largest component: {max(len(c) for c in ccs)} nodes")
        degrees = [d for _, d in G.degree()]
        print(f"Average degree: {sum(degrees) / len(degrees):.2f}  Max: {max(degrees)}")

    # ----- Save graph -------------------------------------------------------
    graph_path = outputs_dir / "mask_graph.gpickle"
    with graph_path.open("wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved graph → {graph_path}")

    # ----- Load existing component IDs for stable incremental assignment -----
    components_path = outputs_dir / "connected_components.json"
    # Map: frozenset(instance_ids) → existing connected_comp_id
    existing_id_map: Dict[frozenset, int] = {}
    max_existing_id: int = -1
    if components_path.exists():
        try:
            with components_path.open("r", encoding="utf-8") as fh:
                existing_list: List[Dict[str, Any]] = json.load(fh)
            for comp in existing_list:
                key = frozenset(comp["instance_ids"])
                existing_id_map[key] = comp["connected_comp_id"]
                max_existing_id = max(max_existing_id, comp["connected_comp_id"])
            if existing_id_map:
                print(
                    f"Loaded {len(existing_id_map)} existing component ID(s) "
                    f"(max id={max_existing_id}) for stable reassignment."
                )
        except Exception as exc:
            print(
                f"Warning: could not load existing components for ID reuse ({exc}); "
                "all IDs will be freshly assigned."
            )
            existing_id_map = {}
            max_existing_id = -1

    next_new_id: int = max_existing_id + 1

    # ----- Connected components with merged 3D points -----------------------
    print("\nExtracting connected components…")
    components_list = []

    if G.number_of_nodes() > 0:
        ccs = list(nx.connected_components(G))
        for component in ccs:
            point3d_union: Set[int] = set()
            instance_ids: List[str] = []
            for node in component:
                point3d_union.update(per_instance_sets[node])
                if node in node_to_instance_id:
                    instance_ids.append(node_to_instance_id[node])

            if len(point3d_union) < min_points_in_3d_segment:
                continue

            sorted_instances = sorted(instance_ids)
            new_key = frozenset(sorted_instances)

            # Exact match with existing component
            if new_key in existing_id_map:
                comp_id = existing_id_map[new_key]
            else:
                # Subset match: an existing component grew by absorbing new instances
                # (new_key is a superset of some existing component's instance set)
                parent_id = None
                for existing_key, existing_id in existing_id_map.items():
                    if existing_key.issubset(new_key):
                        parent_id = existing_id
                        break
                if parent_id is not None:
                    comp_id = parent_id
                else:
                    # Genuinely new component – assign a fresh ID
                    comp_id = next_new_id
                    next_new_id += 1

            components_list.append(
                {
                    "connected_comp_id": comp_id,
                    "instance_ids": sorted_instances,
                    "set_of_point3DIds": sorted(point3d_union),
                }
            )

        # Sort output by component ID for readability / stable ordering
        components_list.sort(key=lambda c: c["connected_comp_id"])

        new_count = sum(
            1
            for c in components_list
            if frozenset(c["instance_ids"]) not in existing_id_map
            and not any(
                ek.issubset(frozenset(c["instance_ids"])) for ek in existing_id_map
            )
        )
        print(
            f"Found {len(components_list)} component(s) with ≥ {min_points_in_3d_segment} points "
            f"({new_count} new, {len(components_list) - new_count} existing)."
        )
        if components_list:
            sizes = [len(c["set_of_point3DIds"]) for c in components_list]
            print(
                f"  Largest: {max(sizes)}  Smallest: {min(sizes)}  Avg: {sum(sizes)/len(sizes):.1f}"
            )

    with components_path.open("w", encoding="utf-8") as fh:
        json.dump(components_list, fh, indent=2)
    print(f"Saved connected components → {components_path}")

    # ----- Statistics -------------------------------------------------------
    stats: Dict[str, Any] = {
        "parameters": {
            "K": K,
            "tau": tau,
            "min_points": min_points,
            "min_points_in_3d_segment": min_points_in_3d_segment,
            "intersection_type": intersection_type,
            "voxel_size_cm": voxel_size_cm,
        },
        "nodes": {"total": total_nodes},
        "edges": {"total": G.number_of_edges()},
        "point_associations": {
            "total": total_pts,
            "average_per_instance": total_pts / total_nodes if total_nodes else 0,
        },
    }
    if G.number_of_edges() > 0:
        ccs = list(nx.connected_components(G))
        degrees = [d for _, d in G.degree()]
        stats["graph_statistics"] = {
            "connected_components": len(ccs),
            "largest_component_nodes": max(len(c) for c in ccs),
            "average_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }

    stats_path = outputs_dir / "mask_graph_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved statistics    → {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mask graph from per-object SAM3 3D associations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--K", type=int, default=30, help="Min overlap count threshold."
    )
    parser.add_argument("--tau", type=float, default=0.8, help="Min Jaccard threshold.")
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Min 3D points for a node to be included.",
    )
    parser.add_argument(
        "--min-points-segment",
        type=int,
        default=10,
        help="Min 3D points in a connected component to be reported.",
    )
    parser.add_argument(
        "--intersection-type",
        choices=["geometric", "id_based"],
        default="geometric",
        help="How to measure instance overlap: 'geometric' (voxel Jaccard) or 'id_based' (point-ID Jaccard).",
    )
    parser.add_argument(
        "--voxel-size-cm",
        type=float,
        default=10.0,
        help="Voxel side length in centimetres (used when --intersection-type=geometric). "
        "Point coordinates are assumed to be in metres.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_object_mask_graph(
        dataset_name=args.dataset,
        K=args.K,
        tau=args.tau,
        min_points=args.min_points,
        min_points_in_3d_segment=args.min_points_segment,
        intersection_type=args.intersection_type,
        voxel_size_cm=args.voxel_size_cm,
    )
