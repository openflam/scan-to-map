"""
polycam.py – Point cloud extraction and mesh sampling utilities.

Provides two main functions:
  - extract_colmap_pointcloud: Loads points3D from a COLMAP sparse model and
    filters by track length.
  - sample_glb_mesh: Loads a .glb mesh with trimesh and samples it using
    Poisson Disk Sampling (falls back to uniform sampling).

When run as a script, both point clouds are prepared for the dataset supplied
via --dataset (defaults to "cfa_test_neg"), printed with spatial diagnostics,
and saved as .pcd files in temp_output/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

# ---------------------------------------------------------------------------
# Path plumbing: reach read_write_model.py that lives in segment3d/src/utils/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "segment3d" / "src" / "utils"))
sys.path.insert(0, str(_REPO_ROOT / "segment3d"))

from read_write_model import (
    read_model,
    write_model,
    qvec2rotmat,
    rotmat2qvec,
)  # noqa: E402
from config import get_config  # noqa: E402


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def extract_colmap_pointcloud(
    colmap_dir: str | Path,
    min_track_length: int = 3,
) -> o3d.geometry.PointCloud:
    """Load and filter the sparse point cloud from a COLMAP workspace.

    Args:
        colmap_dir: Path to the COLMAP workspace containing ``sparse/0/``
                    (with ``.bin`` or ``.txt`` files).
        min_track_length: Discard points seen by fewer than this many images.
                          Default is 3.

    Returns:
        An ``open3d.geometry.PointCloud`` with the filtered 3-D coordinates
        (no colour, because COLMAP colours are unreliable at this level).
    """
    colmap_dir = Path(colmap_dir)
    # Accept either a direct model directory (containing cameras.bin/.txt) or
    # a COLMAP workspace root (which has sparse/0/ underneath).
    if (colmap_dir / "cameras.bin").is_file() or (colmap_dir / "cameras.txt").is_file():
        sparse_dir = colmap_dir
    else:
        sparse_dir = colmap_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise NotADirectoryError(f"Expected COLMAP sparse model in: {sparse_dir}")

    _, _, points3D = read_model(path=str(sparse_dir), ext="")

    xyz_list: list[np.ndarray] = []
    rgb_list: list[np.ndarray] = []
    for pt in points3D.values():
        if len(pt.image_ids) >= min_track_length:
            xyz_list.append(pt.xyz)
            rgb_list.append(pt.rgb / 255.0)

    if not xyz_list:
        raise ValueError(
            f"No points survived the track-length filter "
            f"(min_track_length={min_track_length})."
        )

    xyz = np.array(xyz_list, dtype=np.float64)
    rgb = np.array(rgb_list, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print(
        f"[COLMAP] Loaded {len(points3D)} raw points → "
        f"{len(xyz_list)} after track-length ≥ {min_track_length} filter."
    )
    return pcd


def sample_glb_mesh(
    mesh_path: str | Path,
    num_points: int = 50000,
) -> o3d.geometry.PointCloud:
    """Load a .glb scene, merge all geometries, and sample a point cloud.

    Poisson Disk Sampling is attempted first; if it is unavailable (older
    trimesh versions), uniform surface sampling is used instead.

    Args:
        mesh_path: Path to the ``.glb`` file.
        num_points: Number of surface samples to generate.  Default 100 000.

    Returns:
        An ``open3d.geometry.PointCloud`` with *num_points* points sampled
        from the mesh surface.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"GLB file not found: {mesh_path}")

    scene_or_mesh = trimesh.load(str(mesh_path), force="scene")

    # Merge all sub-geometries into one mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = list(scene_or_mesh.geometry.values())
        if not geometries:
            raise ValueError("The GLB scene contains no geometry.")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = scene_or_mesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh after merging, got {type(mesh).__name__}.")

    # Poisson Disk Sampling (preferred) – available in trimesh >= 3.x
    try:
        points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
        method_used = "Poisson Disk (even surface)"
    except Exception:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        method_used = "Uniform Surface"

    # --- Coordinate Transformation (Y-up to Z-up) ---
    # glTF (points) -> x=x, y=y, z=z
    # Blender (target) -> x=x, y=-z, z=y

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Re-stacking the columns to perform a -90 degree rotation on the X-axis
    corrected_points = np.column_stack((x, -z, y))

    print(
        f"[GLB]   Loaded '{mesh_path.name}' "
        f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces) → "
        f"sampled {len(points):,} points via {method_used}."
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array(corrected_points, dtype=np.float64)
    )
    return pcd


# ---------------------------------------------------------------------------
# GLB mesh transformation
# ---------------------------------------------------------------------------


def transform_glb(
    mesh_path: str | Path,
    T_global: np.ndarray,
    output_path: str | Path,
) -> trimesh.Trimesh:
    """
    Applies T_global to the mesh while maintaining compatibility with
    Blender's glTF importer.
    """
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)

    # 1. Load and merge (consistent with sampling logic)
    scene_or_mesh = trimesh.load(str(mesh_path), force="scene")
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh

    # 2. Define the Y-up to Z-up basis change matrix (R_fix)
    # This is the same rotation used in your sampling function
    T_fix = np.eye(4)
    T_fix[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    # 3. Calculate the Blender-compatible transform
    # Logic:
    #   a. T_fix: Move raw GLB (Y-up) to Z-up space
    #   b. T_global: Move Z-up mesh to COLMAP space
    #   c. inv(T_fix): Move COLMAP space back to Y-up for GLB export
    T_fix_inv = np.linalg.inv(T_fix)
    T_blender_ready = T_fix_inv @ T_global @ T_fix

    # 4. Apply and Export
    mesh.apply_transform(T_blender_ready)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    print(f"[GLB] Exported Blender-aligned mesh to: {output_path}")
    return mesh


# ---------------------------------------------------------------------------
# Global coarse registration
# ---------------------------------------------------------------------------


def _prepare_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Downsample, estimate normals, and compute FPFH features."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(k=15)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return down, fpfh


def _ransac_once(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    dist_thresh: float,
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                dist_thresh
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=5_000_000,
            confidence=0.99999,
        ),
    )


def align_point_clouds(
    colmap_pc: o3d.geometry.PointCloud,
    sampled_pc: o3d.geometry.PointCloud,
    voxel_size: float | None = None,
    num_ransac_restarts: int = 5,
) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """Coarse global registration of *sampled_pc* onto *colmap_pc*.

    Strategy:
      1. Centroid pre-alignment to reduce the rotational search space.
      2. Multi-scale RANSAC: run at three voxel sizes (coarse → fine) and keep
         the best result across *num_ransac_restarts* independent runs per scale.
      3. ICP point-to-plane refinement on the full-resolution clouds.

    Args:
        colmap_pc:            Target (COLMAP sparse cloud).
        sampled_pc:           Source (GLB mesh sample).
        voxel_size:           Base voxel size. Auto-selected as 2 % of the
                              target AABB diagonal when *None*.
        num_ransac_restarts:  Independent RANSAC runs per voxel scale.

    Returns:
        ``(T_sampled_to_colmap, T_colmap_to_sampled, aligned_pc)`` –
        * ``T_sampled_to_colmap``: 4×4 rigid transform that maps *sampled_pc*
          coordinates into the COLMAP world frame.
        * ``T_colmap_to_sampled``: its inverse – maps COLMAP world coordinates
          into the sampled (GLB mesh) frame.
        * ``aligned_pc``: *sampled_pc* after applying ``T_sampled_to_colmap``.
    """
    reg = o3d.pipelines.registration  # convenience alias

    # ------------------------------------------------------------------
    # 0. Statistical outlier removal on the (noisy) COLMAP cloud
    # ------------------------------------------------------------------
    target, _ = colmap_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    source = o3d.geometry.PointCloud(sampled_pc)

    # ------------------------------------------------------------------
    # 1. Centroid pre-alignment  (translate source → target centroid)
    # ------------------------------------------------------------------
    t_centroid = np.asarray(target.points).mean(axis=0) - np.asarray(
        source.points
    ).mean(axis=0)
    T_centroid = np.eye(4)
    T_centroid[:3, 3] = t_centroid
    source.transform(T_centroid)
    print(f"[align] Centroid shift applied: {t_centroid}")

    # ------------------------------------------------------------------
    # 2. Auto voxel size
    # ------------------------------------------------------------------
    if voxel_size is None:
        aabb_diag = float(
            np.linalg.norm(
                np.asarray(target.points).max(axis=0)
                - np.asarray(target.points).min(axis=0)
            )
        )
        voxel_size = aabb_diag * 0.02
        print(
            f"[align] Auto voxel_size = {voxel_size:.4f}  (2 % of target AABB diagonal {aabb_diag:.4f})"
        )

    # ------------------------------------------------------------------
    # 3. Multi-scale RANSAC (coarse → fine)
    # ------------------------------------------------------------------
    scales = [voxel_size * 4, voxel_size * 2, voxel_size]
    best_result = None
    best_T_accum = np.eye(4)  # accumulated from centroid shift onward
    T_accum = np.eye(4)  # tracks transforms applied to `source` so far

    for scale_idx, scale in enumerate(scales):
        print(f"\n[align] Scale {scale_idx + 1}/{len(scales)}  voxel = {scale:.4f}")
        target_down, target_fpfh = _prepare_fpfh(target, scale)
        source_down, source_fpfh = _prepare_fpfh(source, scale)
        print(
            f"[align]   target {len(target_down.points):,} pts | source {len(source_down.points):,} pts"
        )

        dist_thresh = scale * 1.5
        scale_best: o3d.pipelines.registration.RegistrationResult | None = None

        for restart in range(num_ransac_restarts):
            r = _ransac_once(
                source_down, target_down, source_fpfh, target_fpfh, dist_thresh
            )
            print(
                f"[align]   restart {restart + 1}: fitness={r.fitness:.4f}  rmse={r.inlier_rmse:.4f}"
            )
            if scale_best is None or r.fitness > scale_best.fitness:
                scale_best = r

        if scale_best is not None and (
            best_result is None or scale_best.fitness > best_result.fitness
        ):
            best_result = scale_best
            best_T_accum = scale_best.transformation @ T_accum

        # Refine source with best transform at this scale before going finer
        if scale_best is not None and scale_best.fitness > 0.0:
            source.transform(scale_best.transformation)
            T_accum = scale_best.transformation @ T_accum

        print(f"[align]   best at this scale: fitness={scale_best.fitness:.4f}")

    if best_result is None or best_result.fitness == 0.0:
        print(
            "[align] WARNING: RANSAC produced no valid correspondence – returning centroid alignment only."
        )
        T_ransac = np.eye(4)
    else:
        T_ransac = (
            best_T_accum  # relative to centroid-pre-aligned frame, composed correctly
        )

    # Full transform: centroid shift first, then RANSAC
    T_coarse = T_ransac @ T_centroid
    print(
        f"\n[align] RANSAC best  fitness={best_result.fitness:.6f}  rmse={best_result.inlier_rmse:.6f}"
        if best_result
        else ""
    )

    # ------------------------------------------------------------------
    # 4. ICP point-to-plane refinement on full clouds
    # ------------------------------------------------------------------
    # Prepare normals for full target cloud (needed for point-to-plane)
    target_full = o3d.geometry.PointCloud(target)
    target_full.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
    )

    icp_thresh = voxel_size * 2.0
    icp_result = reg.registration_icp(
        source=o3d.geometry.PointCloud(sampled_pc).transform(T_coarse),
        target=target_full,
        max_correspondence_distance=icp_thresh,
        init=np.eye(4),
        estimation_method=reg.TransformationEstimationPointToPlane(),
        criteria=reg.ICPConvergenceCriteria(max_iteration=200),
    )
    T_icp = icp_result.transformation
    T_global = T_icp @ T_coarse

    print(
        f"[align] ICP refinement  fitness={icp_result.fitness:.6f}  rmse={icp_result.inlier_rmse:.6f}"
    )

    # ------------------------------------------------------------------
    # 5. Apply final transform to the full source
    # ------------------------------------------------------------------
    aligned_pc = o3d.geometry.PointCloud(sampled_pc)
    aligned_pc.transform(T_global)

    T_colmap_to_sampled = np.linalg.inv(T_global)
    return T_global, T_colmap_to_sampled, aligned_pc


# ---------------------------------------------------------------------------
# COLMAP model transformation
# ---------------------------------------------------------------------------


def transform_colmap_model(
    colmap_dir: str | Path,
    T: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Apply a rigid transform to a COLMAP sparse model and write the result.

    Transforms every 3-D point and every camera pose by *T*.  Pass
    ``T_colmap_to_gltf`` (= ``T_fix_inv @ T_colmap_to_sampled``) to produce a
    model whose coordinates are in the glTF Y-up frame so that it aligns with
    the raw GLB when both are loaded in Blender.

    Args:
        colmap_dir: Path to the COLMAP sparse model (contains cameras.bin / .txt,
                    images.bin / .txt, points3D.bin / .txt).  Accepts either the
                    sparse-model directory directly or a COLMAP workspace root
                    whose ``sparse/0/`` subdirectory holds the model.
        T: 4×4 rigid transform to apply to every point and camera pose.
        output_dir: Directory where the transformed model will be written.
                    Created automatically if it does not exist.
    """
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)

    # Accept workspace root or direct model directory
    if (colmap_dir / "cameras.bin").is_file() or (colmap_dir / "cameras.txt").is_file():
        sparse_dir = colmap_dir
    else:
        sparse_dir = colmap_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise NotADirectoryError(f"Expected COLMAP sparse model in: {sparse_dir}")

    cameras, images, points3D = read_model(path=str(sparse_dir), ext="")

    R_c2s = T[:3, :3]
    t_c2s = T[:3, 3]

    # --- Transform 3-D points -------------------------------------------------
    new_points3D = {}
    for pid, pt in points3D.items():
        new_xyz = R_c2s @ pt.xyz + t_c2s
        new_points3D[pid] = pt._replace(xyz=new_xyz)

    # --- Transform camera poses -----------------------------------------------
    # COLMAP convention:  p_camera = R_img * p_world + t_img
    # After applying T_colmap_to_sampled to the world:
    #   p_world = R_c2s^T * (p_sampled - t_c2s)
    # So the new pose that satisfies p_camera = R_new * p_sampled + t_new is:
    #   R_new = R_img * R_c2s^T
    #   t_new = t_img - R_new * t_c2s
    new_images = {}
    for iid, img in images.items():
        R_img = qvec2rotmat(img.qvec)
        R_new = R_img @ R_c2s.T
        t_new = img.tvec - R_new @ t_c2s
        new_qvec = rotmat2qvec(R_new)
        new_images[iid] = img._replace(qvec=new_qvec, tvec=t_new)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_model(cameras, new_images, new_points3D, path=str(output_dir), ext=".bin")
    print(f"[COLMAP] Transformed model written to: {output_dir}")


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------


def print_pcd_stats(name: str, pcd: o3d.geometry.PointCloud) -> None:
    """Print AABB and centroid for *pcd*."""
    pts = np.asarray(pcd.points)
    aabb_min = pts.min(axis=0)
    aabb_max = pts.max(axis=0)
    centroid = pts.mean(axis=0)

    print(f"\n{'─' * 50}")
    print(f"  {name}  ({len(pts):,} points)")
    print(f"  AABB min : {aabb_min}")
    print(f"  AABB max : {aabb_max}")
    print(f"  AABB size: {aabb_max - aabb_min}")
    print(f"  Centroid : {centroid}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract COLMAP point cloud and sample GLB mesh."
    )
    parser.add_argument(
        "--dataset",
        default="cfa_test_neg",
        help="Dataset name (must match a folder under data/). " "Default: cfa_test_neg",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="Number of surface samples for GLB mesh. Default: 50000",
    )
    parser.add_argument(
        "--min_track_length",
        type=int,
        default=3,
        help="Minimum track length for COLMAP point filtering. Default: 3",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save .pcd files. Default: data/<dataset>/alignment/",
    )
    args = parser.parse_args()

    # Resolve paths via config
    cfg = get_config(args.dataset)
    data_dir = _REPO_ROOT / "data" / args.dataset
    colmap_dir = _REPO_ROOT / "data" / args.dataset / "hloc_data" / "sfm_reconstruction"
    mesh_path = data_dir / "polycam_data" / "raw.glb"

    print(f"Dataset     : {args.dataset}")
    print(f"COLMAP dir  : {colmap_dir}")
    print(f"Mesh path   : {mesh_path}")

    # --- COLMAP point cloud ---------------------------------------------------
    colmap_pc = extract_colmap_pointcloud(
        colmap_dir=colmap_dir,
        min_track_length=args.min_track_length,
    )
    print_pcd_stats("COLMAP point cloud", colmap_pc)

    # --- GLB mesh sampling ----------------------------------------------------
    sampled_pc = sample_glb_mesh(
        mesh_path=mesh_path,
        num_points=args.num_points,
    )
    print_pcd_stats("GLB sampled point cloud", sampled_pc)

    # --- Coarse global registration -------------------------------------------
    T_global, T_colmap_to_sampled, aligned_pc = align_point_clouds(
        colmap_pc, sampled_pc
    )
    print(f"\n[align] Final T_sampled_to_colmap:\n{T_global}")
    print(f"\n[align] Final T_colmap_to_sampled:\n{T_colmap_to_sampled}")
    print_pcd_stats("Globally aligned GLB point cloud", aligned_pc)

    # --- Save to disk ----------------------------------------------------------
    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    colmap_out = out_dir / "colmap.pcd"
    sampled_out = out_dir / "glb_sampled.pcd"
    aligned_out = out_dir / "global_aligned_glb_sampled.pcd"
    transformed_glb_out = out_dir / "transformed.glb"

    o3d.io.write_point_cloud(str(colmap_out), colmap_pc)
    o3d.io.write_point_cloud(str(sampled_out), sampled_pc)
    o3d.io.write_point_cloud(str(aligned_out), aligned_pc)
    transform_glb(mesh_path, T_global, transformed_glb_out)

    # --- Transform COLMAP model into the sampled (Z-up) frame ----------------
    # When Blender imports the raw GLB it auto-rotates Y-up → Z-up, so the
    # scene lands in the same Z-up "sampled" frame that sample_glb_mesh uses.
    # COLMAP models loaded via a Blender addon are placed directly in Blender's
    # Z-up world space with no extra conversion.  Therefore storing the
    # transformed COLMAP model in T_colmap_to_sampled (Z-up) is the correct
    # target frame — no additional T_fix_inv step is needed.
    #
    # Additionally apply a +90° rotation about Z to align with the raw GLB in
    # Blender (hardcoded correction).
    T_z_pos90 = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    T_colmap_to_sampled_rotated = T_z_pos90 @ T_colmap_to_sampled

    colmap_transformed_out = out_dir / "sfm_reconstruction_transformed"
    transform_colmap_model(
        colmap_dir, T_colmap_to_sampled_rotated, colmap_transformed_out
    )

    # --- Apply an additional -90° rotation about Y to the transformed GLB -----
    transformed_rotated_glb_out = out_dir / "transformed_rotated.glb"
    _rotated = trimesh.load(str(transformed_glb_out), force="mesh")
    T_y_neg90 = np.array(
        [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    _rotated.apply_transform(T_y_neg90)
    _rotated.export(str(transformed_rotated_glb_out))
    print(f"Saved -90° Y-rotated GLB mesh    → {transformed_rotated_glb_out}")

    print(f"\nSaved COLMAP point cloud         → {colmap_out}")
    print(f"Saved sampled point cloud        → {sampled_out}")
    print(f"Saved globally aligned cloud     → {aligned_out}")
    print(f"Saved transformed GLB mesh       → {transformed_glb_out}")
    print(f"Saved transformed COLMAP model   → {colmap_transformed_out}")


if __name__ == "__main__":
    main()
