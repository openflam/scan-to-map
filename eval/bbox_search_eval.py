"""
Bounding-box retrieval evaluation for scan-to-map × ScanQA.

Tests the *search ability* of the pipeline: given a ScanQA question, does the
CLIP-based component search return a component whose 3D bounding box overlaps
the ground-truth annotated object(s)?

For every question the script:
  1. Encodes the question text with OpenCLIP.
  2. Retrieves the top-K scan-to-map components from the per-scene FAISS index.
  3. Looks up the 3D AABB for each retrieved component (bbox_corners.json).
  4. Builds a ground-truth AABB from the ScanNet instance annotations
     (.aggregation.json + _vh_clean_2.0.010000.segs.json + _vh_clean_2.ply).
  5. Computes 3D IoU and centre-to-centre distance between each retrieved
     bbox and the GT bbox.
  6. Reports aggregate metrics:
       - Recall@K for IoU ≥ 0.25 and IoU ≥ 0.5  (K = 1, 5, 10)
       - Mean top-1 IoU and mean top-1 centre distance

Required directory layout
--------------------------
scan-to-map outputs (--outputs-root):
    <outputs_root>/scannet_<scene_id>/
        clip_embeddings.faiss
        clip_embeddings.npz
        bbox_corners.json

ScanNet raw data (--scannet-root):
    <scannet_root>/scans/<scene_id>/
        <scene_id>.aggregation.json
        <scene_id>_vh_clean_2.0.010000.segs.json
        <scene_id>_vh_clean_2.ply

ScanQA data (--scanqa-root):
    <scanqa_root>/ScanQA_v1.0_val.json   (or test_w_obj / test_wo_obj)

Usage
-----
    python eval/bbox_search_eval.py \\
        --scanqa-root   /path/to/ScanQA/data/qa \\
        --outputs-root  /path/to/scan-to-map/outputs \\
        --scannet-root  /path/to/scannet/data \\
        [--split val] \\
        [--top-k 10] \\
        [--iou-thresh 0.25] \\
        [--clip-model ViT-H-14] \\
        [--clip-pretrained laion2B-s32B-b79K] \\
        [--device 0] \\
        [--output-dir eval_results/] \\
        [--save-per-question]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 3-D AABB helpers
# ---------------------------------------------------------------------------

def aabb_iou(min1: np.ndarray, max1: np.ndarray,
             min2: np.ndarray, max2: np.ndarray) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes in 3D."""
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    diff = inter_max - inter_min
    if np.any(diff <= 0):
        return 0.0
    inter_vol = float(np.prod(diff))
    vol1 = float(np.prod(np.maximum(max1 - min1, 0)))
    vol2 = float(np.prod(np.maximum(max2 - min2, 0)))
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def center_distance(min1: np.ndarray, max1: np.ndarray,
                    min2: np.ndarray, max2: np.ndarray) -> float:
    """Euclidean distance between the centres of two AABBs."""
    c1 = (min1 + max1) / 2.0
    c2 = (min2 + max2) / 2.0
    return float(np.linalg.norm(c1 - c2))


# ---------------------------------------------------------------------------
# Ground-truth bbox from ScanNet instance annotations
# ---------------------------------------------------------------------------

def build_gt_bbox(
    scannet_scene_dir: Path,
    scene_id: str,
    object_ids: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the AABB of all mesh vertices belonging to the given ScanNet
    object IDs.

    Requires:
        <scene_dir>/<scene_id>.aggregation.json
        <scene_dir>/<scene_id>_vh_clean_2.0.010000.segs.json
        <scene_dir>/<scene_id>_vh_clean_2.ply

    Returns (bbox_min, bbox_max) as float32 (3,) arrays, or None if any
    required file is missing or no matching vertices are found.
    """
    if not object_ids:
        return None

    agg_path  = scannet_scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scannet_scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    ply_path  = scannet_scene_dir / f"{scene_id}_vh_clean_2.ply"

    for p in (agg_path, segs_path, ply_path):
        if not p.exists():
            return None

    # ---- aggregation: objectId → list of segmentIds -------------------------
    with open(agg_path) as f:
        agg = json.load(f)

    target_segs: Set[int] = set()
    for seg_group in agg.get("segGroups", []):
        if seg_group["objectId"] in object_ids:
            target_segs.update(seg_group["segments"])

    if not target_segs:
        return None

    # ---- segmentation: vertex index → segIndex ------------------------------
    with open(segs_path) as f:
        segs_data = json.load(f)

    seg_indices: List[int] = segs_data["segIndices"]   # one entry per vertex

    # ---- vertex positions from PLY (numpy-based, no open3d required) --------
    verts = _read_ply_vertices(ply_path)
    if verts is None or len(verts) == 0:
        return None

    # Mask vertices that belong to any of the target segments
    seg_arr = np.array(seg_indices, dtype=np.int32)
    mask = np.zeros(len(seg_arr), dtype=bool)
    for sid in target_segs:
        mask |= (seg_arr == sid)

    if not mask.any():
        return None

    pts = verts[mask]
    return pts.min(axis=0).astype(np.float32), pts.max(axis=0).astype(np.float32)


def _read_ply_vertices(ply_path: Path) -> Optional[np.ndarray]:
    """
    Read x,y,z vertex positions from an ASCII or binary PLY file.
    Uses open3d if available, otherwise falls back to a pure-Python parser.
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        return verts if len(verts) > 0 else None
    except ImportError:
        pass

    # Fallback: minimal pure-Python PLY reader (handles binary_little_endian
    # and ascii; reads only x,y,z from the vertex element).
    try:
        return _ply_read_xyz_fallback(ply_path)
    except Exception:
        return None


def _ply_read_xyz_fallback(ply_path: Path) -> Optional[np.ndarray]:
    """Minimal PLY reader for vertex xyz (no open3d dependency)."""
    import struct

    with open(ply_path, "rb") as fh:
        # Parse header
        header_lines: List[bytes] = []
        while True:
            line = fh.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        header_text = b"".join(header_lines).decode("ascii", errors="replace")
        lines = header_text.splitlines()

        fmt = "ascii"
        for l in lines:
            if l.startswith("format"):
                fmt = l.split()[1]

        n_verts = 0
        for l in lines:
            if l.startswith("element vertex"):
                n_verts = int(l.split()[-1])

        # Find x,y,z property offsets in the vertex block
        # We only support float/double for x,y,z
        prop_types: List[Tuple[str, str]] = []
        in_vertex = False
        for l in lines:
            if l.startswith("element vertex"):
                in_vertex = True
            elif l.startswith("element") and not l.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and l.startswith("property"):
                parts = l.split()
                prop_types.append((parts[1], parts[2]))  # (type, name)

        type_sizes = {
            "float": 4, "float32": 4, "double": 8, "float64": 8,
            "int": 4, "int32": 4, "uint": 4, "uint32": 4,
            "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
            "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
        }
        type_fmts = {
            "float": "f", "float32": "f", "double": "d", "float64": "d",
            "int": "i", "int32": "i", "uint": "I", "uint32": "I",
            "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
            "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
        }

        xyz_indices = {name: i for i, (_, name) in enumerate(prop_types)
                       if name in ("x", "y", "z")}
        if len(xyz_indices) < 3:
            return None

        row_size = sum(type_sizes.get(t, 4) for t, _ in prop_types)

        verts = np.empty((n_verts, 3), dtype=np.float32)

        if fmt == "ascii":
            for i in range(n_verts):
                row = fh.readline().split()
                xi = xyz_indices["x"]; yi = xyz_indices["y"]; zi = xyz_indices["z"]
                verts[i] = [float(row[xi]), float(row[yi]), float(row[zi])]
        else:
            endian = "<" if "little" in fmt else ">"
            row_fmt = endian + "".join(type_fmts.get(t, "f") for t, _ in prop_types)
            unpacker = struct.Struct(row_fmt)
            xi = xyz_indices["x"]; yi = xyz_indices["y"]; zi = xyz_indices["z"]
            for i in range(n_verts):
                row_bytes = fh.read(row_size)
                vals = unpacker.unpack(row_bytes)
                verts[i] = [vals[xi], vals[yi], vals[zi]]

    return verts


# ---------------------------------------------------------------------------
# scan-to-map artifact loading
# ---------------------------------------------------------------------------

def load_scene_outputs(
    scene_outputs_dir: Path,
) -> Optional[Tuple[object, List[str], Dict[str, Tuple]]]:
    """
    Load FAISS index, component_ids, and per-component bboxes for one scene.

    Returns (faiss_index, component_ids, bbox_dict) where
        bbox_dict[comp_id] = (bbox_min, bbox_max)  as float32 (3,) arrays
    or None if required files are missing.
    """
    required = ["clip_embeddings.faiss", "clip_embeddings.npz", "bbox_corners.json"]
    if any(not (scene_outputs_dir / r).exists() for r in required):
        missing = [r for r in required if not (scene_outputs_dir / r).exists()]
        print(f"  [SKIP] {scene_outputs_dir.name}: missing {missing}")
        return None

    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu is required.  pip install faiss-cpu")

    index = faiss.read_index(str(scene_outputs_dir / "clip_embeddings.faiss"))

    npz = np.load(str(scene_outputs_dir / "clip_embeddings.npz"), allow_pickle=True)
    component_ids = [str(c) for c in npz["component_ids"].tolist()]

    with open(scene_outputs_dir / "bbox_corners.json") as f:
        bbox_data = json.load(f)

    bbox_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for entry in bbox_data:
        cid  = str(entry["connected_comp_id"])
        bmin = np.array(entry["bbox"]["min"], dtype=np.float32)
        bmax = np.array(entry["bbox"]["max"], dtype=np.float32)
        bbox_dict[cid] = (bmin, bmax)

    return index, component_ids, bbox_dict


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

def load_clip_model(model_name: str, pretrained: str, device_str: str):
    try:
        import open_clip
    except ImportError:
        raise ImportError("open-clip-torch is required.  pip install open-clip-torch")
    import torch
    print(f"Loading CLIP model {model_name} ({pretrained}) on {device_str} …")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device_str).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def encode_texts(questions: List[str], model, tokenizer, device_str: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        tokens = tokenizer(questions).to(device_str)
        feats  = model.encode_text(tokens)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def search_top_k_components(
    query_vec: np.ndarray,          # (1, D) float32
    faiss_index,
    component_ids: List[str],
    top_k: int,
) -> List[str]:
    """Return the top-k component IDs (as strings) from a FAISS search."""
    k = min(top_k, faiss_index.ntotal)
    _, indices = faiss_index.search(query_vec, k)
    result = []
    for idx in indices[0]:
        if 0 <= idx < len(component_ids):
            result.append(component_ids[idx])
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_bbox_eval(
    scanqa_root: str,
    outputs_root: str,
    scannet_root: str,
    split: str = "val",
    top_k: int = 10,
    iou_thresh: float = 0.25,
    clip_model: str = "ViT-H-14",
    clip_pretrained: str = "laion2B-s32B-b79K",
    device: Optional[int] = None,
    output_dir: str = "eval_results",
    save_per_question: bool = False,
) -> dict:
    import torch

    device_str = f"cuda:{device}" if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device_str}")

    scanqa_path  = Path(scanqa_root)
    outputs_path = Path(outputs_root)
    scannet_path = Path(scannet_root)
    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ScanQA questions -----------------------------------------------
    qa_file = scanqa_path / f"ScanQA_v1.0_{split}.json"
    if not qa_file.exists():
        raise FileNotFoundError(f"ScanQA data not found: {qa_file}")

    with open(qa_file) as f:
        qa_data: List[dict] = json.load(f)

    # Filter to questions that actually have object_id annotations
    qa_data = [q for q in qa_data if q.get("object_ids")]
    print(f"Loaded {len(qa_data)} annotated questions from {qa_file.name}")

    by_scene: Dict[str, list] = {}
    for item in qa_data:
        by_scene.setdefault(item["scene_id"], []).append(item)
    print(f"Questions span {len(by_scene)} scenes")

    # ---- find scenes with scan-to-map outputs --------------------------------
    def scene_to_dataset(sid: str) -> str:
        return f"scannet_{sid}"

    available = {d.name for d in outputs_path.iterdir() if d.is_dir()} \
        if outputs_path.exists() else set()
    scenes_to_eval = {sid for sid in by_scene if scene_to_dataset(sid) in available}
    skipped = set(by_scene) - scenes_to_eval
    if skipped:
        print(f"Warning: {len(skipped)} scenes have no scan-to-map outputs and will be skipped.")
    print(f"Evaluating {len(scenes_to_eval)} scenes "
          f"({sum(len(by_scene[s]) for s in scenes_to_eval)} questions)\n")

    if not scenes_to_eval:
        raise RuntimeError(
            "No scenes with outputs found. Run the pipeline first:\n"
            "  cd segment3d/ && python main.py --dataset scannet_<scene_id>"
        )

    # ---- load CLIP model once -----------------------------------------------
    clip_mdl, tokenizer = load_clip_model(clip_model, clip_pretrained, device_str)

    # ---- per-K recall accumulators ------------------------------------------
    K_vals = [k for k in [1, 5, 10] if k <= top_k]
    recalls: Dict[Tuple[int, float], List[float]] = {
        (k, t): [] for k in K_vals for t in [0.25, 0.5]
    }
    top1_ious:   List[float] = []
    top1_dists:  List[float] = []
    gt_missing_count = 0

    per_question_results: List[dict] = []

    for scene_id in sorted(scenes_to_eval):
        dataset_name = scene_to_dataset(scene_id)
        scene_dir    = outputs_path / dataset_name
        scannet_dir  = scannet_path / "scans" / scene_id
        questions    = by_scene[scene_id]

        print(f"  Scene {scene_id}  ({len(questions)} questions)", end="", flush=True)

        artifacts = load_scene_outputs(scene_dir)
        if artifacts is None:
            print()
            continue
        faiss_index, component_ids, bbox_dict = artifacts

        # Batch-encode questions
        q_texts = [q["question"] for q in questions]
        q_vecs  = encode_texts(q_texts, clip_mdl, tokenizer, device_str)

        scene_gt_missing = 0
        for q_item, q_vec in zip(questions, q_vecs):
            # ---- ground truth bbox -------------------------------------------
            gt = build_gt_bbox(scannet_dir, scene_id, q_item["object_ids"])
            if gt is None:
                scene_gt_missing += 1
                gt_missing_count += 1
                continue
            gt_min, gt_max = gt

            # ---- top-K retrieval --------------------------------------------
            top_k_ids = search_top_k_components(
                q_vec[np.newaxis, :].astype(np.float32),
                faiss_index, component_ids, top_k,
            )

            # ---- per-component IoU ------------------------------------------
            comp_results: List[dict] = []
            for cid in top_k_ids:
                bbox = bbox_dict.get(cid)
                if bbox is None:
                    iou  = 0.0
                    dist = float("inf")
                else:
                    iou  = aabb_iou(gt_min, gt_max, bbox[0], bbox[1])
                    dist = center_distance(gt_min, gt_max, bbox[0], bbox[1])
                comp_results.append({"comp_id": cid, "iou": iou, "dist": dist})

            # ---- top-1 metrics ----------------------------------------------
            if comp_results:
                top1_ious.append(comp_results[0]["iou"])
                top1_dists.append(comp_results[0]["dist"])

            # ---- recall@K@thresh -------------------------------------------
            for k in K_vals:
                best_iou_at_k = max(
                    (r["iou"] for r in comp_results[:k]),
                    default=0.0,
                )
                for thresh in [0.25, 0.5]:
                    recalls[(k, thresh)].append(float(best_iou_at_k >= thresh))

            if save_per_question:
                per_question_results.append({
                    "question_id": q_item["question_id"],
                    "scene_id":    scene_id,
                    "question":    q_item["question"],
                    "object_ids":  q_item["object_ids"],
                    "object_names": q_item.get("object_names", []),
                    "gt_bbox":     {"min": gt_min.tolist(), "max": gt_max.tolist()},
                    "top_k_results": comp_results,
                })

        miss_str = f"  ({scene_gt_missing} GT missing)" if scene_gt_missing else ""
        print(f"  ✓{miss_str}")

    # ---- aggregate metrics ---------------------------------------------------
    n_evaluated = len(top1_ious)
    print(f"\n{'='*60}")
    print(f"Evaluated {n_evaluated} questions  "
          f"(GT missing / no label files: {gt_missing_count})")
    print(f"{'='*60}")

    metrics: dict = {}
    if n_evaluated > 0:
        print(f"\n{'Metric':<30} {'Value':>8}")
        print("-" * 40)
        for k in K_vals:
            for thresh in [0.25, 0.5]:
                vals  = recalls[(k, thresh)]
                r     = float(np.mean(vals)) if vals else 0.0
                label = f"Recall@{k}  IoU≥{thresh}"
                print(f"  {label:<28} {r*100:>7.2f}%")
                metrics[f"recall_at{k}_iou{thresh}"] = r
        print("-" * 40)
        mean_iou  = float(np.mean(top1_ious))
        mean_dist = float(np.mean(top1_dists))
        print(f"  {'Mean top-1 IoU':<28} {mean_iou:>8.4f}")
        print(f"  {'Mean top-1 centre dist (m)':<28} {mean_dist:>8.3f}")
        metrics["mean_top1_iou"]  = mean_iou
        metrics["mean_top1_dist"] = mean_dist
    print(f"{'='*60}\n")

    # ---- save results --------------------------------------------------------
    results_path = out_dir / f"bbox_search_eval_{split}.json"
    save_data: dict = {
        "split":          split,
        "top_k":          top_k,
        "n_evaluated":    n_evaluated,
        "gt_missing":     gt_missing_count,
        "metrics":        metrics,
    }
    if save_per_question:
        save_data["per_question"] = per_question_results

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved → {results_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scan-to-map bounding-box search against ScanQA ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scanqa-root",   required=True,
        help="Path to ScanQA data dir (containing ScanQA_v1.0_val.json)")
    parser.add_argument("--outputs-root",  required=True,
        help="Path to scan-to-map outputs/ directory")
    parser.add_argument("--scannet-root",  required=True,
        help="Path to ScanNet root (expects scans/<scene_id>/ subdirs with "
             ".aggregation.json, _vh_clean_2.0.010000.segs.json, _vh_clean_2.ply)")
    parser.add_argument("--split",         default="val",
        choices=["train", "val", "test_w_obj"],
        help="ScanQA split to evaluate (default: val)")
    parser.add_argument("--top-k",         type=int, default=10,
        help="Number of top components to retrieve per question (default: 10)")
    parser.add_argument("--iou-thresh",    type=float, default=0.25,
        help="Primary IoU threshold for reporting (default: 0.25; 0.25 and 0.5 "
             "are always reported)")
    parser.add_argument("--clip-model",    default="ViT-H-14",
        help="OpenCLIP model (must match what segment3d used, default: ViT-H-14)")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K",
        help="OpenCLIP pretrained weights (default: laion2B-s32B-b79K)")
    parser.add_argument("--device",        type=int, default=None,
        help="CUDA device index (default: auto)")
    parser.add_argument("--output-dir",    default="eval_results",
        help="Directory to save results JSON (default: eval_results/)")
    parser.add_argument("--save-per-question", action="store_true",
        help="Include per-question detail (bbox, IoU, dist) in output JSON")

    args = parser.parse_args()

    run_bbox_eval(
        scanqa_root       = args.scanqa_root,
        outputs_root      = args.outputs_root,
        scannet_root      = args.scannet_root,
        split             = args.split,
        top_k             = args.top_k,
        iou_thresh        = args.iou_thresh,
        clip_model        = args.clip_model,
        clip_pretrained   = args.clip_pretrained,
        device            = args.device,
        output_dir        = args.output_dir,
        save_per_question = args.save_per_question,
    )


if __name__ == "__main__":
    main()
