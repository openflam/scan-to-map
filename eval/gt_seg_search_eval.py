"""
Oracle segmentation evaluation for scan-to-map × ScanQA.

Answers the question: "If scan-to-map had perfect segmentation (GT instances),
how well can its CLIP-based search retrieve the correct object?"

This isolates the two failure modes:
    scan-to-map error = segmentation error  +  search/retrieval error

By replacing FastSAM segments with GT ScanNet instance segments, we fix the
first term and measure only the second.

Pipeline
--------
For each scene:
  1. Load GT instance segmentation from ScanNet annotation files.
  2. Project each GT object's 3D points into every camera frame → 2D bbox.
  3. Crop the best-view image(s) per object.
  4. Embed crops with OpenCLIP image encoder → build per-scene FAISS index.
For each ScanQA question:
  5. Embed question text with the same CLIP model.
  6. Search the GT-object FAISS index → top-K object IDs.
  7. Check if any retrieved object_id matches the GT object_ids from ScanQA.
  8. Report Recall@K and optionally 3D bbox IoU.

Required files per scene (--scannet-root  <root>/scans/<scene_id>/)
    <scene_id>.aggregation.json            object_id → segment IDs
    <scene_id>_vh_clean_2.0.010000.segs.json  vertex → segIndex
    <scene_id>_vh_clean_2.ply              vertex XYZ positions

Required from the scan-to-map data directory (--data-root  scan-to-map/data/)
    scannet_<scene_id>/hloc_data/sfm_reconstruction/   COLMAP txt model
    scannet_<scene_id>/ns_data/images/                 RGB frames

Usage
-----
    python eval/gt_seg_search_eval.py \\
        --scanqa-root   /path/to/ScanQA/data/qa/ScanQA_v1.0 \\
        --scannet-root  /path/to/scannet/data \\
        --data-root     /path/to/scan-to-map/data \\
        [--split val] \\
        [--questions-file /path/to/custom_questions.json] \\
        [--top-k 10] \\
        [--min-visible-points 10] \\
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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GT segmentation loader
# ---------------------------------------------------------------------------

def load_gt_instances(
    scannet_scene_dir: Path,
    scene_id: str,
) -> Optional[Dict[int, np.ndarray]]:
    """
    Build a dict: object_id (int) → vertex XYZ positions (N×3 float32).

    Loads three ScanNet annotation files:
        <scene_id>.aggregation.json
        <scene_id>_vh_clean_2.0.010000.segs.json
        <scene_id>_vh_clean_2.ply
    """
    agg_path  = scannet_scene_dir / f"{scene_id}.aggregation.json"
    segs_path = scannet_scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    ply_path  = scannet_scene_dir / f"{scene_id}_vh_clean_2.ply"

    for p in (agg_path, segs_path, ply_path):
        if not p.exists():
            return None

    with open(agg_path) as f:
        agg = json.load(f)
    with open(segs_path) as f:
        segs_data = json.load(f)

    seg_indices = np.array(segs_data["segIndices"], dtype=np.int32)
    verts = _read_ply_vertices(ply_path)
    if verts is None:
        return None

    # object_id → set of segment IDs
    obj_segs: Dict[int, Set[int]] = {}
    for group in agg.get("segGroups", []):
        obj_id = int(group["objectId"])
        obj_segs[obj_id] = set(group["segments"])

    # object_id → vertex mask → XYZ
    instances: Dict[int, np.ndarray] = {}
    for obj_id, seg_set in obj_segs.items():
        mask = np.isin(seg_indices, list(seg_set))
        pts = verts[mask]
        if len(pts) >= 3:
            instances[obj_id] = pts.astype(np.float32)

    return instances


def _read_ply_vertices(ply_path: Path) -> Optional[np.ndarray]:
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        v = np.asarray(mesh.vertices, dtype=np.float32)
        return v if len(v) > 0 else None
    except ImportError:
        pass
    return _ply_read_xyz_fallback(ply_path)


def _ply_read_xyz_fallback(ply_path: Path) -> Optional[np.ndarray]:
    import struct
    with open(ply_path, "rb") as fh:
        header_lines = []
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
        prop_types: List[Tuple[str, str]] = []
        in_vertex = False
        for l in lines:
            if l.startswith("element vertex"):
                in_vertex = True
            elif l.startswith("element") and not l.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and l.startswith("property"):
                parts = l.split()
                prop_types.append((parts[1], parts[2]))
        type_sizes = {"float": 4, "float32": 4, "double": 8, "float64": 8,
                      "int": 4, "int32": 4, "uint": 4, "uint32": 4,
                      "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
                      "char": 1, "int8": 1, "uchar": 1, "uint8": 1}
        type_fmts = {"float": "f", "float32": "f", "double": "d", "float64": "d",
                     "int": "i", "int32": "i", "uint": "I", "uint32": "I",
                     "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                     "char": "b", "int8": "b", "uchar": "B", "uint8": "B"}
        xyz_indices = {name: i for i, (_, name) in enumerate(prop_types)
                       if name in ("x", "y", "z")}
        if len(xyz_indices) < 3:
            return None
        row_size = sum(type_sizes.get(t, 4) for t, _ in prop_types)
        verts = np.empty((n_verts, 3), dtype=np.float32)
        xi, yi, zi = xyz_indices["x"], xyz_indices["y"], xyz_indices["z"]
        if fmt == "ascii":
            for i in range(n_verts):
                row = fh.readline().split()
                verts[i] = [float(row[xi]), float(row[yi]), float(row[zi])]
        else:
            endian = "<" if "little" in fmt else ">"
            row_fmt = endian + "".join(type_fmts.get(t, "f") for t, _ in prop_types)
            unpacker = struct.Struct(row_fmt)
            for i in range(n_verts):
                vals = unpacker.unpack(fh.read(row_size))
                verts[i] = [vals[xi], vals[yi], vals[zi]]
    return verts


# ---------------------------------------------------------------------------
# COLMAP loader (reuses scan-to-map's reader via txt files)
# ---------------------------------------------------------------------------

def load_colmap_txt(colmap_dir: Path):
    """
    Load cameras.txt and images.txt.
    Returns (cameras_dict, images_dict) in a minimal format.
    cameras_dict[1] = {"model": "PINHOLE", "w": W, "h": H,
                        "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    images_dict[img_id] = {"name": str, "R": (3,3), "t": (3,)}
    """
    cameras: Dict[int, dict] = {}
    with open(colmap_dir / "cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            W, H   = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            if model == "PINHOLE" and len(params) >= 4:
                cameras[cam_id] = {"model": model, "W": W, "H": H,
                                   "fx": params[0], "fy": params[1],
                                   "cx": params[2], "cy": params[3]}

    images: Dict[int, dict] = {}
    with open(colmap_dir / "images.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        # cam_id = int(parts[8])
        name = parts[9]

        # quat (w,x,y,z) → rotation matrix  R_w2c
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float32)

        images[img_id] = {"name": name, "R": R, "t": t}
        i += 2   # skip the POINTS2D line

    return cameras, images


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def project_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray,
                   fx: float, fy: float, cx: float, cy: float,
                   W: int, H: int, min_depth: float = 0.1
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project world-space points into camera image.
    Returns (pixel_xy (K,2), valid_mask (N,)).
    """
    cam = (R @ pts.T).T + t           # (N,3) camera-space
    depth = cam[:, 2]
    front = depth > min_depth
    d = np.where(front, depth, 1.0)
    x = fx * cam[:, 0] / d + cx
    y = fy * cam[:, 1] / d + cy
    in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    valid = front & in_img
    xys = np.stack([x[valid], y[valid]], axis=1)
    return xys, valid


def best_crop_for_object(
    pts_xyz: np.ndarray,
    cameras: Dict,
    images: Dict,
    images_dir: Path,
    top_n: int = 1,
    min_visible: int = 10,
) -> List[np.ndarray]:
    """
    Find the top-n camera frames where the object is most visible and
    return RGB crops (as numpy arrays H×W×3).
    """
    from PIL import Image as PILImage

    cam = next(iter(cameras.values()))  # assume single camera model
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["W"], cam["H"]

    visibility: List[Tuple[int, int, np.ndarray]] = []  # (n_visible, img_id, xys)
    for img_id, img_data in images.items():
        xys, valid = project_points(
            pts_xyz, img_data["R"], img_data["t"],
            fx, fy, cx, cy, W, H
        )
        if len(xys) >= min_visible:
            visibility.append((len(xys), img_id, xys))

    if not visibility:
        return []

    visibility.sort(key=lambda x: x[0], reverse=True)
    crops = []
    for _, img_id, xys in visibility[:top_n]:
        img_name = images[img_id]["name"]
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        try:
            img = PILImage.open(img_path).convert("RGB")
            # 2D bounding box of projected points + 10% padding
            x1, y1 = int(xys[:, 0].min()), int(xys[:, 1].min())
            x2, y2 = int(xys[:, 0].max()), int(xys[:, 1].max())
            pw = max(10, int((x2 - x1) * 0.1))
            ph = max(10, int((y2 - y1) * 0.1))
            x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
            x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = img.crop((x1, y1, x2, y2))
            crops.append(np.array(crop))
        except Exception:
            continue
    return crops


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def load_clip_model(model_name: str, pretrained: str, device_str: str):
    try:
        import open_clip
    except ImportError:
        raise ImportError("pip install open-clip-torch")
    import torch
    print(f"Loading CLIP {model_name} ({pretrained}) on {device_str} …")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device_str).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def encode_images_clip(crops: List[np.ndarray], model, preprocess,
                       device_str: str) -> Optional[np.ndarray]:
    """Embed a list of crop arrays → mean L2-normalised (D,) vector."""
    import torch
    from PIL import Image as PILImage
    if not crops:
        return None
    tensors = []
    for arr in crops:
        img = PILImage.fromarray(arr)
        tensors.append(preprocess(img))
    batch = torch.stack(tensors).to(device_str)
    with torch.no_grad():
        feats = model.encode_image(batch)          # (N, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feat  = feats.mean(dim=0)                   # average over views
        feat  = feat / feat.norm()
    return feat.cpu().float().numpy()


def encode_texts_clip(questions: List[str], model, tokenizer,
                      device_str: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        tokens = tokenizer(questions).to(device_str)
        feats  = model.encode_text(tokens)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Per-scene GT index builder
# ---------------------------------------------------------------------------

def build_gt_index(
    scene_id: str,
    scannet_scene_dir: Path,
    data_scene_dir: Path,
    model,
    preprocess,
    device_str: str,
    min_visible: int = 10,
    top_n_views: int = 3,
) -> Optional[Tuple[object, List[int], Dict[int, np.ndarray]]]:
    """
    For one scene build a FAISS index over GT object visual embeddings.

    Returns (faiss_index, object_ids_list, gt_bboxes_dict) or None.
    object_ids_list[i] = ScanNet object_id for index row i
    gt_bboxes_dict[object_id] = (bbox_min, bbox_max)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("pip install faiss-cpu")

    instances = load_gt_instances(scannet_scene_dir, scene_id)
    if not instances:
        return None

    colmap_dir = data_scene_dir / "hloc_data" / "sfm_reconstruction"
    images_dir = data_scene_dir / "ns_data" / "images"
    if not colmap_dir.exists() or not images_dir.exists():
        print(f"  [SKIP] {scene_id}: missing COLMAP or images dir")
        return None

    cameras, images = load_colmap_txt(colmap_dir)
    if not cameras or not images:
        return None

    embeddings: List[np.ndarray] = []
    object_ids: List[int] = []
    gt_bboxes:  Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for obj_id, pts in instances.items():
        # GT bbox
        gt_bboxes[obj_id] = (pts.min(axis=0), pts.max(axis=0))

        # Subsample large objects to speed up projection
        if len(pts) > 5000:
            idx = np.random.default_rng(obj_id).choice(len(pts), 5000, replace=False)
            pts_sub = pts[idx]
        else:
            pts_sub = pts

        crops = best_crop_for_object(
            pts_sub, cameras, images, images_dir,
            top_n=top_n_views, min_visible=min_visible
        )
        if not crops:
            continue

        emb = encode_images_clip(crops, model, preprocess, device_str)
        if emb is None:
            continue

        embeddings.append(emb)
        object_ids.append(obj_id)

    if not embeddings:
        return None

    emb_matrix = np.stack(embeddings).astype(np.float32)   # (N, D)
    D = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(D)                            # inner-product = cosine on L2-normed vecs
    index.add(emb_matrix)

    return index, object_ids, gt_bboxes


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_gt_seg_eval(
    scanqa_root: str,
    scannet_root: str,
    data_root: str,
    split: str = "val",
    questions_file: Optional[str] = None,
    top_k: int = 10,
    min_visible: int = 10,
    top_n_views: int = 3,
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
    scannet_path = Path(scannet_root)
    data_path    = Path(data_root)
    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_file = Path(questions_file) if questions_file else scanqa_path / f"ScanQA_v1.0_{split}.json"
    if not qa_file.exists():
        raise FileNotFoundError(qa_file)

    with open(qa_file) as f:
        qa_data: List[dict] = json.load(f)

    qa_data = [q for q in qa_data if q.get("object_ids")]
    print(f"Loaded {len(qa_data)} annotated questions from {qa_file.name}")

    by_scene: Dict[str, list] = {}
    for q in qa_data:
        by_scene.setdefault(q["scene_id"], []).append(q)

    # Only evaluate scenes that have COLMAP data (ran scannet.py)
    def dataset_dir(sid: str) -> Path:
        return data_path / f"scannet_{sid}"

    scenes_to_eval = {
        sid for sid in by_scene
        if (dataset_dir(sid) / "hloc_data" / "sfm_reconstruction").exists()
    }
    skipped = set(by_scene) - scenes_to_eval
    if skipped:
        print(f"Warning: {len(skipped)} scenes missing COLMAP data (run scannet.py first).")
    print(f"Evaluating {len(scenes_to_eval)} scenes "
          f"({sum(len(by_scene[s]) for s in scenes_to_eval)} questions)\n")

    if not scenes_to_eval:
        raise RuntimeError("No scenes ready. Run data-processor/scannet.py first.")

    model, preprocess, tokenizer = load_clip_model(clip_model, clip_pretrained, device_str)

    K_vals = [k for k in [1, 5, 10] if k <= top_k]
    recalls: Dict[int, List[float]] = {k: [] for k in K_vals}
    no_embed_count = 0
    per_question_results: List[dict] = []

    for scene_id in sorted(scenes_to_eval):
        scannet_dir = scannet_path / "scans" / scene_id
        data_dir    = dataset_dir(scene_id)
        questions   = by_scene[scene_id]

        print(f"  Scene {scene_id}  ({len(questions)} questions)", end="", flush=True)

        result = build_gt_index(
            scene_id, scannet_dir, data_dir,
            model, preprocess, device_str,
            min_visible=min_visible,
            top_n_views=top_n_views,
        )
        if result is None:
            print("  [no GT index built]")
            continue

        faiss_index, obj_id_list, gt_bboxes = result
        print(f"  → {len(obj_id_list)} GT objects indexed", end="")

        # Encode all questions for this scene
        q_texts = [q["question"] for q in questions]
        q_vecs  = encode_texts_clip(q_texts, model, tokenizer, device_str)

        scene_no_emb = 0
        for q_item, q_vec in zip(questions, q_vecs):
            gt_obj_ids: Set[int] = set(q_item["object_ids"])

            k = min(top_k, faiss_index.ntotal)
            _, indices = faiss_index.search(
                q_vec[np.newaxis, :].astype(np.float32), k
            )
            retrieved_ids = [obj_id_list[i] for i in indices[0] if 0 <= i < len(obj_id_list)]

            for kv in K_vals:
                top_k_ids = set(retrieved_ids[:kv])
                hit = float(bool(top_k_ids & gt_obj_ids))
                recalls[kv].append(hit)

            if save_per_question:
                per_question_results.append({
                    "question_id":   q_item["question_id"],
                    "scene_id":      scene_id,
                    "question":      q_item["question"],
                    "gt_object_ids": list(gt_obj_ids),
                    "retrieved_ids": retrieved_ids,
                    "hit@1":  float(bool(set(retrieved_ids[:1]) & gt_obj_ids)),
                    "hit@5":  float(bool(set(retrieved_ids[:5]) & gt_obj_ids)),
                    "hit@10": float(bool(set(retrieved_ids[:10]) & gt_obj_ids)),
                })

        print()

    # ---- print summary -------------------------------------------------------
    n_q = len(recalls[K_vals[0]])
    print(f"\n{'='*55}")
    print(f"Oracle segmentation eval — {n_q} questions")
    print(f"{'='*55}")
    print(f"\n{'Metric':<30} {'Value':>8}")
    print("-" * 40)
    metrics: dict = {}
    for k in K_vals:
        vals = recalls[k]
        r = float(np.mean(vals)) if vals else 0.0
        label = f"Recall@{k}  (object hit)"
        print(f"  {label:<28} {r*100:>7.2f}%")
        metrics[f"recall_at{k}"] = r
    print(f"{'='*55}\n")

    # ---- save ----------------------------------------------------------------
    save_data: dict = {
        "split": split, "top_k": top_k,
        "n_evaluated": n_q,
        "metrics": metrics,
    }
    if save_per_question:
        save_data["per_question"] = per_question_results

    results_path = out_dir / f"gt_seg_search_eval_{split}.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved → {results_path}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle segmentation eval: GT segments + CLIP search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scanqa-root",   required=True,
        help="Path to ScanQA dir containing ScanQA_v1.0_val.json")
    parser.add_argument("--scannet-root",  required=True,
        help="ScanNet root (expects scans/<scene_id>/ with annotation files)")
    parser.add_argument("--data-root",     required=True,
        help="scan-to-map data/ dir (expects scannet_<scene_id>/hloc_data/ + ns_data/)")
    parser.add_argument("--split",         default="val",
        choices=["train", "val", "test_w_obj"])
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional path to a custom questions JSON file in ScanQA format. "
             "If provided, overrides --split file selection.",
    )
    parser.add_argument("--top-k",         type=int, default=10)
    parser.add_argument("--min-visible",   type=int, default=10,
        help="Min projected points to consider a frame valid (default: 10)")
    parser.add_argument("--top-n-views",   type=int, default=3,
        help="Number of best views to crop per object (default: 3)")
    parser.add_argument("--clip-model",    default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--device",        type=int, default=None)
    parser.add_argument("--output-dir",    default="eval_results")
    parser.add_argument("--save-per-question", action="store_true")

    args = parser.parse_args()
    run_gt_seg_eval(
        scanqa_root      = args.scanqa_root,
        scannet_root     = args.scannet_root,
        data_root        = args.data_root,
        split            = args.split,
        questions_file   = args.questions_file,
        top_k            = args.top_k,
        min_visible      = args.min_visible,
        top_n_views      = args.top_n_views,
        clip_model       = args.clip_model,
        clip_pretrained  = args.clip_pretrained,
        device           = args.device,
        output_dir       = args.output_dir,
        save_per_question= args.save_per_question,
    )


if __name__ == "__main__":
    main()
