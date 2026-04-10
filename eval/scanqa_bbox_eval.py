"""
Evaluate ScanQA model predictions against ScanNet GT object boxes.

This script is the ScanQA-side companion to `bbox_search_eval.py`.
It answers:
    "Given a trained ScanQA model's predicted bbox for each question, how often
     does that bbox land on the correct GT object(s)?"

Supported prediction formats
----------------------------
1. JSON list from `ScanQA/scripts/eval.py` or `predict.py`
   Each item typically contains:
       {
         "scene_id": "...",
         "question_id": "...",
         "bbox": [[x,y,z], ... 8 corners ...],
         "answer_top10": [...]
       }

2. Pickle from `ScanQA/scripts/eval.py`
   Nested dict:
       predictions[scene_id][question_id] = {
         "pred_bbox": np.ndarray(8,3),
         "pred_answers_at10": [...],
         ...
       }

Metrics
-------
- Object hit @ IoU>=0.25 / 0.5 (best overlap against any GT object_id)
- Mean best-object IoU
- Mean union-IoU (pred bbox vs union of all GT object_ids for the question)
- Mean centre distance (pred bbox vs GT union bbox)
- Optional exact-match text metrics if answer predictions are present
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bbox_search_eval import aabb_iou, build_gt_bbox, center_distance


def _normalize_pred_bbox(raw_bbox) -> Optional[np.ndarray]:
    """
    Convert a ScanQA predicted bbox into an (8,3) numpy array.
    Returns None if the shape is unsupported.
    """
    arr = np.array(raw_bbox, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.shape == (8, 3):
        return arr
    if arr.ndim == 1 and arr.size == 24:
        return arr.reshape(8, 3)
    return None


def _bbox_to_aabb(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return corners.min(axis=0), corners.max(axis=0)


def _load_predictions(pred_path: Path) -> Dict[str, dict]:
    """
    Returns normalized dict:
        preds[question_id] = {
            "scene_id": str,
            "bbox": np.ndarray(8,3),
            "answer_top10": list[str]   # optional
        }
    """
    if pred_path.suffix == ".json":
        with open(pred_path, "r") as f:
            raw = json.load(f)

        preds: Dict[str, dict] = {}
        for item in raw:
            qid = str(item["question_id"])
            bbox = _normalize_pred_bbox(item.get("bbox") or item.get("pred_bbox"))
            if bbox is None:
                continue
            preds[qid] = {
                "scene_id": str(item["scene_id"]),
                "bbox": bbox,
                "answer_top10": item.get("answer_top10")
                or item.get("pred_answers_at10")
                or [],
            }
        return preds

    if pred_path.suffix == ".pkl":
        with open(pred_path, "rb") as f:
            raw = pickle.load(f)

        preds = {}
        for scene_id, scene_preds in raw.items():
            for qid, item in scene_preds.items():
                bbox = _normalize_pred_bbox(item.get("pred_bbox") or item.get("bbox"))
                if bbox is None:
                    continue
                preds[str(qid)] = {
                    "scene_id": str(scene_id),
                    "bbox": bbox,
                    "answer_top10": item.get("pred_answers_at10")
                    or item.get("answer_top10")
                    or [],
                }
        return preds

    raise ValueError(f"Unsupported prediction file type: {pred_path}")


def _topk_em(pred_answers: List[str], ref_answers: List[str]) -> Tuple[float, float]:
    if not pred_answers:
        return 0.0, 0.0
    top1 = float(pred_answers[0] in ref_answers)
    top10 = float(any(ans in ref_answers for ans in pred_answers[:10]))
    return top1, top10


def run_scanqa_bbox_eval(
    scanqa_root: str,
    scannet_root: str,
    pred_path: str,
    split: str = "val",
    output_dir: str = "eval_results",
    save_per_question: bool = False,
) -> dict:
    scanqa_path = Path(scanqa_root)
    scannet_path = Path(scannet_root)
    pred_file = Path(pred_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_file = scanqa_path / f"ScanQA_v1.0_{split}.json"
    if not qa_file.exists():
        raise FileNotFoundError(f"ScanQA data not found: {qa_file}")

    with open(qa_file, "r") as f:
        qa_data: List[dict] = json.load(f)

    qa_data = [q for q in qa_data if q.get("object_ids")]
    print(f"Loaded {len(qa_data)} annotated questions from {qa_file.name}")

    preds = _load_predictions(pred_file)
    print(f"Loaded {len(preds)} predictions from {pred_file}")

    hit25: List[float] = []
    hit50: List[float] = []
    best_ious: List[float] = []
    union_ious: List[float] = []
    center_dists: List[float] = []
    top1_ems: List[float] = []
    top10_ems: List[float] = []
    missing_pred = 0
    missing_gt = 0

    per_question = []

    for item in qa_data:
        qid = str(item["question_id"])
        if qid not in preds:
            missing_pred += 1
            continue

        pred = preds[qid]
        scene_id = item["scene_id"]
        scene_dir = scannet_path / "scans" / scene_id

        pred_min, pred_max = _bbox_to_aabb(pred["bbox"])

        gt_union = build_gt_bbox(scene_dir, scene_id, item["object_ids"])
        if gt_union is None:
            missing_gt += 1
            continue
        gt_union_min, gt_union_max = gt_union

        # Compare against each GT object individually as well as the union bbox.
        per_obj_ious = []
        for obj_id in item["object_ids"]:
            gt_single = build_gt_bbox(scene_dir, scene_id, [obj_id])
            if gt_single is None:
                continue
            gt_min, gt_max = gt_single
            per_obj_ious.append(aabb_iou(pred_min, pred_max, gt_min, gt_max))

        best_iou = max(per_obj_ious) if per_obj_ious else 0.0
        union_iou = aabb_iou(pred_min, pred_max, gt_union_min, gt_union_max)
        dist = center_distance(pred_min, pred_max, gt_union_min, gt_union_max)
        em1, em10 = _topk_em(pred.get("answer_top10", []), item.get("answers", []))

        hit25.append(float(best_iou >= 0.25))
        hit50.append(float(best_iou >= 0.5))
        best_ious.append(best_iou)
        union_ious.append(union_iou)
        center_dists.append(dist)
        top1_ems.append(em1)
        top10_ems.append(em10)

        if save_per_question:
            per_question.append(
                {
                    "question_id": qid,
                    "scene_id": scene_id,
                    "question": item["question"],
                    "object_ids": item["object_ids"],
                    "object_names": item.get("object_names", []),
                    "pred_answer_top10": pred.get("answer_top10", []),
                    "gt_answers": item.get("answers", []),
                    "pred_bbox_min": pred_min.tolist(),
                    "pred_bbox_max": pred_max.tolist(),
                    "gt_union_bbox_min": gt_union_min.tolist(),
                    "gt_union_bbox_max": gt_union_max.tolist(),
                    "best_object_iou": best_iou,
                    "union_iou": union_iou,
                    "center_dist": dist,
                    "hit_iou25": float(best_iou >= 0.25),
                    "hit_iou50": float(best_iou >= 0.5),
                    "top1_em": em1,
                    "top10_em": em10,
                }
            )

    n_eval = len(best_ious)
    print("\n" + "=" * 60)
    print(f"Evaluated {n_eval} questions")
    print(f"Missing predictions: {missing_pred}")
    print(f"Missing GT boxes  : {missing_gt}")
    print("=" * 60)

    metrics = {
        "object_hit_iou25": float(np.mean(hit25)) if hit25 else 0.0,
        "object_hit_iou50": float(np.mean(hit50)) if hit50 else 0.0,
        "mean_best_object_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "mean_union_iou": float(np.mean(union_ious)) if union_ious else 0.0,
        "mean_center_dist": float(np.mean(center_dists)) if center_dists else 0.0,
        "top1_em": float(np.mean(top1_ems)) if top1_ems else 0.0,
        "top10_em": float(np.mean(top10_ems)) if top10_ems else 0.0,
    }

    print("\nMetric                            Value")
    print("-" * 40)
    print(f"  Object hit  IoU>=0.25         {metrics['object_hit_iou25'] * 100:7.2f}%")
    print(f"  Object hit  IoU>=0.50         {metrics['object_hit_iou50'] * 100:7.2f}%")
    print(f"  Mean best-object IoU          {metrics['mean_best_object_iou']:8.4f}")
    print(f"  Mean union IoU                {metrics['mean_union_iou']:8.4f}")
    print(f"  Mean centre dist (m)          {metrics['mean_center_dist']:8.3f}")
    print(f"  Top-1 EM                      {metrics['top1_em'] * 100:7.2f}%")
    print(f"  Top-10 EM                     {metrics['top10_em'] * 100:7.2f}%")
    print("=" * 60 + "\n")

    out = {
        "split": split,
        "n_evaluated": n_eval,
        "missing_predictions": missing_pred,
        "missing_gt": missing_gt,
        "metrics": metrics,
    }
    if save_per_question:
        out["per_question"] = per_question

    out_path = out_dir / f"scanqa_bbox_eval_{split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ScanQA predicted bboxes against ScanNet GT object boxes."
    )
    parser.add_argument(
        "--scanqa-root",
        required=True,
        help="Path to ScanQA QA dir containing ScanQA_v1.0_<split>.json",
    )
    parser.add_argument(
        "--scannet-root",
        required=True,
        help="Path to ScanNet root (expects scans/<scene_id>/...)",
    )
    parser.add_argument(
        "--pred-path",
        required=True,
        help="Path to ScanQA pred JSON/PKL file (e.g. pred.val.json or pred.val.pkl)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test_w_obj"],
        help="Which ScanQA split the prediction file corresponds to",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory to save eval JSON",
    )
    parser.add_argument(
        "--save-per-question",
        action="store_true",
        help="Include per-question details in the output JSON",
    )
    args = parser.parse_args()

    run_scanqa_bbox_eval(
        scanqa_root=args.scanqa_root,
        scannet_root=args.scannet_root,
        pred_path=args.pred_path,
        split=args.split,
        output_dir=args.output_dir,
        save_per_question=args.save_per_question,
    )


if __name__ == "__main__":
    main()
