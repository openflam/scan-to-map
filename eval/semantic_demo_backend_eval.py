#!/usr/bin/env python3
"""
Backend-only evaluation for the semantic-3d-search stack (same API as the web demo).

Calls search-server HTTP routes — no browser:
  POST /search  →  reason, search_time_ms, components[{ bbox, caption, component_id }],
  optional answer_selection. When ScanQA ground truth is on the task (answers, object_ids),
  each result includes an ``eval`` block when GT exists. Use ``--output-json`` for one JSON file
  with overall stats plus a list of questions (ground truth, system return, eval).

Prerequisites:
  • search-server running (e.g. python app.py or docker compose)
  • PostGIS populated:  search-server/spatial_db/create_tables.py <dataset_name>
  • OPENAI_API_KEY if using OpenAI-based providers (not needed for BM25)

Examples:
  # ScanNet scene in DB as scannet_scene0000_00 (shortcut: --scannet-scene)
  python semantic_demo_backend_eval.py \\
      --scannet-scene scene0000_00 \\
      --question "where is the sofa" \\
      --method BM25

  # Questions from ScanQA JSON for that scene (same dataset name convention)
  python semantic_demo_backend_eval.py \\
      --scannet-scene scene0000_00 \\
      --scanqa-json ../../../ScanQA/data/qa/ScanQA_v1.0/ScanQA_v1.0_val.json \\
      --limit 20 \\
      --output-json results_scene0000.json

  # Only simplified-bucket "object" questions (what is / what object / …) for that scene
  python semantic_demo_backend_eval.py \\
      --scannet-scene scene0000_00 \\
      --scanqa-json ../../../ScanQA/data/qa/ScanQA_v1.0/ScanQA_v1.0_val.json \\
      --scanqa-bucket object_retrieval \\
      --method "gpt-5-mini [Full]" \\
      --output-json scene0000_object_retrieval.json

  python semantic_demo_backend_eval.py \\
      --questions-file questions.txt \\
      --dataset-name scannet_scene0000_00 \\
      --method "gpt-5-mini [Full]" \\
      --output-json results.json

  python semantic_demo_backend_eval.py --list-methods --server http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Repo layout: <workspace>/scan-to-map/scan-to-map/eval/this_file.py → parents[2] = workspace root with ScanQA/
_SCAN_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_PARENT = Path(__file__).resolve().parents[1]
_SS_UTILS = _EVAL_PARENT / "search-server" / "utils"
if str(_SS_UTILS) not in sys.path:
    sys.path.insert(0, str(_SS_UTILS))

from scanqa_category import CATEGORY_ORDER, bucket  # noqa: E402
DEFAULT_SCANQA_VAL = (
    _SCAN_REPO_ROOT
    / "ScanQA"
    / "data"
    / "qa"
    / "ScanQA_v1.0"
    / "ScanQA_v1.0_val.json"
)


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _get_json(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else None


def _normalized_answer_set(ref_answers: Any) -> Optional[Set[str]]:
    if not isinstance(ref_answers, list) or not ref_answers:
        return None
    out: Set[str] = set()
    for a in ref_answers:
        if a is None:
            continue
        s = str(a).strip()
        if s:
            out.add(s.casefold())
    return out if out else None


def compute_eval_metrics(task_item: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare /search response to ScanQA ground truth when present on the task.

    Text: exact match (case-insensitive, stripped) of best_answer_from_pool and of
    top_10_answers_from_pool against ScanQA ``answers``.

    Bbox: ``component_id`` in the search-server DB is aligned with ScanNet ``objectId``;
    correctness = whether predicted component id(s) overlap ScanQA ``object_ids``.
    """
    ref_answers = task_item.get("answers")
    ref_objects = task_item.get("object_ids")

    text_block: Dict[str, Any] = {"ground_truth_available": False}
    ref_norm = _normalized_answer_set(ref_answers)
    if ref_norm is not None:
        text_block["ground_truth_available"] = True
        text_block["reference_answers"] = list(ref_answers)
        sel = response.get("answer_selection") or {}
        best = sel.get("best_answer_from_pool")
        top10_raw = sel.get("top_10_answers_from_pool")
        if not isinstance(top10_raw, list):
            top10_raw = []
        top10 = [str(x) for x in top10_raw[:10]]

        best_s = str(best).strip() if best is not None else ""
        pool_pred = bool(best_s) and best_s.upper() != "NONE"
        text_block["prediction_available"] = pool_pred
        text_block["predicted_best"] = best if pool_pred else None
        text_block["predicted_top10"] = top10

        if pool_pred:
            text_block["text_top1_correct"] = best_s.casefold() in ref_norm
        else:
            text_block["text_top1_correct"] = False
        text_block["text_top10_correct"] = any(
            str(p).strip() and str(p).strip().casefold() in ref_norm for p in top10
        )

    bbox_block: Dict[str, Any] = {"ground_truth_available": False}
    if isinstance(ref_objects, list) and ref_objects:
        try:
            gt_ids = {int(x) for x in ref_objects}
        except (TypeError, ValueError):
            gt_ids = set()
        if gt_ids:
            bbox_block["ground_truth_available"] = True
            bbox_block["reference_object_ids"] = list(ref_objects)
            comps = response.get("components") or []
            pred_ids: List[int] = []
            for c in comps:
                try:
                    pred_ids.append(int(c.get("component_id")))
                except (TypeError, ValueError):
                    continue
            bbox_block["predicted_component_ids"] = pred_ids
            top1 = pred_ids[0] if pred_ids else None
            bbox_block["bbox_top1_correct"] = top1 in gt_ids if top1 is not None else False
            bbox_block["bbox_any_retrieved_correct"] = any(pid in gt_ids for pid in pred_ids)

    return {"text": text_block, "bbox": bbox_block}


def simplify_system_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of POST /search JSON for reports."""
    comps: List[Dict[str, Any]] = []
    for c in response.get("components") or []:
        comps.append(
            {
                "component_id": c.get("component_id"),
                "caption": c.get("caption"),
                "bbox": c.get("bbox"),
            }
        )
    out: Dict[str, Any] = {
        "reason": response.get("reason"),
        "search_time_ms": response.get("search_time_ms"),
        "components": comps,
    }
    if response.get("answer_selection") is not None:
        out["answer_selection"] = response["answer_selection"]
    return out


def ground_truth_from_task(task_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gt: Dict[str, Any] = {}
    if task_item.get("answers") is not None:
        gt["answers"] = task_item["answers"]
    if task_item.get("object_ids") is not None:
        gt["object_ids"] = task_item["object_ids"]
    if task_item.get("object_names"):
        gt["object_names"] = task_item["object_names"]
    return gt if gt else None


def build_ground_truth_metrics_summary(counters: Dict[str, int]) -> Dict[str, Any]:
    """Aggregate text/bbox metrics for the output JSON ``summary`` section."""
    block: Dict[str, Any] = {}
    n_txt = counters["text_gt"]
    n_bbox = counters["bbox_gt"]
    if n_txt:
        block["text_answers"] = {
            "questions_with_reference": n_txt,
            "questions_with_pool_prediction": counters["text_pred"],
            "top1_accuracy": round(counters["text_top1_ok"] / n_txt, 6),
            "top10_recall": round(counters["text_top10_ok"] / n_txt, 6),
        }
    if n_bbox:
        block["bbox_components"] = {
            "questions_with_reference": n_bbox,
            "top1_component_hit_rate": round(counters["bbox_top1_ok"] / n_bbox, 6),
            "any_retrieved_component_hit_rate": round(counters["bbox_any_ok"] / n_bbox, 6),
        }
    return block


def _eval_for_json_export(
    ev: Optional[Dict[str, Any]],
    task_item: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Prepare eval block for JSON: show GT next to predictions; drop duplicate ref keys."""
    if not ev:
        return None
    out = json.loads(json.dumps(ev))
    t = out.get("text")
    if isinstance(t, dict):
        t.pop("reference_answers", None)
        if t.get("ground_truth_available"):
            ans = task_item.get("answers")
            t["gt_answers"] = ans if isinstance(ans, list) else None
    b = out.get("bbox")
    if isinstance(b, dict):
        b.pop("reference_object_ids", None)
        if b.get("ground_truth_available"):
            oids = task_item.get("object_ids")
            b["gt_object_ids"] = oids if isinstance(oids, list) else None
    return out


def make_json_question_entry(
    index: int,
    task_item: Dict[str, Any],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """One question record for ``--output-json``."""
    gt_ans = task_item.get("answers")
    gt_obj = task_item.get("object_ids")
    rec: Dict[str, Any] = {
        "index": index,
        "question": task_item["question"],
        "question_id": task_item.get("question_id"),
        "scene_id": task_item.get("scene_id"),
        "scanqa_bucket": task_item.get("scanqa_bucket"),
        "gt_answers": gt_ans if isinstance(gt_ans, list) else None,
        "gt_object_ids": gt_obj if isinstance(gt_obj, list) else None,
        "ground_truth": ground_truth_from_task(task_item),
        "ok": bool(row.get("ok")),
    }
    if row.get("ok"):
        rec["error"] = None
        rec["system"] = simplify_system_response(row["response"])
        rec["eval"] = _eval_for_json_export(row.get("eval"), task_item)
    else:
        rec["error"] = row.get("error")
        rec["system"] = None
        rec["eval"] = None
    return rec


def _update_eval_counters(ev: Dict[str, Any], c: Dict[str, int]) -> None:
    t = ev.get("text") or {}
    if t.get("ground_truth_available"):
        c["text_gt"] += 1
        if t.get("prediction_available"):
            c["text_pred"] += 1
        if t.get("text_top1_correct"):
            c["text_top1_ok"] += 1
        if t.get("text_top10_correct"):
            c["text_top10_ok"] += 1
    b = ev.get("bbox") or {}
    if b.get("ground_truth_available"):
        c["bbox_gt"] += 1
        if b.get("bbox_top1_correct"):
            c["bbox_top1_ok"] += 1
        if b.get("bbox_any_retrieved_correct"):
            c["bbox_any_ok"] += 1


def viewer_bbox_to_storage_min_max(bbox: Dict[str, float]) -> Dict[str, List[float]]:
    """
    Invert search-server app.transform_bbox (viewer → COLMAP-style min/max arrays).

    transform_bbox sets:
      x_min = min[1], y_min = min[2], z_min = min[0]  (viewer from storage)
    So storage min = [z_min, x_min, y_min] in viewer keys.
    """
    return {
        "min": [bbox["z_min"], bbox["x_min"], bbox["y_min"]],
        "max": [bbox["z_max"], bbox["x_max"], bbox["y_max"]],
    }


def run_search(
    base_url: str,
    dataset_name: str,
    method: str,
    question: str,
) -> Dict[str, Any]:
    base = base_url.rstrip("/")
    payload = {
        "dataset_name": dataset_name,
        "method": method,
        "query": [{"type": "text", "value": question}],
    }
    try:
        return _post_json(f"{base}/search", payload)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
        except json.JSONDecodeError:
            err_json = {"error": err_body or str(e)}
        raise RuntimeError(f"HTTP {e.code}: {err_json}") from e


def print_task_ground_truth(task_item: Dict[str, Any]) -> None:
    """Print ScanQA reference answers / object ids when present."""
    if task_item.get("answers") is not None:
        print("gt_answers:", json.dumps(task_item["answers"], ensure_ascii=False))
    if task_item.get("object_ids") is not None:
        print("gt_object_ids:", json.dumps(task_item["object_ids"], ensure_ascii=False))


def print_result(result: Dict[str, Any], show_storage_bbox: bool) -> None:
    print("reason:", result.get("reason", ""))
    print("search_time_ms:", result.get("search_time_ms"))
    sel = result.get("answer_selection")
    if sel:
        print("answer_selection:", json.dumps(sel, ensure_ascii=False))
    comps = result.get("components") or []
    print(f"components ({len(comps)}):")
    for i, c in enumerate(comps):
        bid = c.get("component_id")
        cap = (c.get("caption") or "")[:200]
        bbox = c.get("bbox") or {}
        print(f"  [{i}] component_id={bid}")
        print(f"      caption: {cap}{'...' if len(str(cap)) == 200 else ''}")
        if all(k in bbox for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
            print(
                f"      bbox (viewer / Three.js, same as web demo): "
                f"x[{bbox['x_min']:.4f}, {bbox['x_max']:.4f}] "
                f"y[{bbox['y_min']:.4f}, {bbox['y_max']:.4f}] "
                f"z[{bbox['z_min']:.4f}, {bbox['z_max']:.4f}]"
            )
        else:
            print(f"      bbox: {bbox}")
        if show_storage_bbox and bbox:
            mm = viewer_bbox_to_storage_min_max(bbox)
            print(f"      bbox (storage min/max [x,y,z] before API transform): min={mm['min']} max={mm['max']}")


def load_scanqa_for_scene(
    path: Path,
    scene_id: str,
    limit: Optional[int] = None,
    bucket_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load ScanQA entries (train/val/test JSON array) filtered by scene_id.

    If bucket_filter is set, only questions whose simplified category (bucket()) matches
    are included (same rules as search-server answer pools).

    Each item includes `question` plus ScanQA fields for JSONL ground-truth.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {path}")
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("scene_id") != scene_id:
            continue
        q = item.get("question")
        if not q:
            continue
        bq = bucket(str(q))
        if bucket_filter is not None and bq != bucket_filter:
            continue
        out.append(
            {
                "question": q,
                "question_id": item.get("question_id"),
                "scene_id": item.get("scene_id"),
                "answers": item.get("answers"),
                "object_ids": item.get("object_ids"),
                "object_names": item.get("object_names"),
                "scanqa_bucket": bq,
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def count_scanqa_rows_for_scene(path: Path, scene_id: str) -> int:
    """Number of ScanQA items for scene_id with a non-empty question (before bucket filter)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return 0
    n = 0
    for item in raw:
        if not isinstance(item, dict) or item.get("scene_id") != scene_id:
            continue
        if not item.get("question"):
            continue
        n += 1
    return n


def load_questions(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # JSON line: {"question": "..."}
        if line.startswith("{"):
            obj = json.loads(line)
            q = obj.get("question") or obj.get("q") or obj.get("text")
            if q:
                lines.append(str(q))
        else:
            lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call semantic search /search like the web demo (backend only).",
        epilog=(
            "Example: scene0000_00 appears in train.json but not in val.json — use train for that scene, "
            "or set --scene-id to a scene that exists in your JSON (e.g. from val). "
            "One line: python semantic_demo_backend_eval.py --scannet-scene scene0000_00 "
            "--scanqa-json /path/to/ScanQA_v1.0_train.json --scanqa-bucket object_retrieval "
            '--method "gpt-5-mini [Full]" --output-json /tmp/out.json'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:5000",
        help="search-server base URL (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="PostGIS / outputs folder name, e.g. scannet_scene0000_00. "
        "If omitted, derived from --scannet-scene as scannet_<scene_id>.",
    )
    parser.add_argument(
        "--scannet-scene",
        default=None,
        metavar="SCENE_ID",
        help="ScanNet scene id (e.g. scene0000_00). Sets dataset to scannet_<SCENE_ID> unless --dataset-name is set.",
    )
    parser.add_argument(
        "--scanqa-json",
        type=Path,
        default=None,
        help="ScanQA JSON array (train/val/test). Questions filtered by --scene-id.",
    )
    parser.add_argument(
        "--use-default-scanqa-val",
        action="store_true",
        help=f"If set, use ScanQA val JSON at {DEFAULT_SCANQA_VAL} (must exist).",
    )
    parser.add_argument(
        "--scene-id",
        default="scene0000_00",
        help="When using --scanqa-json, only include entries with this scene_id (default: scene0000_00).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max ScanQA questions to run (after scene and optional --scanqa-bucket filter).",
    )
    parser.add_argument(
        "--scanqa-bucket",
        default=None,
        choices=list(CATEGORY_ORDER),
        metavar="BUCKET",
        help=(
            "When using --scanqa-json, only questions matching this simplified category "
            f"({', '.join(CATEGORY_ORDER)}). E.g. object_retrieval = 'what is' / 'what object' style."
        ),
    )
    parser.add_argument(
        "--method",
        default="BM25",
        help='Search method (default: BM25). Examples: "gpt-5-mini [Full]", "CLIP ViT-H-14"',
    )
    parser.add_argument("--question", default=None, help="Single text question")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="One question per line, or JSONL with {\"question\": \"...\"} per line",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Append one JSON result per line (legacy; use --output-json for a single report file)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Write one JSON file: meta, summary (counts + accuracy), and a ``questions`` array "
            "(each entry: question, ground_truth, system return, eval)."
        ),
    )
    parser.add_argument(
        "--show-storage-bbox",
        action="store_true",
        help="Also print min/max in DB/COLMAP frame (inverse of transform_bbox)",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="GET /get_providers_list and exit",
    )
    parser.add_argument(
        "--no-eval-summary",
        action="store_true",
        help="Do not print aggregate text/bbox accuracy at the end (per-row eval is unchanged).",
    )
    args = parser.parse_args()

    base = args.server.rstrip("/")

    dataset_name = args.dataset_name
    if args.scannet_scene and dataset_name:
        print(
            "Note: both --dataset-name and --scannet-scene set; using --dataset-name.",
            file=sys.stderr,
        )
    elif args.scannet_scene and not dataset_name:
        dataset_name = f"scannet_{args.scannet_scene}"

    if args.list_methods:
        try:
            data = _get_json(f"{base}/get_providers_list")
            print("providers:", ", ".join(data.get("providers", [])))
        except urllib.error.URLError as e:
            print(f"Failed to reach server: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if not dataset_name:
        parser.error(
            "Set --dataset-name (e.g. scannet_scene0000_00) or --scannet-scene (e.g. scene0000_00)."
        )

    scanqa_path: Optional[Path] = args.scanqa_json
    if args.use_default_scanqa_val:
        scanqa_path = DEFAULT_SCANQA_VAL
        if not scanqa_path.is_file():
            parser.error(f"Default ScanQA val not found: {scanqa_path}")

    tasks: List[Dict[str, Any]] = []
    if args.question:
        tasks.append({"question": args.question})
    if args.questions_file:
        for q in load_questions(args.questions_file):
            tasks.append({"question": q})
    if scanqa_path is not None:
        if not scanqa_path.is_file():
            parser.error(f"ScanQA file not found: {scanqa_path}")
        tasks.extend(
            load_scanqa_for_scene(
                scanqa_path,
                args.scene_id,
                args.limit,
                bucket_filter=args.scanqa_bucket,
            )
        )
    if not tasks:
        if scanqa_path is not None and scanqa_path.is_file():
            n_scene = count_scanqa_rows_for_scene(scanqa_path, args.scene_id)
            if n_scene == 0:
                parser.error(
                    f"No ScanQA rows for scene_id={args.scene_id!r} in {scanqa_path}. "
                    "Splits differ: e.g. scene0000_00 has questions in ScanQA_v1.0_train.json but "
                    "not in ScanQA_v1.0_val.json. Use --scanqa-json pointing to a file that "
                    "contains this scene, or change --scene-id."
                )
            if args.scanqa_bucket:
                parser.error(
                    f"ScanQA has {n_scene} question(s) for scene_id={args.scene_id!r} in {scanqa_path}, "
                    f"but none match --scanqa-bucket {args.scanqa_bucket!r}. "
                    "Omit --scanqa-bucket or choose another bucket."
                )
        parser.error(
            "No questions to run: pass --scanqa-json PATH (or --use-default-scanqa-val), "
            "and/or --question, and/or --questions-file. "
            "If you used a multi-line shell command, a broken line continuation (space after \\, "
            "or only the last line executed) often drops --scanqa-json — use one line or fix \\."
        )

    out_f = None
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output_jsonl.open("a", encoding="utf-8")

    json_entries: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    eval_counters: Dict[str, int] = {
        "text_gt": 0,
        "text_pred": 0,
        "text_top1_ok": 0,
        "text_top10_ok": 0,
        "bbox_gt": 0,
        "bbox_top1_ok": 0,
        "bbox_any_ok": 0,
    }

    exit_code = 0
    try:
        for idx, item in enumerate(tasks):
            q = item["question"]
            row: Dict[str, Any] = {
                **{k: v for k, v in item.items() if k != "question"},
                "question": q,
                "dataset_name": dataset_name,
                "method": args.method,
            }
            try:
                result = run_search(base, dataset_name, args.method, q)
                row["ok"] = True
                row["response"] = result
                ev = compute_eval_metrics(item, result)
                row["eval"] = ev
                if ev["text"].get("ground_truth_available") or ev["bbox"].get(
                    "ground_truth_available"
                ):
                    _update_eval_counters(ev, eval_counters)
                n_ok += 1
                print("=" * 60)
                if item.get("question_id"):
                    print("question_id:", item.get("question_id"))
                print("question:", q)
                print_task_ground_truth(item)
                print_result(result, args.show_storage_bbox)
                if ev["text"].get("ground_truth_available") or ev["bbox"].get(
                    "ground_truth_available"
                ):
                    print("eval:", json.dumps(ev, ensure_ascii=False))
            except Exception as e:
                exit_code = 1
                row["ok"] = False
                row["error"] = str(e)
                n_fail += 1
                print("=" * 60)
                print("question:", q)
                print_task_ground_truth(item)
                print("ERROR:", e, file=sys.stderr)
            if args.output_json:
                json_entries.append(make_json_question_entry(idx, item, row))
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        gt_metrics = build_ground_truth_metrics_summary(eval_counters)
        report: Dict[str, Any] = {
            "meta": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "method": args.method,
                "server": base,
                "scene_id": args.scene_id,
                "scanqa_json": str(scanqa_path) if scanqa_path is not None else None,
                "scanqa_bucket_filter": args.scanqa_bucket,
            },
            "summary": {
                "total_questions": len(tasks),
                "successful_requests": n_ok,
                "failed_requests": n_fail,
                "ground_truth_metrics": gt_metrics,
            },
            "questions": json_entries,
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote report: {args.output_json.resolve()}")

    if not args.no_eval_summary and (
        eval_counters["text_gt"] or eval_counters["bbox_gt"]
    ):
        print("\n" + "=" * 60)
        print("Ground-truth eval summary (ScanQA answers / object_ids on task)")
        if eval_counters["text_gt"]:
            n = eval_counters["text_gt"]
            print(
                f"  Text (n={n} with reference answers): "
                f"top1 accuracy={eval_counters['text_top1_ok'] / n:.3f}, "
                f"top10 recall={eval_counters['text_top10_ok'] / n:.3f} "
                f"(answer_selection from pool; {eval_counters['text_pred']}/{n} had a non-NONE prediction)"
            )
        if eval_counters["bbox_gt"]:
            n = eval_counters["bbox_gt"]
            print(
                f"  Bbox / component (n={n} with reference object_ids): "
                f"top-1 component hit={eval_counters['bbox_top1_ok'] / n:.3f}, "
                f"any retrieved hit={eval_counters['bbox_any_ok'] / n:.3f}"
            )
        print("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
