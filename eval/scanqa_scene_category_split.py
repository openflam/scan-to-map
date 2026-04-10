#!/usr/bin/env python3
"""
Split ScanQA val (or any ScanQA JSON array) by scene_id, then bucket questions
into simplified categories using the **same rules** as
ScanQA/scripts/scanqa_simplified_categories.py (`bucket()` order).

Categories (first match wins):
  color | number | location | object_retrieval | others

Outputs per category (JSONL + plain question list) so you can run
`semantic_demo_backend_eval.py` on one category at a time.

Note: Official ScanQA splits put **scene0000_00 only in train**, not val. Use
``--input .../ScanQA_v1.0_train.json`` for that scene, or pick a scene that
exists in val (see ``--list-scenes``).

Example:
  python scanqa_scene_category_split.py \\
      --input ../../ScanQA/data/scannet/ScanQA_v1.0_val.json \\
      --scene-id scene0011_00 \\
      --output-dir ./scanqa_split_scene0011

  python scanqa_scene_category_split.py \\
      --input ../../ScanQA/data/qa/ScanQA_v1.0/ScanQA_v1.0_train.json \\
      --scene-id scene0000_00 \\
      --output-dir ./scanqa_split_scene0000_train

Then:
  python semantic_demo_backend_eval.py \\
      --scannet-scene scene0000_00 \\
      --questions-file ./scanqa_split_scene0000_00/by_category/color_questions.txt \\
      --method BM25
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Must stay in sync with ScanQA/scripts/scanqa_simplified_categories.py
# ---------------------------------------------------------------------------


def bucket(question: str) -> str:
    """
    Assign exactly one of:
    color | number | location | object_retrieval | others
    """
    s = question.lower()

    if "what color" in s or "what is the color" in s or "what's the color" in s:
        return "color"

    if "how many" in s:
        return "number"

    if "what side" in s or re.search(r"\bwhere\b", s):
        return "location"

    object_phrases = (
        "what object",
        "what sits",
        "what can",
        "what are",
        "what is",
    )
    for phrase in object_phrases:
        if phrase in s:
            return "object_retrieval"

    return "others"


CATEGORY_ORDER = (
    "object_retrieval",
    "location",
    "color",
    "number",
    "others",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent


def _default_scanqa_json_path() -> Optional[Path]:
    """
    Walk parents and prefer ScanQA/data/scannet/ScanQA_v1.0_val.json, then
    ScanQA/data/qa/ScanQA_v1.0/ScanQA_v1.0_val.json (official layout).
    """
    preferred = (
        ("data", "scannet", "ScanQA_v1.0_val.json"),
        ("data", "qa", "ScanQA_v1.0", "ScanQA_v1.0_val.json"),
    )
    p = _EVAL_DIR
    for _ in range(8):
        for parts in preferred:
            candidate = p.joinpath("ScanQA", *parts)
            if candidate.is_file():
                return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


_DEFAULT_SCANQA = _default_scanqa_json_path()
DEFAULT_INPUT = _DEFAULT_SCANQA  # may be None; pass --input explicitly


def load_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter ScanQA by scene and split into simplified categories.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="ScanQA JSON array (default: discover ScanQA/data/scannet/ScanQA_v1.0_val.json upward from this script)",
    )
    parser.add_argument(
        "--scene-id",
        default="scene0000_00",
        help="Only include entries with this scene_id (default: scene0000_00)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for manifest, summary, by_category/ (required unless --list-scenes)",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List scene_id and question counts for the input file, then exit",
    )
    args = parser.parse_args()

    input_path = args.input or _DEFAULT_SCANQA
    if input_path is None or not input_path.is_file():
        raise FileNotFoundError(
            "Could not find ScanQA JSON. Pass --input or place ScanQA next to the repo "
            "(see ScanQA/data/scannet/ScanQA_v1.0_val.json)."
        )

    records = load_records(input_path)

    if args.list_scenes:
        cnt = Counter(str(r.get("scene_id", "")) for r in records)
        print(f"Source: {input_path.resolve()}")
        print(f"Total questions: {len(records)}")
        print(f"Unique scenes: {len(cnt)}")
        for sid, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {sid}\t{n}")
        return

    if not args.output_dir:
        parser.error("--output-dir is required (unless using --list-scenes)")

    scene_rows = [r for r in records if r.get("scene_id") == args.scene_id]

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        q = row.get("question") or ""
        cat = bucket(str(q))
        entry = dict(row)
        entry["_category"] = cat
        by_cat[cat].append(entry)

    out_dir = args.output_dir.resolve()
    cat_dir = out_dir / "by_category"
    cat_dir.mkdir(parents=True, exist_ok=True)

    files_per_category: Dict[str, Dict[str, str]] = {}

    for cat in CATEGORY_ORDER:
        rows = by_cat.get(cat, [])
        jsonl_path = cat_dir / f"{cat}.jsonl"
        txt_path = cat_dir / f"{cat}_questions.txt"
        files_per_category[cat] = {
            "jsonl": str(jsonl_path),
            "questions_txt": str(txt_path),
        }
        # Full records, one JSON object per line (includes _category, original fields)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # One question per line (for semantic_demo_backend_eval --questions-file)
        with txt_path.open("w", encoding="utf-8") as f:
            for row in rows:
                q = row.get("question", "")
                f.write(q.replace("\n", " ").strip() + "\n")

    notes: List[str] = []
    if len(scene_rows) == 0:
        notes.append(
            "No questions for this scene_id in this file. "
            "scene0000_00 is only in ScanQA_v1.0_train.json in the official split, not val."
        )

    summary = {
        "source_json": str(input_path.resolve()),
        "scene_id": args.scene_id,
        "total_in_file": len(records),
        "total_for_scene": len(scene_rows),
        "by_category": {c: len(by_cat.get(c, [])) for c in CATEGORY_ORDER},
        "notes": notes,
    }

    manifest = {
        **summary,
        "categories_order": list(CATEGORY_ORDER),
        "output": {
            "by_category_dir": str(cat_dir),
            "files_per_category": files_per_category,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Human-readable report
    lines = [
        f"Source: {input_path}",
        f"Scene: {args.scene_id}",
        f"Questions for scene: {len(scene_rows)}",
        "",
        f"{'Category':<22} {'Count':>8}",
        "-" * 32,
    ]
    for c in CATEGORY_ORDER:
        lines.append(f"{c:<22} {len(by_cat.get(c, [])):>8}")
    lines.append("")
    lines.append(f"Wrote: {out_dir / 'manifest.json'}")
    lines.append(f"       {cat_dir}/{{color,number,location,object_retrieval,others}}.jsonl")
    lines.append(
        f"       {cat_dir}/{{...}}_questions.txt"
    )
    if notes:
        lines.append("Notes:")
        for n in notes:
            lines.append(f"  - {n}")
        lines.append("")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
