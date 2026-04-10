#!/usr/bin/env python3
"""
Build per-category sets of unique answer strings from ScanQA JSON (array of items
with `question`, `answers` list, etc.).

Category assignment uses the same `bucket()` rules as
ScanQA/scripts/scanqa_simplified_categories.py (via scan-to-map search-server copy).

Outputs (under --output-dir):
  pools.json          — { "pools": { category: [str, ...] }, "meta": {...} }
  pools/<category>.txt — one answer per line (unique, sorted)
  summary.txt         — counts per category

Use with search-server: set env SCANQA_ANSWER_POOLS_JSON to pools.json

Example:
  python build_scanqa_category_answer_pools.py \\
      --input ../../ScanQA/data/qa/ScanQA_v1.0/ScanQA_v1.0_train.json \\
      --output-dir ../../search-server/data/scanqa_answer_pools
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

# Import bucket from search-server utils (single source of truth)
_REPO = Path(__file__).resolve().parents[1]
_SS_UTILS = _REPO / "search-server" / "utils"
if str(_SS_UTILS) not in sys.path:
    sys.path.insert(0, str(_SS_UTILS))

from scanqa_category import CATEGORY_ORDER, bucket  # noqa: E402


def _unique_strings(answers: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(answers, list):
        return out
    for a in answers:
        if isinstance(a, str) and a.strip():
            out.add(a.strip())
        elif a is not None:
            t = str(a).strip()
            if t:
                out.add(t)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ScanQA JSON array")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for pools.json and pools/*.txt",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected top-level JSON array")

    per_cat: Dict[str, Set[str]] = defaultdict(set)
    for row in records:
        if not isinstance(row, dict):
            continue
        q = row.get("question") or ""
        cat = bucket(str(q))
        per_cat[cat].update(_unique_strings(row.get("answers")))

    out_dir = args.output_dir.resolve()
    pool_dir = out_dir / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)

    pools_serial: Dict[str, List[str]] = {}
    for cat in CATEGORY_ORDER:
        uniq = sorted(per_cat.get(cat, set()), key=lambda s: s.lower())
        pools_serial[cat] = uniq
        (pool_dir / f"{cat}.txt").write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")

    meta = {
        "source_json": str(args.input.resolve()),
        "total_questions": len(records),
        "unique_answers_per_category": {c: len(pools_serial[c]) for c in CATEGORY_ORDER},
        "total_unique_answers": sum(len(pools_serial[c]) for c in CATEGORY_ORDER),
    }

    payload = {
        "meta": meta,
        "pools": pools_serial,
        "category_order": list(CATEGORY_ORDER),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pools.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"Source: {args.input}",
        "",
        f"{'Category':<22} {'Unique answers':>15}",
        "-" * 40,
    ]
    for c in CATEGORY_ORDER:
        lines.append(f"{c:<22} {len(pools_serial[c]):>15}")
    lines.append("")
    lines.append(f"Wrote: {out_dir / 'pools.json'}")
    lines.append(f"       {pool_dir}/<category>.txt")
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
