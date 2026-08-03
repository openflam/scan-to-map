#!/usr/bin/env python3
"""Summarize currently available split/merge results with completeness metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize import DEFAULT_METRICS, summarize


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize available split/merge metrics without requiring every "
            "non-baseline condition to be complete."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "split_merge_components",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = build_parser(repo_root)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    experiment_dir = args.experiment_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else experiment_dir / "analysis_partial"
    )
    summarize(
        experiment_dir,
        out_dir,
        list(dict.fromkeys(args.metrics)),
        args.bootstrap_samples,
        args.bootstrap_seed,
        allow_incomplete=True,
    )
    print(
        "Partial estimates use only available results; inspect "
        "condition_completeness.csv before interpreting them."
    )


if __name__ == "__main__":
    main()
