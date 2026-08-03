#!/usr/bin/env python3
"""Run prepared split/merge conditions without touching legacy benchmark paths."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "search_dist_around_image_exec"


def read_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Experiment manifest not found: {manifest_path}. Run prepare.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("experiment") != "split_merge_components":
        raise ValueError(f"Unexpected experiment type in {manifest_path}.")
    return manifest


def resolve_experiment_path(experiment_dir: Path, relative_path: str) -> Path:
    path = (experiment_dir / relative_path).resolve()
    if path != experiment_dir and experiment_dir not in path.parents:
        raise ValueError(f"Manifest path escapes experiment directory: {relative_path}")
    return path


def condition_subset(
    manifest: dict[str, Any],
    requested_ids: list[str] | None,
) -> list[dict[str, Any]]:
    conditions = manifest.get("conditions", [])
    if not requested_ids:
        return conditions
    by_id = {condition["condition_id"]: condition for condition in conditions}
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise ValueError(f"Unknown condition IDs: {', '.join(missing)}")
    return [by_id[condition_id] for condition_id in requested_ids]


def perturbed_dataset_names(conditions: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            perturbed_name
            for condition in conditions
            for source_name, perturbed_name in condition["dataset_mapping"].items()
            if perturbed_name != source_name
        }
    )


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str]) -> None:
    print(f"$ {command_text(command)}", flush=True)
    subprocess.run(command, check=True)


def answer_command(
    repo_root: Path,
    questions_dir: Path,
    results_dir: Path,
    configs: list[str],
    model: str,
    workers: int,
    request_timeout: float,
    fail_on_existing: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "benchmark-eval" / "gen_answers.py"),
        "--data_dir",
        str(questions_dir),
        "--out_dir",
        str(results_dir),
        "--configs",
        *configs,
        "--model",
        model,
        "--workers",
        str(workers),
        "--request_timeout",
        str(request_timeout),
        "--fail_on_query_error",
        "--split_merge_flex",
    ]
    if fail_on_existing:
        command.append("--fail_on_existing")
    return command


def metrics_command(
    repo_root: Path,
    results_dir: Path,
    metrics_dir: Path,
    disable_ai_judge: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "benchmark-eval" / "metrics.py"),
        "--results_dir",
        str(results_dir),
        "--metrics_dir",
        str(metrics_dir),
    ]
    if disable_ai_judge:
        command.append("--disable_ai_judge")
    return command


def preflight_output_dirs(
    experiment_dir: Path,
    conditions: list[dict[str, Any]],
    key: str,
    resume: bool,
) -> None:
    if resume:
        return
    existing = [
        resolve_experiment_path(experiment_dir, condition[key])
        for condition in conditions
        if resolve_experiment_path(experiment_dir, condition[key]).exists()
    ]
    if existing:
        preview = "\n".join(str(path) for path in existing[:10])
        raise FileExistsError(
            "Refusing to mix a new run with existing condition outputs. Use "
            f"--resume to fill only missing files:\n{preview}"
        )


def load_database_tables(
    repo_root: Path,
    dataset_names: list[str],
    reuse_existing: bool,
) -> None:
    if not dataset_names:
        print("No perturbed dataset tables need to be loaded.")
        return
    sys.path.insert(0, str(repo_root / "search-server"))
    from spatial_db import database

    existing = [
        dataset_name
        for dataset_name in dataset_names
        if database.check_dataset_exists(dataset_name)
    ]
    if existing and not reuse_existing:
        raise RuntimeError(
            "Perturbed database tables already exist. Refusing to invoke the "
            "table-replacing loader: "
            f"{', '.join(existing[:10])}. Use --reuse-existing-db-tables to trust them."
        )
    to_create = [name for name in dataset_names if name not in existing]
    if to_create:
        run_command(
            [
                sys.executable,
                str(repo_root / "search-server" / "spatial_db" / "create_tables.py"),
                *to_create,
            ]
        )
    if existing:
        print(f"Reusing {len(existing)} existing perturbed database tables.")


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated split/merge component sensitivity experiment."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "split_merge_components",
    )
    parser.add_argument(
        "--stage",
        choices=("commands", "load-db", "answers", "metrics", "all"),
        default="commands",
        help="The default prints commands without changing the database or calling models.",
    )
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--configs", nargs="+", default=[DEFAULT_CONFIG])
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--disable-ai-judge", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing result/metric files and fill only missing files.",
    )
    parser.add_argument(
        "--reuse-existing-db-tables",
        action="store_true",
        help="Trust matching perturbed PostGIS tables that already exist.",
    )
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = build_parser(repo_root)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")

    experiment_dir = args.experiment_dir.resolve()
    manifest = read_manifest(experiment_dir)
    conditions = condition_subset(manifest, args.conditions)
    dataset_names = perturbed_dataset_names(conditions)

    answer_commands: list[list[str]] = []
    metric_commands: list[list[str]] = []
    for condition in conditions:
        questions_dir = resolve_experiment_path(experiment_dir, condition["questions_dir"])
        results_dir = resolve_experiment_path(experiment_dir, condition["results_dir"])
        metrics_dir = resolve_experiment_path(experiment_dir, condition["metrics_dir"])
        answer_commands.append(
            answer_command(
                repo_root,
                questions_dir,
                results_dir,
                args.configs,
                args.model,
                args.workers,
                args.request_timeout,
                fail_on_existing=not args.resume,
            )
        )
        metric_commands.append(
            metrics_command(repo_root, results_dir, metrics_dir, args.disable_ai_judge)
        )

    if args.stage == "commands":
        if dataset_names:
            print("# Load only the new perturbed scene-memory tables")
            print(
                command_text(
                    [
                        sys.executable,
                        str(repo_root / "search-server" / "spatial_db" / "create_tables.py"),
                        *dataset_names,
                    ]
                )
            )
        print("\n# Generate answers")
        for command in answer_commands:
            print(command_text(command))
        print("\n# Compute metrics")
        for command in metric_commands:
            print(command_text(command))
        return

    if args.stage in {"answers", "all"}:
        preflight_output_dirs(experiment_dir, conditions, "results_dir", args.resume)
    if args.stage in {"metrics", "all"}:
        preflight_output_dirs(experiment_dir, conditions, "metrics_dir", args.resume)
    if args.stage in {"load-db", "all"}:
        load_database_tables(
            repo_root,
            dataset_names,
            reuse_existing=args.reuse_existing_db_tables,
        )
    if args.stage in {"answers", "all"}:
        for command in answer_commands:
            run_command(command)
    if args.stage in {"metrics", "all"}:
        for condition, command in zip(conditions, metric_commands):
            results_dir = resolve_experiment_path(experiment_dir, condition["results_dir"])
            if not results_dir.is_dir():
                raise FileNotFoundError(
                    f"Results missing for {condition['condition_id']}: {results_dir}"
                )
            run_command(command)


if __name__ == "__main__":
    main()
