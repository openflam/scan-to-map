#!/usr/bin/env python3
"""Run the split/merge component sensitivity experiment end to end."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import socket
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_RATES = ("0", "0.05", "0.10", "0.20", "0.30")
DEFAULT_SEEDS = (0, 1)
DEFAULT_CONFIG = "search_dist_around_image_exec"
DEFAULT_CAPTION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_METRICS = (
    "AI-Judge",
    "F1 Score",
    "Recall",
    "Source F1 Score",
    "Spatial IoU",
    "Source Recall",
    "Topology Ceiling F1",
)
CONFIGS = (
    "no_tools",
    "search_only",
    "search_distance",
    "search_distance_around",
    "search_dist_around_image",
    "search_dist_around_image_exec",
    "only_exec",
)
REQUIRED_MODULES = (
    "litellm",
    "matplotlib",
    "nltk",
    "numpy",
    "psycopg2",
    "requests",
)
CAPTION_REQUIRED_MODULES = ("PIL", "transformers", "vllm")


def parse_rate(raw_value: str) -> Decimal:
    try:
        value = Decimal(raw_value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"Invalid event rate: {raw_value}") from exc
    if not value.is_finite() or value < 0 or value >= Decimal("0.5"):
        raise argparse.ArgumentTypeError("Event rates must be in [0, 0.5).")
    if value != value.quantize(Decimal("0.0001")):
        raise argparse.ArgumentTypeError(
            "Event rates may have at most four digits after the decimal point."
        )
    return value


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str]) -> None:
    print(f"$ {command_text(command)}", flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_question_files(
    benchmark_data_dir: Path,
    dataset_prefix: str,
) -> list[Path]:
    question_files: list[Path] = []
    for path in sorted(benchmark_data_dir.glob("*.json")):
        record = read_json(path)
        dataset_name = record.get("dataset_name")
        if isinstance(dataset_name, str) and dataset_name.startswith(dataset_prefix):
            question_files.append(path)
    if not question_files:
        raise ValueError(
            f"No questions with dataset prefix {dataset_prefix!r} found in "
            f"{benchmark_data_dir}."
        )
    return question_files


def expected_condition_count(rates: list[Decimal], seeds: list[int]) -> int:
    return 1 + 2 * len([rate for rate in rates if rate != 0]) * len(seeds)


def load_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("experiment") != "split_merge_components":
        raise ValueError(f"Unexpected experiment type in {manifest_path}.")
    return manifest


def validate_resume_manifest(
    manifest: dict[str, Any],
    benchmark_data_dir: Path,
    outputs_dir: Path,
    rates: list[Decimal],
    seeds: list[int],
    recreate_crops_captions: bool,
    projection_min_fraction: float,
    captioner_type: str,
    caption_model: str,
    caption_device: int,
    caption_n_images: int,
    caption_batch_size: int,
    max_crops_per_component: int,
    scannetpp_data_root: Path | None,
) -> None:
    sampling = manifest.get("sampling", {})
    manifest_rates = sorted(Decimal(str(value)) for value in sampling.get("rates", []))
    manifest_seeds = sorted(int(value) for value in sampling.get("seeds", []))
    if manifest_rates != rates:
        raise ValueError(
            f"Existing manifest rates {manifest_rates} do not match {rates}."
        )
    if manifest_seeds != seeds:
        raise ValueError(
            f"Existing manifest seeds {manifest_seeds} do not match {seeds}."
        )
    if sampling.get("operations") != ["split", "merge"]:
        raise ValueError("Existing manifest does not contain both split and merge operations.")
    if Path(manifest["source_benchmark_data_dir"]).resolve() != benchmark_data_dir:
        raise ValueError("Existing manifest uses a different benchmark data directory.")
    if Path(manifest["outputs_dir"]).resolve() != outputs_dir:
        raise ValueError("Existing manifest uses a different outputs directory.")
    artifact_protocol = manifest.get("artifact_protocol", {})
    if bool(artifact_protocol.get("recreate_crops_captions", False)) != recreate_crops_captions:
        raise ValueError("Existing manifest uses a different crop/caption protocol.")
    if recreate_crops_captions:
        expected = {
            "projection_min_fraction": projection_min_fraction,
            "captioner_type": captioner_type,
            "caption_model": caption_model,
            "caption_device": caption_device,
            "caption_n_images": caption_n_images,
            "caption_batch_size": caption_batch_size,
            "max_crops_per_component": max_crops_per_component,
        }
        if scannetpp_data_root is not None:
            expected["scannetpp_data_root"] = str(scannetpp_data_root)
        mismatched = {
            key: (artifact_protocol.get(key), value)
            for key, value in expected.items()
            if artifact_protocol.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"Existing manifest uses different recreation settings: {mismatched}"
            )


def missing_runtime_modules(recreate_crops_captions: bool = False) -> list[str]:
    modules = REQUIRED_MODULES + (
        CAPTION_REQUIRED_MODULES if recreate_crops_captions else ()
    )
    return [name for name in modules if importlib.util.find_spec(name) is None]


def check_search_server() -> None:
    try:
        with socket.create_connection(("127.0.0.1", 5000), timeout=3):
            return
    except OSError as exc:
        raise RuntimeError(
            "The search server is not reachable at localhost:5000. Run this "
            "script in the search-server container after Docker Compose is up."
        ) from exc


def prepare_command(
    script_dir: Path,
    benchmark_data_dir: Path,
    outputs_dir: Path,
    experiment_dir: Path,
    rates: list[Decimal],
    seeds: list[int],
    dataset_prefix: str,
    alias_prefix: str,
    dry_run: bool,
    recreate_crops_captions: bool,
    projection_min_fraction: float,
    captioner_type: str,
    caption_model: str,
    caption_device: int,
    caption_n_images: int,
    caption_batch_size: int,
    max_crops_per_component: int,
    scannetpp_data_root: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(script_dir / "prepare.py"),
        "--benchmark-data-dir",
        str(benchmark_data_dir),
        "--outputs-dir",
        str(outputs_dir),
        "--experiment-dir",
        str(experiment_dir),
        "--rates",
        *(format(rate, "f") for rate in rates),
        "--seeds",
        *(str(seed) for seed in seeds),
        "--dataset-prefix",
        dataset_prefix,
        "--alias-prefix",
        alias_prefix,
    ]
    if recreate_crops_captions:
        command.extend(
            [
                "--recreate-crops-captions",
                "--projection-min-fraction",
                str(projection_min_fraction),
                "--captioner-type",
                captioner_type,
                "--caption-model",
                caption_model,
                "--caption-device",
                str(caption_device),
                "--caption-n-images",
                str(caption_n_images),
                "--caption-batch-size",
                str(caption_batch_size),
                "--max-crops-per-component",
                str(max_crops_per_component),
            ]
        )
        if scannetpp_data_root is not None:
            command.extend(["--scannetpp-data-root", str(scannetpp_data_root)])
    if dry_run:
        command.append("--dry-run")
    return command


def execution_command(
    script_dir: Path,
    experiment_dir: Path,
    configs: list[str],
    model: str,
    workers: int,
    request_timeout: float,
    disable_ai_judge: bool,
    resume: bool,
    reuse_existing_db_tables: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(script_dir / "run.py"),
        "--experiment-dir",
        str(experiment_dir),
        "--stage",
        "all",
        "--configs",
        *configs,
        "--model",
        model,
        "--workers",
        str(workers),
        "--request-timeout",
        str(request_timeout),
    ]
    if disable_ai_judge:
        command.append("--disable-ai-judge")
    if resume:
        command.append("--resume")
    if reuse_existing_db_tables:
        command.append("--reuse-existing-db-tables")
    return command


def summary_command(
    script_dir: Path,
    experiment_dir: Path,
    analysis_dir: Path,
    metrics: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(script_dir / "summarize.py"),
        "--experiment-dir",
        str(experiment_dir),
        "--out-dir",
        str(analysis_dir),
        "--metrics",
        *metrics,
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--bootstrap-seed",
        str(bootstrap_seed),
    ]


def print_query_plan(
    num_questions: int,
    num_conditions: int,
    configs: list[str],
    model: str,
    disable_ai_judge: bool,
) -> None:
    answer_queries = num_questions * num_conditions * len(configs)
    judge_queries = 0 if disable_ai_judge else answer_queries
    min_agent_turns = answer_queries
    max_agent_turns = 8 * answer_queries
    print("\nModel-query plan")
    print(f"  Benchmark questions: {num_questions}")
    print(f"  Conditions: {num_conditions} (1 baseline + split and merge conditions)")
    print(f"  Tool configurations: {len(configs)} ({', '.join(configs)})")
    print(f"  {model} top-level benchmark queries: {answer_queries}")
    print(
        f"  {model} API turns including the tool loop: "
        f"{min_agent_turns}-{max_agent_turns} (maximum 8 per query)"
    )
    print(
        "  GPT-4o-mini judge queries: "
        + ("0 (disabled)" if disable_ai_judge else f"up to {judge_queries}")
    )
    print(
        f"  Total possible model API calls: "
        f"{min_agent_turns + judge_queries}-{max_agent_turns + judge_queries}"
    )


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, execute, score, and plot split/merge sensitivity."
    )
    parser.add_argument(
        "--benchmark-data-dir",
        type=Path,
        default=repo_root / "benchmark" / "data",
    )
    parser.add_argument("--outputs-dir", type=Path, default=repo_root / "outputs")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "split_merge_components",
    )
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument(
        "--rates",
        nargs="+",
        type=parse_rate,
        default=[parse_rate(rate) for rate in DEFAULT_RATES],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--dataset-prefix", default="scannetpp_")
    parser.add_argument("--alias-prefix", default="split_merge")
    parser.add_argument("--scannetpp-data-root", type=Path, default=None)
    parser.add_argument(
        "--recreate-crops-captions",
        action="store_true",
        help="Reproject, recrop, and recaption affected split/merge components.",
    )
    parser.add_argument("--projection-min-fraction", type=float, default=0.3)
    parser.add_argument("--captioner-type", default="vllm")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--caption-device", type=int, default=0)
    parser.add_argument("--caption-n-images", type=int, default=1)
    parser.add_argument("--caption-batch-size", type=int, default=512)
    parser.add_argument("--max-crops-per-component", type=int, default=5)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIGS,
        default=[DEFAULT_CONFIG],
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--disable-ai-judge", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and print exact query counts without writing or calling models.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-existing-db-tables", action="store_true")
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_dir = Path(__file__).resolve().parent
    parser = build_parser(repo_root)
    args = parser.parse_args()
    rates = sorted(set(args.rates))
    seeds = sorted(set(args.seeds))
    configs = list(dict.fromkeys(args.configs))
    if Decimal(0) not in rates:
        parser.error("--rates must include 0 so the study has a clean baseline")
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("--seeds must contain at least one non-negative integer")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if not 0 <= args.projection_min_fraction <= 1:
        parser.error("--projection-min-fraction must be in [0, 1]")
    if args.caption_device < 0:
        parser.error("--caption-device must be non-negative")
    if args.caption_n_images < 1 or args.caption_batch_size < 1:
        parser.error("Caption image and batch counts must be positive")
    if args.max_crops_per_component < 1:
        parser.error("--max-crops-per-component must be positive")

    benchmark_data_dir = args.benchmark_data_dir.resolve()
    outputs_dir = args.outputs_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    scannetpp_data_root = (
        args.scannetpp_data_root.resolve() if args.scannetpp_data_root else None
    )
    analysis_dir = (
        args.analysis_dir.resolve() if args.analysis_dir else experiment_dir / "analysis"
    )
    if not benchmark_data_dir.is_dir():
        parser.error(f"Benchmark data directory does not exist: {benchmark_data_dir}")
    if not outputs_dir.is_dir():
        parser.error(f"Outputs directory does not exist: {outputs_dir}")

    question_files = discover_question_files(benchmark_data_dir, args.dataset_prefix)
    num_conditions = expected_condition_count(rates, seeds)
    print_query_plan(
        len(question_files),
        num_conditions,
        configs,
        args.model,
        args.disable_ai_judge,
    )

    prepare = prepare_command(
        script_dir,
        benchmark_data_dir,
        outputs_dir,
        experiment_dir,
        rates,
        seeds,
        args.dataset_prefix,
        args.alias_prefix,
        args.dry_run,
        args.recreate_crops_captions,
        args.projection_min_fraction,
        args.captioner_type,
        args.caption_model,
        args.caption_device,
        args.caption_n_images,
        args.caption_batch_size,
        args.max_crops_per_component,
        scannetpp_data_root,
    )
    execute = execution_command(
        script_dir,
        experiment_dir,
        configs,
        args.model,
        args.workers,
        args.request_timeout,
        args.disable_ai_judge,
        args.resume,
        args.reuse_existing_db_tables,
    )
    metrics = [
        metric
        for metric in list(dict.fromkeys(args.metrics))
        if not (args.disable_ai_judge and metric == "AI-Judge")
    ]
    if not metrics:
        parser.error("At least one non-AI-judge metric is required when disabled")
    summarize = summary_command(
        script_dir,
        experiment_dir,
        analysis_dir,
        metrics,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    if args.dry_run:
        if experiment_dir.exists() or experiment_dir.is_symlink():
            if not args.resume:
                raise FileExistsError(
                    f"Experiment directory already exists: {experiment_dir}. "
                    "Use --resume only if it belongs to this design."
                )
            validate_resume_manifest(
                load_manifest(experiment_dir),
                benchmark_data_dir,
                outputs_dir,
                rates,
                seeds,
                args.recreate_crops_captions,
                args.projection_min_fraction,
                args.captioner_type,
                args.caption_model,
                args.caption_device,
                args.caption_n_images,
                args.caption_batch_size,
                args.max_crops_per_component,
                scannetpp_data_root,
            )
            print("\nExisting prepared experiment validated for resume.")
        else:
            print("\nPreparation validation")
            run_command(prepare)
        missing = missing_runtime_modules(
            args.recreate_crops_captions
            and not (experiment_dir.exists() or experiment_dir.is_symlink())
        )
        if missing:
            dependency_hint = (
                " Prepare aliases in the segment3d environment and rerun with "
                "--resume."
                if args.recreate_crops_captions
                else " Rebuild the search-server image before running."
            )
            print(
                "\nFull-run dependency warning: missing Python modules: "
                f"{', '.join(missing)}.{dependency_hint}"
            )
        print("\nCommands that would run after preparation")
        print(f"$ {command_text(execute)}")
        print(f"$ {command_text(summarize)}")
        print("\nDry run complete. No files, database tables, or model queries were created.")
        return

    if analysis_dir.exists() or analysis_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {analysis_dir}")
    experiment_exists = experiment_dir.exists() or experiment_dir.is_symlink()
    missing = missing_runtime_modules(
        args.recreate_crops_captions and not experiment_exists
    )
    if missing:
        recreation_hint = (
            " Prepare the experiment with --recreate-crops-captions in the "
            "segment3d environment, then rerun this command with --resume."
            if args.recreate_crops_captions and not experiment_exists
            else " Rebuild the search-server image before running."
        )
        raise RuntimeError(
            "Missing Python modules required for the full workflow: "
            f"{', '.join(missing)}.{recreation_hint}"
        )
    check_search_server()
    if experiment_exists:
        if not args.resume:
            raise FileExistsError(
                f"Refusing to overwrite experiment directory: {experiment_dir}"
            )
        validate_resume_manifest(
            load_manifest(experiment_dir),
            benchmark_data_dir,
            outputs_dir,
            rates,
            seeds,
            args.recreate_crops_captions,
            args.projection_min_fraction,
            args.captioner_type,
            args.caption_model,
            args.caption_device,
            args.caption_n_images,
            args.caption_batch_size,
            args.max_crops_per_component,
            scannetpp_data_root,
        )
        print(f"Resuming prepared experiment at {experiment_dir}")
    else:
        run_command(prepare)
    run_command(execute)
    run_command(summarize)
    print("\nSplit/merge component sensitivity experiment complete.")
    print(f"  Experiment artifacts: {experiment_dir}")
    print(f"  Tables and plots: {analysis_dir}")


if __name__ == "__main__":
    main()
