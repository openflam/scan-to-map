#!/usr/bin/env python3
"""Prepare non-destructive scene memories for missing-component sensitivity tests.

The source scene memories and benchmark questions are never modified. Each non-zero
condition receives a new dataset alias under outputs/ and rewritten question files
under a new experiment directory. Component deletion is synchronized across
captions, bounding boxes, the crop manifest, and optional connected components.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


DEFAULT_RATES = ("0", "0.05", "0.10", "0.20", "0.30")
DEFAULT_SEEDS = (0, 1)
COMPONENT_TAG_RE = re.compile(r"<component_(\d+)>")
SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9_]+$")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def parse_rate(raw_value: str) -> Decimal:
    try:
        value = Decimal(raw_value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"Invalid deletion rate: {raw_value}") from exc

    if not value.is_finite() or value < 0 or value >= 1:
        raise argparse.ArgumentTypeError(
            "Deletion rates must be finite fractions in the interval [0, 1)."
        )
    if value != value.quantize(Decimal("0.0001")):
        raise argparse.ArgumentTypeError(
            "Deletion rates may have at most four digits after the decimal point."
        )
    return value


def rate_basis_points(rate: Decimal) -> int:
    return int((rate * Decimal(10_000)).to_integral_value())


def condition_id(rate: Decimal, seed: int | None) -> str:
    basis_points = rate_basis_points(rate)
    if rate == 0:
        return "drop_0000bp_baseline"
    if seed is None:
        raise ValueError("A non-zero deletion rate requires a seed.")
    return f"drop_{basis_points:04d}bp_seed_{seed:03d}"


def dataset_alias(
    alias_prefix: str,
    dataset_name: str,
    rate: Decimal,
    seed: int,
) -> str:
    alias = (
        f"{alias_prefix}_{dataset_name}_"
        f"bp{rate_basis_points(rate):04d}_s{seed:03d}"
    )
    if len(alias) > 63:
        raise ValueError(
            f"Dataset alias exceeds PostgreSQL's 63-character identifier limit: {alias}"
        )
    return alias


def stable_component_order(dataset_name: str, component_ids: set[int], seed: int) -> list[int]:
    def sort_key(component_id: int) -> bytes:
        payload = (
            f"missing-components-v1\0{seed}\0{dataset_name}\0{component_id}"
        ).encode("utf-8")
        return hashlib.sha256(payload).digest()

    return sorted(component_ids, key=sort_key)


def deletion_count(num_components: int, rate: Decimal) -> int:
    count = int(
        (Decimal(num_components) * rate).to_integral_value(rounding=ROUND_HALF_UP)
    )
    return min(count, max(0, num_components - 1))


def unique_id_map(
    records: list[dict[str, Any]],
    id_field: str,
    source_path: Path,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if id_field not in record:
            raise ValueError(f"{source_path} has a record without {id_field}.")
        component_id = int(record[id_field])
        if component_id in result:
            raise ValueError(
                f"{source_path} contains duplicate component ID {component_id}."
            )
        result[component_id] = record
    return result


def load_scene_memory(outputs_dir: Path, dataset_name: str) -> dict[str, Any]:
    scene_dir = outputs_dir / dataset_name
    captions_path = scene_dir / "component_captions.json"
    bboxes_path = scene_dir / "bbox_corners.json"
    manifest_path = scene_dir / "crops" / "manifest.json"

    missing = [
        path for path in (captions_path, bboxes_path, manifest_path) if not path.is_file()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Incomplete scene memory for {dataset_name}: {missing_text}")

    captions = read_json(captions_path)
    bboxes = read_json(bboxes_path)
    manifest = read_json(manifest_path)
    if not isinstance(captions, list) or not isinstance(bboxes, list):
        raise ValueError(f"Caption and bounding-box files must be lists for {dataset_name}.")
    if not isinstance(manifest, dict):
        raise ValueError(f"Crop manifest must be an object for {dataset_name}.")

    caption_map = unique_id_map(captions, "component_id", captions_path)
    bbox_map = unique_id_map(bboxes, "connected_comp_id", bboxes_path)
    caption_ids = set(caption_map)
    bbox_ids = set(bbox_map)
    missing_bbox_ids = caption_ids - bbox_ids
    if missing_bbox_ids:
        raise ValueError(
            f"Bounding-box data for {dataset_name} is missing searchable component "
            f"IDs {sorted(missing_bbox_ids)}."
        )

    try:
        manifest_ids = {int(component_id) for component_id in manifest}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Crop manifest keys must be integer component IDs for {dataset_name}."
        ) from exc

    missing_manifest_ids = caption_ids - manifest_ids
    if missing_manifest_ids:
        raise ValueError(
            f"Crop manifest for {dataset_name} is missing component IDs "
            f"{sorted(missing_manifest_ids)}."
        )

    crops_dir = scene_dir / "crops"
    for component_id in caption_ids:
        manifest_entry = manifest[str(component_id)]
        if manifest_entry.get("crops") and not (
            crops_dir / f"component_{component_id}"
        ).is_dir():
            raise FileNotFoundError(
                f"Crop directory missing for {dataset_name} component {component_id}."
            )

    return {
        "scene_dir": scene_dir,
        "captions": captions,
        "bboxes": bboxes,
        "manifest": manifest,
        "component_ids": caption_ids,
    }


def discover_questions(
    benchmark_data_dir: Path,
    dataset_prefix: str,
) -> list[tuple[Path, dict[str, Any]]]:
    questions: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(benchmark_data_dir.glob("*.json")):
        record = read_json(path)
        dataset_name = record.get("dataset_name")
        if isinstance(dataset_name, str) and dataset_name.startswith(dataset_prefix):
            questions.append((path, record))

    if not questions:
        raise ValueError(
            f"No questions with dataset prefix {dataset_prefix!r} found in "
            f"{benchmark_data_dir}."
        )
    return questions


def removed_reference_status(
    referenced_ids: set[int],
    removed_ids: set[int],
) -> tuple[list[int], str]:
    removed_references = sorted(referenced_ids & removed_ids)
    if not referenced_ids:
        return removed_references, "no_referenced_components"
    if not removed_references:
        return removed_references, "all_retained"
    if len(removed_references) == len(referenced_ids):
        return removed_references, "all_removed"
    return removed_references, "partially_removed"


def make_relative_symlink(source: Path, destination: Path) -> None:
    relative_source = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_source, target_is_directory=source.is_dir())


def materialize_scene_alias(
    outputs_dir: Path,
    dataset_name: str,
    alias: str,
    scene_memory: dict[str, Any],
    removed_ids: set[int],
    rate: Decimal,
    seed: int,
) -> dict[str, Any]:
    target_dir = outputs_dir / alias
    if target_dir.exists() or target_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite scene-memory alias: {target_dir}")

    retained_ids = scene_memory["component_ids"] - removed_ids
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{alias}.tmp-", dir=outputs_dir))
    try:
        filtered_captions = [
            record
            for record in scene_memory["captions"]
            if int(record["component_id"]) in retained_ids
        ]
        filtered_bboxes = [
            record
            for record in scene_memory["bboxes"]
            if int(record["connected_comp_id"]) in retained_ids
        ]
        filtered_manifest = {
            key: value
            for key, value in scene_memory["manifest"].items()
            if int(key) in retained_ids
        }

        write_json(temp_dir / "component_captions.json", filtered_captions)
        write_json(temp_dir / "bbox_corners.json", filtered_bboxes)
        crops_target = temp_dir / "crops"
        crops_target.mkdir()
        write_json(crops_target / "manifest.json", filtered_manifest)

        source_crops = scene_memory["scene_dir"] / "crops"
        for component_id in sorted(retained_ids):
            source_component_dir = source_crops / f"component_{component_id}"
            if source_component_dir.is_dir():
                make_relative_symlink(
                    source_component_dir,
                    crops_target / f"component_{component_id}",
                )

        source_mesh = scene_memory["scene_dir"] / "raw.glb"
        if source_mesh.is_file():
            make_relative_symlink(source_mesh, temp_dir / "raw.glb")

        source_components = scene_memory["scene_dir"] / "connected_components.json"
        if source_components.is_file():
            component_records = read_json(source_components)
            if not isinstance(component_records, list):
                raise ValueError(f"{source_components} must contain a list.")
            filtered_components = [
                record
                for record in component_records
                if int(record["connected_comp_id"]) in retained_ids
            ]
            write_json(
                temp_dir / "connected_components.json",
                filtered_components,
            )

        actual_rate = len(removed_ids) / len(scene_memory["component_ids"])
        perturbation = {
            "schema_version": 1,
            "perturbation": "missing_components",
            "sampling": "uniform_without_replacement_nested_sha256_v1",
            "source_dataset_name": dataset_name,
            "dataset_name": alias,
            "requested_deletion_rate": float(rate),
            "actual_deletion_rate": actual_rate,
            "seed": seed,
            "num_source_components": len(scene_memory["component_ids"]),
            "num_removed_components": len(removed_ids),
            "num_retained_components": len(retained_ids),
            "removed_component_ids": sorted(removed_ids),
        }
        write_json(temp_dir / "scene_memory_perturbation.json", perturbation)
        temp_dir.rename(target_dir)
        return perturbation
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def preflight_paths(
    experiment_dir: Path,
    outputs_dir: Path,
    aliases: set[str],
) -> None:
    if experiment_dir.exists() or experiment_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite experiment directory: {experiment_dir}"
        )
    for alias in sorted(aliases):
        alias_path = outputs_dir / alias
        if alias_path.exists() or alias_path.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite scene-memory alias: {alias_path}"
            )


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic missing-component scene-memory variants without "
            "modifying existing outputs or benchmark artifacts."
        )
    )
    parser.add_argument(
        "--benchmark-data-dir",
        type=Path,
        default=repo_root / "benchmark" / "data",
        help="Source benchmark question directory.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Source outputs directory and destination for new dataset aliases.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "missing_components",
        help=(
            "New experiment directory. It must not exist; benchmark/data and "
            "benchmark/plots are never written."
        ),
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=parse_rate,
        default=[parse_rate(rate) for rate in DEFAULT_RATES],
        help="Component-deletion fractions. Include 0 for the clean baseline.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds for non-zero deletion masks.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="scannetpp_",
        help="Only benchmark datasets with this prefix are included.",
    )
    parser.add_argument(
        "--alias-prefix",
        default="missing_components",
        help="Prefix for new dataset directories under outputs/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without creating anything.",
    )
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = build_parser(repo_root)
    args = parser.parse_args()

    rates = sorted(set(args.rates))
    seeds = sorted(set(args.seeds))
    if Decimal(0) not in rates:
        parser.error("--rates must include 0 so the study has a clean baseline")
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("--seeds must contain at least one non-negative integer")
    if not SAFE_PREFIX_RE.fullmatch(args.alias_prefix):
        parser.error("--alias-prefix may contain only letters, digits, and underscores")

    benchmark_data_dir = args.benchmark_data_dir.resolve()
    outputs_dir = args.outputs_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    if not benchmark_data_dir.is_dir():
        parser.error(f"Benchmark data directory does not exist: {benchmark_data_dir}")
    if not outputs_dir.is_dir():
        parser.error(f"Outputs directory does not exist: {outputs_dir}")
    if experiment_dir in {benchmark_data_dir, outputs_dir}:
        parser.error("Experiment directory cannot be a source directory")

    questions = discover_questions(benchmark_data_dir, args.dataset_prefix)
    dataset_names = sorted({record["dataset_name"] for _, record in questions})
    scene_memories = {
        dataset_name: load_scene_memory(outputs_dir, dataset_name)
        for dataset_name in dataset_names
    }

    for question_path, question_record in questions:
        dataset_name = question_record["dataset_name"]
        referenced_ids = {
            int(component_id)
            for component_id in COMPONENT_TAG_RE.findall(
                question_record.get("expected_answer", "")
            )
        }
        unknown_ids = referenced_ids - scene_memories[dataset_name]["component_ids"]
        if unknown_ids:
            raise ValueError(
                f"{question_path} references unknown components {sorted(unknown_ids)} "
                f"in {dataset_name}."
            )

    conditions: list[tuple[Decimal, int | None]] = [(Decimal(0), None)]
    conditions.extend(
        (rate, seed)
        for rate in rates
        if rate != 0
        for seed in seeds
    )

    aliases: set[str] = set()
    for rate, seed in conditions:
        if rate == 0:
            continue
        assert seed is not None
        for dataset_name in dataset_names:
            aliases.add(
                dataset_alias(args.alias_prefix, dataset_name, rate, seed)
            )
    preflight_paths(experiment_dir, outputs_dir, aliases)

    num_model_queries = len(questions) * len(conditions)
    print(
        f"Validated {len(questions)} questions across {len(dataset_names)} scenes; "
        f"{len(conditions)} conditions and {len(aliases)} new scene-memory aliases."
    )
    print(
        f"Default full experiment: {num_model_queries} top-level gpt-5.4 queries "
        "(one model and one tool configuration)."
    )
    print(
        "Tool-calling may require multiple model API turns per top-level query."
    )
    if args.dry_run:
        for rate, seed in conditions:
            print(
                f"  {condition_id(rate, seed)}: "
                f"rate={float(rate):.4f}, seed={seed}"
            )
        return

    condition_records: list[dict[str, Any]] = []
    experiment_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_experiment_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{experiment_dir.name}.tmp-",
            dir=experiment_dir.parent,
        )
    )

    try:
        for rate, seed in conditions:
            current_condition_id = condition_id(rate, seed)
            questions_dir = (
                temp_experiment_dir / "conditions" / current_condition_id / "questions"
            )
            questions_dir.mkdir(parents=True)

            dataset_mapping: dict[str, str] = {}
            scene_records: dict[str, dict[str, Any]] = {}
            removed_by_dataset: dict[str, set[int]] = {}

            for dataset_name, scene_memory in scene_memories.items():
                if rate == 0:
                    removed_ids: set[int] = set()
                    perturbed_dataset_name = dataset_name
                    perturbation = {
                        "schema_version": 1,
                        "perturbation": "missing_components",
                        "sampling": "clean_baseline",
                        "source_dataset_name": dataset_name,
                        "dataset_name": dataset_name,
                        "requested_deletion_rate": 0.0,
                        "actual_deletion_rate": 0.0,
                        "seed": None,
                        "num_source_components": len(scene_memory["component_ids"]),
                        "num_removed_components": 0,
                        "num_retained_components": len(scene_memory["component_ids"]),
                        "removed_component_ids": [],
                    }
                else:
                    assert seed is not None
                    ordering = stable_component_order(
                        dataset_name,
                        scene_memory["component_ids"],
                        seed,
                    )
                    num_removed = deletion_count(
                        len(scene_memory["component_ids"]),
                        rate,
                    )
                    removed_ids = set(ordering[:num_removed])
                    perturbed_dataset_name = dataset_alias(
                        args.alias_prefix,
                        dataset_name,
                        rate,
                        seed,
                    )
                    perturbation = materialize_scene_alias(
                        outputs_dir=outputs_dir,
                        dataset_name=dataset_name,
                        alias=perturbed_dataset_name,
                        scene_memory=scene_memory,
                        removed_ids=removed_ids,
                        rate=rate,
                        seed=seed,
                    )

                removed_by_dataset[dataset_name] = removed_ids
                dataset_mapping[dataset_name] = perturbed_dataset_name
                scene_records[dataset_name] = perturbation

            status_counts: dict[str, int] = {}
            for source_path, source_record in questions:
                dataset_name = source_record["dataset_name"]
                removed_ids = removed_by_dataset[dataset_name]
                referenced_ids = {
                    int(component_id)
                    for component_id in COMPONENT_TAG_RE.findall(
                        source_record.get("expected_answer", "")
                    )
                }
                removed_references, reference_status = removed_reference_status(
                    referenced_ids,
                    removed_ids,
                )
                status_counts[reference_status] = (
                    status_counts.get(reference_status, 0) + 1
                )

                question_record = dict(source_record)
                question_record["dataset_name"] = dataset_mapping[dataset_name]
                question_record["sensitivity"] = {
                    "schema_version": 1,
                    "perturbation": "missing_components",
                    "condition_id": current_condition_id,
                    "requested_deletion_rate": float(rate),
                    "seed": seed,
                    "source_dataset_name": dataset_name,
                    "perturbed_dataset_name": dataset_mapping[dataset_name],
                    "referenced_component_ids": sorted(referenced_ids),
                    "removed_referenced_component_ids": removed_references,
                    "reference_status": reference_status,
                }
                write_json(questions_dir / source_path.name, question_record)

            condition_records.append(
                {
                    "condition_id": current_condition_id,
                    "requested_deletion_rate": float(rate),
                    "seed": seed,
                    "questions_dir": str(
                        Path("conditions") / current_condition_id / "questions"
                    ),
                    "results_dir": str(
                        Path("conditions") / current_condition_id / "results"
                    ),
                    "metrics_dir": str(
                        Path("conditions") / current_condition_id / "metrics"
                    ),
                    "dataset_mapping": dataset_mapping,
                    "reference_status_counts": status_counts,
                    "scenes": scene_records,
                }
            )

        manifest = {
            "schema_version": 1,
            "experiment": "missing_components",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling": {
                "method": "uniform_without_replacement_nested_sha256_v1",
                "description": (
                    "For each scene and seed, SHA-256 defines one component ranking. "
                    "Higher deletion rates remove prefixes of the same ranking."
                ),
                "rates": [float(rate) for rate in rates],
                "seeds": seeds,
                "baseline_repetitions": 1,
            },
            "source_benchmark_data_dir": str(benchmark_data_dir),
            "outputs_dir": str(outputs_dir),
            "num_questions": len(questions),
            "num_scenes": len(dataset_names),
            "datasets": dataset_names,
            "conditions": condition_records,
        }
        write_json(temp_experiment_dir / "manifest.json", manifest)
        temp_experiment_dir.rename(experiment_dir)
    except Exception:
        if temp_experiment_dir.exists():
            shutil.rmtree(temp_experiment_dir)
        raise

    print(f"Prepared experiment at {experiment_dir}")
    print(
        "Existing benchmark/data, benchmark/results, benchmark/metrics, and "
        "benchmark/plots were not modified."
    )


if __name__ == "__main__":
    main()
