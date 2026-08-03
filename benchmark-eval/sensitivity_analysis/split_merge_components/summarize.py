#!/usr/bin/env python3
"""Aggregate split/merge sensitivity metrics into paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = (
    "AI-Judge",
    "F1 Score",
    "Recall",
    "Source F1 Score",
    "Source Recall",
    "Topology Ceiling F1",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def set_metrics(expected: set[int], predicted: set[int]) -> tuple[float, float, float]:
    true_positives = len(expected & predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def integer_set(values: Iterable[Any]) -> set[int]:
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def derived_component_metrics(
    record: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, float]:
    """Score predictions in clean source-component space using lineage metadata."""
    expected = integer_set(record.get("expected_components", []))
    predicted = integer_set(record.get("predicted_components", []))
    lineage = {
        int(component_id): integer_set(source_ids)
        for component_id, source_ids in scene["component_lineage"].items()
    }

    predicted_sources: set[int] = set()
    for component_id in predicted:
        predicted_sources.update(lineage.get(component_id, {component_id}))
    source_precision, source_recall, source_f1 = set_metrics(
        expected,
        predicted_sources,
    )

    oracle_sources: set[int] = set()
    for source_ids in lineage.values():
        if source_ids & expected:
            oracle_sources.update(source_ids)
    _, _, ceiling_f1 = set_metrics(expected, oracle_sources)
    retrieval_efficiency = source_f1 / ceiling_f1 if ceiling_f1 > 0 else 0.0

    merged_sources: set[int] = set()
    for component_id in predicted:
        source_ids = lineage.get(component_id, {component_id})
        if len(source_ids) > 1:
            merged_sources.update(source_ids)
    merge_contamination = (
        len(merged_sources - expected) / len(merged_sources)
        if merged_sources
        else 0.0
    )

    split_parent_counts: Counter[int] = Counter()
    for component_id in predicted:
        source_ids = lineage.get(component_id, {component_id})
        if len(source_ids) == 1:
            split_parent_counts[next(iter(source_ids))] += 1
    redundant_fragments = sum(
        max(0, count - 1) for count in split_parent_counts.values()
    )
    split_redundancy = redundant_fragments / len(predicted) if predicted else 0.0

    return {
        "Source Precision": source_precision,
        "Source Recall": source_recall,
        "Source F1 Score": source_f1,
        "Topology Ceiling F1": ceiling_f1,
        "Retrieval Efficiency": retrieval_efficiency,
        "Merge Contamination": merge_contamination,
        "Split Redundancy": split_redundancy,
    }


def bootstrap_cluster_ci(
    values: list[float],
    clusters: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    cluster_names = sorted(grouped)
    if len(cluster_names) == 1:
        mean = float(np.mean(values))
        return mean, mean
    bootstrap_means = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sampled_indices = rng.integers(0, len(cluster_names), size=len(cluster_names))
        sampled_values = [
            value
            for sampled_index in sampled_indices
            for value in grouped[cluster_names[int(sampled_index)]]
        ]
        bootstrap_means[index] = np.mean(sampled_values)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def grouped_summary(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        values = [float(row["value"]) for row in group]
        clusters = [row["cluster"] for row in group]
        ci_low, ci_high = bootstrap_cluster_ci(
            values,
            clusters,
            bootstrap_samples,
            rng,
        )
        summary = dict(zip(group_fields, key))
        summary.update(
            {
                "num_observations": len(values),
                "num_clusters": len(set(clusters)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
        summaries.append(summary)
    return summaries


def scene_lookup(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (condition["condition_id"], dataset_name): scene
        for condition in manifest["conditions"]
        for dataset_name, scene in condition["scenes"].items()
    }


def metric_values(
    record: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    values = dict(record.get("metrics", {}))
    values.update(derived_component_metrics(record, scene))
    return values


def metric_row(
    record: dict[str, Any],
    sensitivity: dict[str, Any],
    metric_name: str,
    value: float,
    metric_file: Path,
) -> dict[str, Any]:
    source_dataset = sensitivity["source_dataset_name"]
    seed = sensitivity.get("seed")
    return {
        "condition_id": sensitivity["condition_id"],
        "operation": sensitivity["operation"],
        "event_rate": float(sensitivity["requested_event_rate"]),
        "actual_event_rate": float(sensitivity["actual_event_rate"]),
        "component_count_change_rate": float(
            sensitivity["component_count_change_rate"]
        ),
        "affected_source_fraction": float(sensitivity["affected_source_fraction"]),
        "seed": seed,
        "benchmark_file": record["file"],
        "source_dataset_name": source_dataset,
        "perturbed_dataset_name": sensitivity["perturbed_dataset_name"],
        "benchmark_type": record.get("benchmark_type", "Unknown"),
        "reference_status": sensitivity["reference_status"],
        "merge_pairing": sensitivity.get("merge_pairing", "none"),
        "num_referenced_components": len(sensitivity["referenced_component_ids"]),
        "num_affected_referenced_components": len(
            sensitivity["affected_referenced_component_ids"]
        ),
        "config": record.get("config", metric_file.parent.name),
        "model": record.get("model", "unknown"),
        "metric": metric_name,
        "value": value,
        "cluster": (
            f"baseline:{source_dataset}"
            if seed is None
            else f"seed_{seed}:{source_dataset}"
        ),
    }


def load_metric_rows(
    experiment_dir: Path,
    manifest: dict[str, Any],
    selected_metrics: set[str],
    allow_incomplete: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, str, str]] | None = None
    scenes = scene_lookup(manifest)

    for condition in manifest["conditions"]:
        metrics_dir = experiment_dir / condition["metrics_dir"]
        metric_files = sorted(metrics_dir.rglob("*_result.json")) if metrics_dir.exists() else []
        if not metric_files and not allow_incomplete:
            raise FileNotFoundError(
                f"No metric files found for {condition['condition_id']}: {metrics_dir}"
            )
        condition_keys: set[tuple[str, str, str]] = set()
        for metric_file in metric_files:
            record = read_json(metric_file)
            sensitivity = record.get("sensitivity")
            if not isinstance(sensitivity, dict):
                raise ValueError(f"Sensitivity metadata missing: {metric_file}")
            if sensitivity.get("condition_id") != condition["condition_id"]:
                raise ValueError(f"Condition mismatch: {metric_file}")
            source_dataset = sensitivity["source_dataset_name"]
            scene = scenes[(condition["condition_id"], source_dataset)]
            config = record.get("config", metric_file.parent.name)
            model = record.get("model", "unknown")
            result_key = (record["file"], config, model)
            if result_key in condition_keys:
                raise ValueError(
                    f"Duplicate result for {result_key} in {condition['condition_id']}."
                )
            condition_keys.add(result_key)
            for metric_name, raw_value in metric_values(record, scene).items():
                if metric_name not in selected_metrics:
                    continue
                value = numeric_metric(raw_value)
                if value is not None:
                    rows.append(
                        metric_row(record, sensitivity, metric_name, value, metric_file)
                    )

        if condition["operation"] == "baseline":
            if not condition_keys:
                raise FileNotFoundError(f"Clean baseline metrics are missing: {metrics_dir}")
            expected_keys = condition_keys
        elif expected_keys is None:
            raise ValueError("The clean baseline must precede perturbed conditions.")

        if expected_keys is not None:
            missing = expected_keys - condition_keys
            extra = condition_keys - expected_keys
            status = "complete"
            if not condition_keys:
                status = "missing"
            elif missing or extra:
                status = "partial"
            completeness.append(
                {
                    "condition_id": condition["condition_id"],
                    "operation": condition["operation"],
                    "event_rate": float(condition["requested_event_rate"]),
                    "seed": condition["seed"],
                    "status": status,
                    "num_available_results": len(condition_keys),
                    "num_expected_results": len(expected_keys),
                    "num_missing_results": len(missing),
                    "num_extra_results": len(extra),
                    "fraction_complete": (
                        len(condition_keys) / len(expected_keys) if expected_keys else 0.0
                    ),
                }
            )
            if (missing or extra) and not allow_incomplete:
                raise ValueError(
                    f"Incomplete condition {condition['condition_id']}; "
                    f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}."
                )

    if not rows:
        raise ValueError(
            "No selected numeric metrics were found. Check --metrics and whether "
            "AI-Judge was disabled during metric generation."
        )
    return rows, completeness


def add_baseline_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["operation"] != "baseline":
            continue
        key = (row["benchmark_file"], row["config"], row["model"], row["metric"])
        if key in baseline:
            raise ValueError(f"Duplicate clean baseline metric for {key}.")
        baseline[key] = row["value"]
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["operation"] == "baseline":
            continue
        key = (row["benchmark_file"], row["config"], row["model"], row["metric"])
        if key not in baseline:
            raise ValueError(f"Clean baseline metric missing for {key}.")
        delta = dict(row)
        delta["baseline_value"] = baseline[key]
        delta["value"] = row["value"] - baseline[key]
        deltas.append(delta)
    return deltas


def add_operation_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["operation"] not in {"split", "merge"}:
            continue
        key = (
            row["event_rate"],
            row["seed"],
            row["benchmark_file"],
            row["config"],
            row["model"],
            row["metric"],
        )
        indexed[key][row["operation"]] = row
    contrasts: list[dict[str, Any]] = []
    for key, operations in indexed.items():
        if set(operations) != {"split", "merge"}:
            continue
        merge = operations["merge"]
        contrast = dict(merge)
        contrast["operation"] = "merge_minus_split"
        contrast["value"] = merge["value"] - operations["split"]["value"]
        contrasts.append(contrast)
    return contrasts


def summary_fieldnames(group_fields: Iterable[str]) -> list[str]:
    return [
        *group_fields,
        "num_observations",
        "num_clusters",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
    ]


def plot_summaries(
    summaries: list[dict[str, Any]],
    metrics: list[str],
    output_path: Path,
    ylabel_suffix: str,
    include_zero_line: bool,
) -> None:
    available_metrics = [
        metric for metric in metrics if any(row["metric"] == metric for row in summaries)
    ]
    if not available_metrics:
        return
    fig, axes = plt.subplots(
        1,
        len(available_metrics),
        figsize=(4.4 * len(available_metrics), 3.6),
        squeeze=False,
    )
    colors = {"split": "#0072B2", "merge": "#D55E00"}
    markers = {"split": "o", "merge": "s"}
    for axis, metric in zip(axes[0], available_metrics):
        metric_rows = [row for row in summaries if row["metric"] == metric]
        baselines = {
            (row["config"], row["model"]): row
            for row in metric_rows
            if row["operation"] == "baseline"
        }
        series_keys = sorted(
            {
                (row["operation"], row["config"], row["model"])
                for row in metric_rows
                if row["operation"] in colors
            }
        )
        for operation, config, model in series_keys:
            series = sorted(
                [
                    row
                    for row in metric_rows
                    if row["operation"] == operation
                    and row["config"] == config
                    and row["model"] == model
                ],
                key=lambda row: row["event_rate"],
            )
            baseline = baselines.get((config, model))
            if baseline is not None and not include_zero_line:
                series = [baseline, *series]
            x_values = np.array(
                [0.0 if row["operation"] == "baseline" else 100 * row["event_rate"] for row in series]
            )
            means = np.array([row["mean"] for row in series])
            lower = np.array([row["ci95_low"] for row in series])
            upper = np.array([row["ci95_high"] for row in series])
            errors = np.vstack((np.maximum(0, means - lower), np.maximum(0, upper - means)))
            label = operation if len({(c, m) for _, c, m in series_keys}) == 1 else f"{operation}: {model}/{config}"
            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker=markers[operation],
                capsize=3,
                linewidth=2,
                color=colors[operation],
                label=label,
            )
        if include_zero_line:
            axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(metric)
        axis.set_xlabel("Component-boundary events (%)")
        axis.set_ylabel(f"{metric}{ylabel_suffix}")
        axis.grid(True, linestyle="--", alpha=0.4)
        if series_keys:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize split/merge metrics with scene-seed cluster intervals."
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


def summarize(
    experiment_dir: Path,
    out_dir: Path,
    metrics: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    allow_incomplete: bool = False,
) -> None:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("experiment") != "split_merge_components":
        raise ValueError(f"Unexpected experiment type in {manifest_path}.")
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {out_dir}")

    rows, completeness = load_metric_rows(
        experiment_dir,
        manifest,
        selected_metrics=set(metrics),
        allow_incomplete=allow_incomplete,
    )
    delta_rows = add_baseline_deltas(rows)
    contrast_rows = add_operation_contrasts(rows)
    rng = np.random.default_rng(bootstrap_seed)
    overall_fields = ("operation", "event_rate", "config", "model", "metric")
    status_fields = (*overall_fields, "reference_status")
    type_fields = (*overall_fields, "benchmark_type")
    pairing_fields = (*overall_fields, "merge_pairing")
    contrast_fields = ("event_rate", "config", "model", "metric")
    overall_summary = grouped_summary(rows, overall_fields, bootstrap_samples, rng)
    delta_summary = grouped_summary(delta_rows, overall_fields, bootstrap_samples, rng)
    status_summary = grouped_summary(rows, status_fields, bootstrap_samples, rng)
    type_summary = grouped_summary(rows, type_fields, bootstrap_samples, rng)
    pairing_summary = grouped_summary(
        [row for row in rows if row["operation"] == "merge"],
        pairing_fields,
        bootstrap_samples,
        rng,
    )
    contrast_summary = grouped_summary(
        contrast_rows,
        contrast_fields,
        bootstrap_samples,
        rng,
    )

    long_fields = [
        "condition_id",
        "operation",
        "event_rate",
        "actual_event_rate",
        "component_count_change_rate",
        "affected_source_fraction",
        "seed",
        "benchmark_file",
        "source_dataset_name",
        "perturbed_dataset_name",
        "benchmark_type",
        "reference_status",
        "merge_pairing",
        "num_referenced_components",
        "num_affected_referenced_components",
        "config",
        "model",
        "metric",
        "value",
        "cluster",
    ]
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "per_question_metrics.csv", rows, long_fields)
    if allow_incomplete:
        completeness_fields = [
            "condition_id",
            "operation",
            "event_rate",
            "seed",
            "status",
            "num_available_results",
            "num_expected_results",
            "num_missing_results",
            "num_extra_results",
            "fraction_complete",
        ]
        write_csv(
            out_dir / "condition_completeness.csv",
            completeness,
            completeness_fields,
        )
    write_csv(
        out_dir / "overall_summary.csv",
        overall_summary,
        summary_fieldnames(overall_fields),
    )
    write_csv(
        out_dir / "delta_from_baseline_summary.csv",
        delta_summary,
        summary_fieldnames(overall_fields),
    )
    write_csv(
        out_dir / "reference_status_summary.csv",
        status_summary,
        summary_fieldnames(status_fields),
    )
    write_csv(
        out_dir / "benchmark_type_summary.csv",
        type_summary,
        summary_fieldnames(type_fields),
    )
    write_csv(
        out_dir / "merge_pairing_summary.csv",
        pairing_summary,
        summary_fieldnames(pairing_fields),
    )
    write_csv(
        out_dir / "merge_minus_split_summary.csv",
        contrast_summary,
        summary_fieldnames(contrast_fields),
    )
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment": "split_merge_components",
            "analysis_status": "partial" if allow_incomplete else "complete",
            "confidence_interval": {
                "method": "scene-seed cluster bootstrap percentile interval",
                "level": 0.95,
                "num_samples": bootstrap_samples,
                "seed": bootstrap_seed,
            },
            "metrics": metrics,
            "condition_completeness": completeness if allow_incomplete else None,
            "overall": overall_summary,
            "delta_from_clean_baseline": delta_summary,
            "by_reference_status": status_summary,
            "by_benchmark_type": type_summary,
            "by_merge_pairing": pairing_summary,
            "merge_minus_split": contrast_summary,
        },
    )
    prefix = "split_merge_components_partial" if allow_incomplete else "split_merge_components"
    plot_summaries(
        overall_summary,
        metrics,
        out_dir / f"{prefix}_error_propagation.pdf",
        ylabel_suffix="",
        include_zero_line=False,
    )
    plot_summaries(
        delta_summary,
        metrics,
        out_dir / f"{prefix}_delta_from_baseline.pdf",
        ylabel_suffix=" change",
        include_zero_line=True,
    )
    print(f"Wrote {'partial ' if allow_incomplete else ''}sensitivity analysis to {out_dir}")
    print("No files under benchmark/data or benchmark/plots were modified.")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = build_parser(repo_root)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    experiment_dir = args.experiment_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else experiment_dir / "analysis"
    summarize(
        experiment_dir,
        out_dir,
        list(dict.fromkeys(args.metrics)),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
