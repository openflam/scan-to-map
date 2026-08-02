#!/usr/bin/env python3
"""Aggregate missing-component sensitivity metrics into paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = ("AI-Judge", "F1 Score", "Recall")


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
        sampled_indices = rng.integers(
            low=0,
            high=len(cluster_names),
            size=len(cluster_names),
        )
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
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row)

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
                "std": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
        summaries.append(summary)
    return summaries


def load_metric_rows(
    experiment_dir: Path,
    manifest: dict[str, Any],
    selected_metrics: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, str, str]] | None = None

    for condition in manifest["conditions"]:
        metrics_dir = experiment_dir / condition["metrics_dir"]
        metric_files = sorted(metrics_dir.rglob("*_result.json"))
        if not metric_files:
            raise FileNotFoundError(
                f"No metric files found for {condition['condition_id']}: {metrics_dir}"
            )

        condition_keys: set[tuple[str, str, str]] = set()
        for metric_file in metric_files:
            record = read_json(metric_file)
            sensitivity = record.get("sensitivity")
            if not isinstance(sensitivity, dict):
                raise ValueError(
                    f"Sensitivity metadata missing from metric file: {metric_file}"
                )
            if sensitivity.get("condition_id") != condition["condition_id"]:
                raise ValueError(
                    f"Condition mismatch in metric file: {metric_file}"
                )

            config = record.get("config", metric_file.parent.name)
            model = record.get("model", "unknown")
            result_key = (record["file"], config, model)
            if result_key in condition_keys:
                raise ValueError(
                    f"Duplicate result for {result_key} in "
                    f"{condition['condition_id']}."
                )
            condition_keys.add(result_key)

            for metric_name, raw_value in record.get("metrics", {}).items():
                if metric_name not in selected_metrics:
                    continue
                value = numeric_metric(raw_value)
                if value is None:
                    continue

                seed = sensitivity.get("seed")
                source_dataset = sensitivity["source_dataset_name"]
                rows.append(
                    {
                        "condition_id": condition["condition_id"],
                        "deletion_rate": float(
                            sensitivity["requested_deletion_rate"]
                        ),
                        "seed": seed,
                        "benchmark_file": record["file"],
                        "source_dataset_name": source_dataset,
                        "perturbed_dataset_name": sensitivity[
                            "perturbed_dataset_name"
                        ],
                        "benchmark_type": record.get(
                            "benchmark_type", "Unknown"
                        ),
                        "reference_status": sensitivity["reference_status"],
                        "num_referenced_components": len(
                            sensitivity["referenced_component_ids"]
                        ),
                        "num_removed_referenced_components": len(
                            sensitivity["removed_referenced_component_ids"]
                        ),
                        "config": config,
                        "model": model,
                        "metric": metric_name,
                        "value": value,
                        "cluster": (
                            f"baseline:{source_dataset}"
                            if seed is None
                            else f"seed_{seed}:{source_dataset}"
                        ),
                    }
                )

        if condition["requested_deletion_rate"] == 0:
            expected_keys = condition_keys
        elif expected_keys is not None and condition_keys != expected_keys:
            missing = sorted(expected_keys - condition_keys)
            extra = sorted(condition_keys - expected_keys)
            raise ValueError(
                f"Incomplete condition {condition['condition_id']}; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )

    if not rows:
        raise ValueError(
            "No selected numeric metrics were found. Check --metrics and whether "
            "AI-Judge was disabled during metric generation."
        )
    return rows


def add_baseline_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["deletion_rate"] != 0:
            continue
        key = (
            row["benchmark_file"],
            row["config"],
            row["model"],
            row["metric"],
        )
        if key in baseline:
            raise ValueError(f"Duplicate clean baseline metric for {key}.")
        baseline[key] = row["value"]

    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["deletion_rate"] == 0:
            continue
        key = (
            row["benchmark_file"],
            row["config"],
            row["model"],
            row["metric"],
        )
        if key not in baseline:
            raise ValueError(f"Clean baseline metric missing for {key}.")
        delta_row = dict(row)
        delta_row["baseline_value"] = baseline[key]
        delta_row["value"] = row["value"] - baseline[key]
        deltas.append(delta_row)
    return deltas


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
        metric
        for metric in metrics
        if any(row["metric"] == metric for row in summaries)
    ]
    if not available_metrics:
        return

    fig, axes = plt.subplots(
        1,
        len(available_metrics),
        figsize=(4.2 * len(available_metrics), 3.4),
        squeeze=False,
    )
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
    ]

    for axis, metric in zip(axes[0], available_metrics):
        metric_rows = [row for row in summaries if row["metric"] == metric]
        series_keys = sorted(
            {(row["config"], row["model"]) for row in metric_rows}
        )
        for color, (config, model) in zip(palette, series_keys):
            series = sorted(
                (
                    row
                    for row in metric_rows
                    if row["config"] == config and row["model"] == model
                ),
                key=lambda row: row["deletion_rate"],
            )
            x_values = np.array(
                [100 * row["deletion_rate"] for row in series]
            )
            means = np.array([row["mean"] for row in series])
            lower = np.array([row["ci95_low"] for row in series])
            upper = np.array([row["ci95_high"] for row in series])
            errors = np.vstack(
                (
                    np.maximum(0.0, means - lower),
                    np.maximum(0.0, upper - means),
                )
            )
            label = model if len(series_keys) == 1 else f"{model}: {config}"
            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=2,
                color=color,
                label=label,
            )

        if include_zero_line:
            axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(metric)
        axis.set_xlabel("Components removed (%)")
        axis.set_ylabel(f"{metric}{ylabel_suffix}")
        axis.grid(True, linestyle="--", alpha=0.4)
        if len(series_keys) > 1:
            axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize missing-component performance with scene-seed cluster "
            "bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "missing_components",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="New analysis directory. Defaults to <experiment-dir>/analysis.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric names to aggregate and plot.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
    )
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = build_parser(repo_root)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    experiment_dir = args.experiment_dir.resolve()
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Experiment manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else experiment_dir / "analysis"
    )
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite analysis directory: {out_dir}"
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    rows = load_metric_rows(
        experiment_dir,
        manifest,
        selected_metrics=set(args.metrics),
    )
    delta_rows = add_baseline_deltas(rows)

    overall_fields = ("deletion_rate", "config", "model", "metric")
    status_fields = (
        "deletion_rate",
        "config",
        "model",
        "metric",
        "reference_status",
    )
    overall_summary = grouped_summary(
        rows,
        overall_fields,
        args.bootstrap_samples,
        rng,
    )
    delta_summary = grouped_summary(
        delta_rows,
        overall_fields,
        args.bootstrap_samples,
        rng,
    )
    status_summary = grouped_summary(
        rows,
        status_fields,
        args.bootstrap_samples,
        rng,
    )

    long_fields = [
        "condition_id",
        "deletion_rate",
        "seed",
        "benchmark_file",
        "source_dataset_name",
        "perturbed_dataset_name",
        "benchmark_type",
        "reference_status",
        "num_referenced_components",
        "num_removed_referenced_components",
        "config",
        "model",
        "metric",
        "value",
        "cluster",
    ]
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "per_question_metrics.csv", rows, long_fields)
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
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment": "missing_components",
            "confidence_interval": {
                "method": "scene-seed cluster bootstrap percentile interval",
                "level": 0.95,
                "num_samples": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
            },
            "metrics": args.metrics,
            "overall": overall_summary,
            "delta_from_clean_baseline": delta_summary,
            "by_reference_status": status_summary,
        },
    )

    plot_summaries(
        overall_summary,
        args.metrics,
        out_dir / "missing_components_error_propagation.pdf",
        ylabel_suffix="",
        include_zero_line=False,
    )
    plot_summaries(
        delta_summary,
        args.metrics,
        out_dir / "missing_components_delta_from_baseline.pdf",
        ylabel_suffix=" change",
        include_zero_line=True,
    )

    print(f"Wrote sensitivity analysis to {out_dir}")
    print("No files under benchmark/data or benchmark/plots were modified.")


if __name__ == "__main__":
    main()
