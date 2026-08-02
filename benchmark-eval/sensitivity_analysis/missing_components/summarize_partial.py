#!/usr/bin/env python3
"""Summarize currently available missing-components results.

This is intentionally separate from summarize.py, whose strict completeness
checks are retained for the final experiment. Partial rates are recorded in a
completeness table and displayed with hollow markers in the generated plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from summarize import (
    add_baseline_deltas,
    grouped_summary,
    numeric_metric,
    read_json,
    summary_fieldnames,
    write_csv,
    write_json,
)


DEFAULT_METRICS = ("F1 Score", "Recall")


def load_available_metric_rows(
    metrics_root: Path,
    manifest: dict[str, Any],
    selected_metrics: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load available metrics and report completeness against the baseline."""
    rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, str, str]] | None = None

    for condition in manifest["conditions"]:
        condition_id = condition["condition_id"]
        metrics_dir = metrics_root / condition_id
        metric_files = sorted(metrics_dir.rglob("*_result.json"))
        condition_keys: set[tuple[str, str, str]] = set()

        for metric_file in metric_files:
            record = read_json(metric_file)
            sensitivity = record.get("sensitivity")
            if not isinstance(sensitivity, dict):
                raise ValueError(
                    f"Sensitivity metadata missing from metric file: {metric_file}"
                )
            if sensitivity.get("condition_id") != condition_id:
                raise ValueError(
                    f"Condition mismatch in metric file: {metric_file}"
                )

            config = record.get("config", metric_file.parent.name)
            model = record.get("model", "unknown")
            result_key = (record["file"], config, model)
            if result_key in condition_keys:
                raise ValueError(
                    f"Duplicate result for {result_key} in {condition_id}."
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
                        "condition_id": condition_id,
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
            if not condition_keys:
                raise FileNotFoundError(
                    f"Clean baseline metrics are missing: {metrics_dir}"
                )
            expected_keys = condition_keys
        elif expected_keys is None:
            raise ValueError("The clean baseline must precede perturbed conditions.")

        assert expected_keys is not None
        missing_keys = expected_keys - condition_keys
        extra_keys = condition_keys - expected_keys
        if not condition_keys:
            status = "missing"
        elif missing_keys or extra_keys:
            status = "partial"
        else:
            status = "complete"

        completeness.append(
            {
                "condition_id": condition_id,
                "deletion_rate": float(condition["requested_deletion_rate"]),
                "seed": condition["seed"],
                "status": status,
                "num_available_results": len(condition_keys),
                "num_expected_results": len(expected_keys),
                "num_missing_results": len(missing_keys),
                "num_extra_results": len(extra_keys),
                "fraction_complete": (
                    len(condition_keys) / len(expected_keys)
                    if expected_keys
                    else 0.0
                ),
            }
        )

    if not rows:
        raise ValueError(
            "No selected numeric metrics were found below "
            f"{metrics_root}. Check --metrics."
        )
    return rows, completeness


def plot_partial_summaries(
    summaries: list[dict[str, Any]],
    metrics: list[str],
    output_path: Path,
    ylabel_suffix: str,
    include_zero_line: bool,
    incomplete_rates: set[float],
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
        figsize=(4.2 * len(available_metrics), 3.7),
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
                fmt="-",
                capsize=3,
                linewidth=2,
                color=color,
                label=label,
            )
            for x_value, mean, row in zip(x_values, means, series):
                incomplete = row["deletion_rate"] in incomplete_rates
                axis.plot(
                    x_value,
                    mean,
                    marker="o",
                    markersize=6,
                    markerfacecolor="white" if incomplete else color,
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                )

        if include_zero_line:
            axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(metric)
        axis.set_xlabel("Components removed (%)")
        axis.set_ylabel(f"{metric}{ylabel_suffix}")
        axis.grid(True, linestyle="--", alpha=0.4)
        if len(series_keys) > 1:
            axis.legend(fontsize=8)

    if incomplete_rates:
        labels = ", ".join(
            f"{100 * rate:g}%" for rate in sorted(incomplete_rates)
        )
        fig.text(
            0.5,
            0.015,
            f"Hollow markers indicate incomplete current results ({labels}).",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize currently available missing-components metrics while "
            "explicitly marking incomplete conditions."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "missing_components",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/current_metrics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment-dir>/analysis_current_results.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
    )
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
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Experiment manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)

    metrics_root = (
        args.metrics_root.resolve()
        if args.metrics_root
        else experiment_dir / "current_metrics"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else experiment_dir / "analysis_current_results"
    )
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite analysis directory: {out_dir}"
        )

    rows, completeness = load_available_metric_rows(
        metrics_root,
        manifest,
        selected_metrics=set(args.metrics),
    )
    delta_rows = add_baseline_deltas(rows)
    incomplete_rates = {
        row["deletion_rate"]
        for row in completeness
        if row["status"] != "complete"
    }

    rng = np.random.default_rng(args.bootstrap_seed)
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
    completeness_fields = [
        "condition_id",
        "deletion_rate",
        "seed",
        "status",
        "num_available_results",
        "num_expected_results",
        "num_missing_results",
        "num_extra_results",
        "fraction_complete",
    ]

    out_dir.mkdir(parents=True)
    write_csv(out_dir / "per_question_metrics.csv", rows, long_fields)
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
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment": "missing_components",
            "analysis_status": "partial_current_results",
            "warning": (
                "Incomplete conditions are summarized only over available "
                "results and must not be reported as final estimates."
            ),
            "confidence_interval": {
                "method": "scene-seed cluster bootstrap percentile interval",
                "level": 0.95,
                "num_samples": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
            },
            "metrics": args.metrics,
            "condition_completeness": completeness,
            "overall": overall_summary,
            "delta_from_clean_baseline": delta_summary,
            "by_reference_status": status_summary,
        },
    )

    plot_partial_summaries(
        overall_summary,
        args.metrics,
        out_dir / "missing_components_current_error_propagation.pdf",
        ylabel_suffix="",
        include_zero_line=False,
        incomplete_rates=incomplete_rates,
    )
    plot_partial_summaries(
        delta_summary,
        args.metrics,
        out_dir / "missing_components_current_delta_from_baseline.pdf",
        ylabel_suffix=" change",
        include_zero_line=True,
        incomplete_rates=incomplete_rates,
    )

    print(f"Wrote partial sensitivity analysis to {out_dir}")
    print(
        "Incomplete conditions are recorded in condition_completeness.csv "
        "and shown with hollow plot markers."
    )
    print("No files under benchmark/data or benchmark/plots were modified.")


if __name__ == "__main__":
    main()
