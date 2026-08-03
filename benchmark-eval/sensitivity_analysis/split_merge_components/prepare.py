#!/usr/bin/env python3
"""Prepare deterministic split/merge scene-memory sensitivity conditions.

The clean scene memories and benchmark questions are immutable.  Every
non-baseline condition is materialized as a new dataset alias with consistent
captions, boxes, crop manifests, crop access, and component-lineage metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


DEFAULT_RATES = ("0", "0.05", "0.10", "0.20", "0.30")
DEFAULT_SEEDS = (0, 1)
OPERATIONS = ("split", "merge")
COMPONENT_TAG_RE = re.compile(r"<component_(\d+)>")
SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9_]+$")
DEFAULT_CAPTION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


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
        raise argparse.ArgumentTypeError(f"Invalid event rate: {raw_value}") from exc
    if not value.is_finite() or value < 0 or value >= Decimal("0.5"):
        raise argparse.ArgumentTypeError(
            "Event rates must be finite fractions in [0, 0.5)."
        )
    if value != value.quantize(Decimal("0.0001")):
        raise argparse.ArgumentTypeError(
            "Event rates may have at most four digits after the decimal point."
        )
    return value


def rate_basis_points(rate: Decimal) -> int:
    return int((rate * Decimal(10_000)).to_integral_value())


def event_count(num_components: int, rate: Decimal) -> int:
    return int(
        (Decimal(num_components) * rate).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def condition_id(
    operation: str | None,
    rate: Decimal,
    seed: int | None,
) -> str:
    if rate == 0:
        return "clean_0000bp_baseline"
    if operation not in OPERATIONS or seed is None:
        raise ValueError("Non-zero conditions require an operation and seed.")
    return f"{operation}_{rate_basis_points(rate):04d}bp_seed_{seed:03d}"


def dataset_alias(
    alias_prefix: str,
    operation: str,
    dataset_name: str,
    rate: Decimal,
    seed: int,
) -> str:
    alias = (
        f"{alias_prefix}_{operation}_{dataset_name}_"
        f"bp{rate_basis_points(rate):04d}_s{seed:03d}"
    )
    if len(alias) > 63:
        raise ValueError(
            f"Dataset alias exceeds PostgreSQL's 63-character limit: {alias}"
        )
    return alias


def stable_hash(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def stable_component_order(
    dataset_name: str,
    component_ids: set[int],
    seed: int,
    operation: str,
) -> list[int]:
    return sorted(
        component_ids,
        key=lambda component_id: stable_hash(
            "split-merge-components-v1",
            operation,
            seed,
            dataset_name,
            component_id,
        ),
    )


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


def bbox_limits(bbox_record: dict[str, Any], source: str) -> tuple[list[float], list[float]]:
    bbox = bbox_record.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"Bounding-box record has no bbox object: {source}")
    corners = bbox.get("corners")
    if not isinstance(corners, list) or len(corners) != 8:
        raise ValueError(f"Bounding box must contain eight corners: {source}")
    try:
        values = [[float(value) for value in corner] for corner in corners]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bounding box has non-numeric corners: {source}") from exc
    if any(len(corner) != 3 for corner in values):
        raise ValueError(f"Bounding-box corners must be 3-D: {source}")
    if any(not math.isfinite(value) for corner in values for value in corner):
        raise ValueError(f"Bounding box has non-finite coordinates: {source}")
    minimum = [min(corner[axis] for corner in values) for axis in range(3)]
    maximum = [max(corner[axis] for corner in values) for axis in range(3)]
    if max(maximum[axis] - minimum[axis] for axis in range(3)) <= 0:
        raise ValueError(f"Bounding box has no positive extent: {source}")
    return minimum, maximum


def axis_aligned_bbox(minimum: list[float], maximum: list[float]) -> dict[str, Any]:
    corners = [
        [x, y, z]
        for z in (minimum[2], maximum[2])
        for y in (minimum[1], maximum[1])
        for x in (minimum[0], maximum[0])
    ]
    return {
        "corners": corners,
        "min": minimum,
        "max": maximum,
        "center": [
            (minimum[axis] + maximum[axis]) / 2 for axis in range(3)
        ],
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def split_bbox_record(
    source_record: dict[str, Any],
    child_ids: tuple[int, int],
    source: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    minimum, maximum = bbox_limits(source_record, source)
    sizes = [maximum[axis] - minimum[axis] for axis in range(3)]
    axis = max(range(3), key=lambda index: (sizes[index], -index))
    midpoint = (minimum[axis] + maximum[axis]) / 2

    first_maximum = list(maximum)
    first_maximum[axis] = midpoint
    second_minimum = list(minimum)
    second_minimum[axis] = midpoint

    children: list[dict[str, Any]] = []
    for component_id, child_minimum, child_maximum in (
        (child_ids[0], list(minimum), first_maximum),
        (child_ids[1], second_minimum, list(maximum)),
    ):
        record = deepcopy(source_record)
        record["connected_comp_id"] = component_id
        record["bbox"] = axis_aligned_bbox(child_minimum, child_maximum)
        children.append(record)
    return children[0], children[1], axis


def merge_bbox_records(
    first: dict[str, Any],
    second: dict[str, Any],
    canonical_id: int,
    source: str,
) -> dict[str, Any]:
    first_minimum, first_maximum = bbox_limits(first, f"{source}:first")
    second_minimum, second_maximum = bbox_limits(second, f"{source}:second")
    minimum = [min(first_minimum[axis], second_minimum[axis]) for axis in range(3)]
    maximum = [max(first_maximum[axis], second_maximum[axis]) for axis in range(3)]
    record = deepcopy(first)
    record["connected_comp_id"] = canonical_id
    record["bbox"] = axis_aligned_bbox(minimum, maximum)
    return record


def bbox_gap(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_minimum, first_maximum = bbox_limits(first, "merge candidate")
    second_minimum, second_maximum = bbox_limits(second, "merge candidate")
    squared_gap = 0.0
    for axis in range(3):
        gap = max(
            0.0,
            first_minimum[axis] - second_maximum[axis],
            second_minimum[axis] - first_maximum[axis],
        )
        squared_gap += gap * gap
    return math.sqrt(squared_gap)


def bbox_center_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_minimum, first_maximum = bbox_limits(first, "merge candidate")
    second_minimum, second_maximum = bbox_limits(second, "merge candidate")
    first_center = [
        (first_minimum[axis] + first_maximum[axis]) / 2 for axis in range(3)
    ]
    second_center = [
        (second_minimum[axis] + second_maximum[axis]) / 2 for axis in range(3)
    ]
    return math.sqrt(
        sum(
            (first_center[axis] - second_center[axis]) ** 2
            for axis in range(3)
        )
    )


def load_scene_memory(outputs_dir: Path, dataset_name: str) -> dict[str, Any]:
    scene_dir = outputs_dir / dataset_name
    captions_path = scene_dir / "component_captions.json"
    bboxes_path = scene_dir / "bbox_corners.json"
    manifest_path = scene_dir / "crops" / "manifest.json"
    missing = [
        path for path in (captions_path, bboxes_path, manifest_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete scene memory for {dataset_name}: "
            + ", ".join(str(path) for path in missing)
        )

    captions = read_json(captions_path)
    bboxes = read_json(bboxes_path)
    manifest = read_json(manifest_path)
    if not isinstance(captions, list) or not isinstance(bboxes, list):
        raise ValueError(f"Caption and bounding-box files must be lists: {dataset_name}")
    if not isinstance(manifest, dict):
        raise ValueError(f"Crop manifest must be an object: {dataset_name}")

    caption_map = unique_id_map(captions, "component_id", captions_path)
    bbox_map = unique_id_map(bboxes, "connected_comp_id", bboxes_path)
    component_ids = set(caption_map)
    if component_ids - set(bbox_map):
        raise ValueError(
            f"Bounding boxes missing IDs {sorted(component_ids - set(bbox_map))}: "
            f"{dataset_name}"
        )
    try:
        manifest_ids = {int(component_id) for component_id in manifest}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest keys must be integer IDs: {dataset_name}") from exc
    if component_ids - manifest_ids:
        raise ValueError(
            f"Crop manifest missing IDs {sorted(component_ids - manifest_ids)}: "
            f"{dataset_name}"
        )

    for component_id in component_ids:
        bbox_limits(bbox_map[component_id], f"{dataset_name}:{component_id}")
        entry = manifest[str(component_id)]
        if not isinstance(entry, dict) or not isinstance(entry.get("crops", []), list):
            raise ValueError(f"Invalid manifest entry: {dataset_name}:{component_id}")
        for crop in entry.get("crops", []):
            filename = crop.get("crop_filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(
                    f"Unsafe crop filename for {dataset_name}:{component_id}: {filename!r}"
                )
            crop_path = scene_dir / "crops" / f"component_{component_id}" / filename
            if not crop_path.is_file():
                raise FileNotFoundError(f"Crop image missing: {crop_path}")

    return {
        "scene_dir": scene_dir,
        "captions": captions,
        "caption_map": caption_map,
        "bboxes": bboxes,
        "bbox_map": bbox_map,
        "manifest": manifest,
        "component_ids": component_ids,
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


def build_split_plan(
    dataset_name: str,
    scene_memory: dict[str, Any],
    seed: int,
) -> list[int]:
    return stable_component_order(
        dataset_name,
        scene_memory["component_ids"],
        seed,
        "split",
    )


def build_merge_plan(
    dataset_name: str,
    scene_memory: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    order = stable_component_order(
        dataset_name,
        scene_memory["component_ids"],
        seed,
        "merge",
    )
    unpaired = set(order)
    pairs: list[dict[str, Any]] = []
    for anchor in order:
        if anchor not in unpaired:
            continue
        candidates = unpaired - {anchor}
        if not candidates:
            break
        partner = min(
            candidates,
            key=lambda candidate: (
                bbox_gap(
                    scene_memory["bbox_map"][anchor],
                    scene_memory["bbox_map"][candidate],
                ),
                stable_hash(
                    "split-merge-components-v1",
                    "merge-partner",
                    seed,
                    dataset_name,
                    anchor,
                    candidate,
                ),
            ),
        )
        canonical_id = min(
            (anchor, partner),
            key=lambda component_id: stable_hash(
                "split-merge-components-v1",
                "merge-canonical",
                seed,
                dataset_name,
                min(anchor, partner),
                max(anchor, partner),
                component_id,
            ),
        )
        absorbed_id = partner if canonical_id == anchor else anchor
        pairs.append(
            {
                "source_component_ids": sorted((anchor, partner)),
                "canonical_component_id": canonical_id,
                "absorbed_component_id": absorbed_id,
                "box_gap": bbox_gap(
                    scene_memory["bbox_map"][anchor],
                    scene_memory["bbox_map"][partner],
                ),
                "center_distance": bbox_center_distance(
                    scene_memory["bbox_map"][anchor],
                    scene_memory["bbox_map"][partner],
                ),
            }
        )
        unpaired.remove(anchor)
        unpaired.remove(partner)
    return pairs


def make_relative_symlink(source: Path, destination: Path) -> None:
    relative_source = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_source, target_is_directory=source.is_dir())


def materialize_crops(
    target_crops_dir: Path,
    target_component_id: int,
    source_component_ids: list[int],
    scene_memory: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[int, str], str]]:
    target_component_dir = target_crops_dir / f"component_{target_component_id}"
    target_component_dir.mkdir()
    crops: list[dict[str, Any]] = []
    filename_map: dict[tuple[int, str], str] = {}
    for source_component_id in source_component_ids:
        source_entry = scene_memory["manifest"][str(source_component_id)]
        source_dir = (
            scene_memory["scene_dir"] / "crops" / f"component_{source_component_id}"
        )
        for crop in source_entry.get("crops", []):
            source_filename = crop["crop_filename"]
            target_filename = f"source_{source_component_id}_{source_filename}"
            filename_map[(source_component_id, source_filename)] = target_filename
            target_crop = deepcopy(crop)
            target_crop["crop_filename"] = target_filename
            target_crop["source_component_id"] = source_component_id
            crops.append(target_crop)
            make_relative_symlink(
                source_dir / source_filename,
                target_component_dir / target_filename,
            )
    crops.sort(
        key=lambda crop: (
            -float(crop.get("fraction_visible", 0.0)),
            crop["crop_filename"],
        )
    )
    return (
        {
            "component_id": target_component_id,
            "total_crops": len(crops),
            "crops": crops,
        },
        filename_map,
    )


def transformed_caption(
    target_id: int,
    source_ids: list[int],
    scene_memory: dict[str, Any],
    filename_map: dict[tuple[int, str], str],
) -> dict[str, Any]:
    source_records = [scene_memory["caption_map"][source_id] for source_id in source_ids]
    result = deepcopy(source_records[0])
    result["component_id"] = target_id
    result["caption"] = "; ".join(
        str(record.get("caption", "")).strip() for record in source_records
    )
    crop_filenames: list[str] = []
    for source_id, record in zip(source_ids, source_records):
        for filename in record.get("crop_filenames", []):
            mapped = filename_map.get((source_id, filename))
            if mapped is not None:
                crop_filenames.append(mapped)
    result["crop_filenames"] = crop_filenames
    result["num_images_used"] = len(crop_filenames)
    return result


def materialize_scene_alias(
    outputs_dir: Path,
    dataset_name: str,
    alias: str,
    scene_memory: dict[str, Any],
    operation: str,
    selected: list[int] | list[dict[str, Any]],
    rate: Decimal,
    seed: int,
    *,
    repo_root: Path | None = None,
    projection_context: Any | None = None,
    captioner: Any | None = None,
    caption_n_images: int = 1,
    caption_batch_size: int = 512,
) -> dict[str, Any]:
    target_dir = outputs_dir / alias
    if target_dir.exists() or target_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite scene-memory alias: {target_dir}")
    recreate_crops_captions = projection_context is not None
    if recreate_crops_captions and (repo_root is None or captioner is None):
        raise ValueError("Crop/caption recreation requires repo_root and a captioner")
    if recreate_crops_captions:
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from recreate_crops_captions import (
            caption_recreated_components,
            merge_component_point_ids,
            recreate_component_crops,
            split_component_point_ids,
        )

    source_ids = set(scene_memory["component_ids"])
    captions: dict[int, dict[str, Any]] = {
        component_id: deepcopy(scene_memory["caption_map"][component_id])
        for component_id in source_ids
    }
    bboxes: dict[int, dict[str, Any]] = {
        component_id: deepcopy(scene_memory["bbox_map"][component_id])
        for component_id in source_ids
    }
    manifests: dict[int, dict[str, Any]] = {
        component_id: deepcopy(scene_memory["manifest"][str(component_id)])
        for component_id in source_ids
    }
    lineage: dict[int, list[int]] = {
        component_id: [component_id] for component_id in source_ids
    }
    clean_to_perturbed: dict[int, list[int]] = {
        component_id: [component_id] for component_id in source_ids
    }
    untouched_ids = set(source_ids)

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{alias}.tmp-", dir=outputs_dir))
    try:
        crops_target = temp_dir / "crops"
        crops_target.mkdir()
        operation_records: list[dict[str, Any]] = []
        recreated_component_ids: set[int] = set()
        projection_records: dict[int, list[dict[str, Any]]] = {}

        if operation == "split":
            split_ids = [int(component_id) for component_id in selected]
            next_id = max(source_ids, default=-1) + 1
            for source_id in split_ids:
                new_id = next_id
                next_id += 1
                first_bbox, second_bbox, axis = split_bbox_record(
                    scene_memory["bbox_map"][source_id],
                    (source_id, new_id),
                    f"{dataset_name}:{source_id}",
                )
                split_point_ids: tuple[list[int] | None, list[int] | None] = (
                    None,
                    None,
                )
                if recreate_crops_captions:
                    midpoint = float(first_bbox["bbox"]["max"][axis])
                    split_point_ids = split_component_point_ids(
                        projection_context,
                        source_id,
                        axis,
                        midpoint,
                    )
                for child_index, (child_id, child_bbox) in enumerate((
                    (source_id, first_bbox),
                    (new_id, second_bbox),
                )):
                    if recreate_crops_captions:
                        manifest, projections = recreate_component_crops(
                            projection_context,
                            crops_target,
                            child_id,
                            child_bbox,
                            [source_id],
                            split_point_ids[child_index],
                        )
                        captions[child_id] = {"component_id": child_id}
                        projection_records[child_id] = projections
                        recreated_component_ids.add(child_id)
                    else:
                        manifest, filename_map = materialize_crops(
                            crops_target,
                            child_id,
                            [source_id],
                            scene_memory,
                        )
                        captions[child_id] = transformed_caption(
                            child_id,
                            [source_id],
                            scene_memory,
                            filename_map,
                        )
                    bboxes[child_id] = child_bbox
                    manifests[child_id] = manifest
                    lineage[child_id] = [source_id]
                clean_to_perturbed[source_id] = [source_id, new_id]
                untouched_ids.remove(source_id)
                operation_records.append(
                    {
                        "source_component_id": source_id,
                        "child_component_ids": [source_id, new_id],
                        "split_axis": axis,
                        "split_fraction": 0.5,
                    }
                )
        elif operation == "merge":
            merge_pairs = [dict(pair) for pair in selected]
            for pair in merge_pairs:
                pair_source_ids = [int(value) for value in pair["source_component_ids"]]
                canonical_id = int(pair["canonical_component_id"])
                absorbed_id = int(pair["absorbed_component_id"])
                merged_bbox = merge_bbox_records(
                    scene_memory["bbox_map"][pair_source_ids[0]],
                    scene_memory["bbox_map"][pair_source_ids[1]],
                    canonical_id,
                    f"{dataset_name}:{pair_source_ids}",
                )
                if recreate_crops_captions:
                    merged_point_ids = merge_component_point_ids(
                        projection_context,
                        pair_source_ids,
                    )
                    manifest, projections = recreate_component_crops(
                        projection_context,
                        crops_target,
                        canonical_id,
                        merged_bbox,
                        pair_source_ids,
                        merged_point_ids,
                    )
                    captions[canonical_id] = {"component_id": canonical_id}
                    projection_records[canonical_id] = projections
                    recreated_component_ids.add(canonical_id)
                else:
                    manifest, filename_map = materialize_crops(
                        crops_target,
                        canonical_id,
                        pair_source_ids,
                        scene_memory,
                    )
                    captions[canonical_id] = transformed_caption(
                        canonical_id,
                        pair_source_ids,
                        scene_memory,
                        filename_map,
                    )
                bboxes[canonical_id] = merged_bbox
                manifests[canonical_id] = manifest
                lineage[canonical_id] = sorted(pair_source_ids)
                clean_to_perturbed[pair_source_ids[0]] = [canonical_id]
                clean_to_perturbed[pair_source_ids[1]] = [canonical_id]
                for component_id in pair_source_ids:
                    untouched_ids.remove(component_id)
                captions.pop(absorbed_id)
                bboxes.pop(absorbed_id)
                manifests.pop(absorbed_id)
                lineage.pop(absorbed_id)
                operation_records.append(pair)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        if recreate_crops_captions:
            captions.update(
                caption_recreated_components(
                    repo_root,
                    captioner,
                    crops_target,
                    manifests,
                    recreated_component_ids,
                    caption_n_images,
                    caption_batch_size,
                )
            )

        for component_id in sorted(untouched_ids):
            source_component_dir = (
                scene_memory["scene_dir"] / "crops" / f"component_{component_id}"
            )
            if source_component_dir.is_dir():
                make_relative_symlink(
                    source_component_dir,
                    crops_target / f"component_{component_id}",
                )

        final_ids = set(captions)
        if final_ids != set(bboxes) or final_ids != set(manifests) or final_ids != set(lineage):
            raise AssertionError("Perturbed scene-memory component IDs are inconsistent.")

        write_json(
            temp_dir / "component_captions.json",
            [captions[component_id] for component_id in sorted(final_ids)],
        )
        write_json(
            temp_dir / "bbox_corners.json",
            [bboxes[component_id] for component_id in sorted(final_ids)],
        )
        write_json(
            crops_target / "manifest.json",
            {str(component_id): manifests[component_id] for component_id in sorted(final_ids)},
        )
        if recreate_crops_captions:
            write_json(
                temp_dir / "image_crop_coordinates.json",
                {
                    str(component_id): projection_records[component_id]
                    for component_id in sorted(projection_records)
                },
            )

        source_mesh = scene_memory["scene_dir"] / "raw.glb"
        if source_mesh.is_file():
            make_relative_symlink(source_mesh, temp_dir / "raw.glb")

        num_events = len(operation_records)
        affected_ids = sorted(
            {
                source_id
                for record in operation_records
                for source_id in (
                    [record["source_component_id"]]
                    if operation == "split"
                    else record["source_component_ids"]
                )
            }
        )
        perturbation = {
            "schema_version": 1,
            "perturbation": "split_merge_components",
            "operation": operation,
            "sampling": "nested_sha256_spatial_nearest_v1",
            "source_dataset_name": dataset_name,
            "dataset_name": alias,
            "requested_event_rate": float(rate),
            "actual_event_rate": num_events / len(source_ids),
            "component_count_change_rate": (len(final_ids) - len(source_ids)) / len(source_ids),
            "affected_source_fraction": len(affected_ids) / len(source_ids),
            "seed": seed,
            "num_source_components": len(source_ids),
            "num_perturbed_components": len(final_ids),
            "num_events": num_events,
            "affected_source_component_ids": affected_ids,
            "operations": operation_records,
            "component_lineage": {
                str(component_id): lineage[component_id]
                for component_id in sorted(lineage)
            },
            "clean_to_perturbed_ids": {
                str(component_id): clean_to_perturbed[component_id]
                for component_id in sorted(clean_to_perturbed)
            },
            "recreate_crops_captions": recreate_crops_captions,
            "caption_protocol": (
                "segment3d_captioning_from_reprojected_crops"
                if recreate_crops_captions
                else (
                    "duplicate_source_caption"
                    if operation == "split"
                    else "neutral_concatenation"
                )
            ),
            "crop_protocol": (
                "project_perturbed_bbox_and_crop_source_images"
                if recreate_crops_captions
                else "oracle_preserving_source_crop_union"
            ),
            "visibility_protocol": (
                projection_context.visibility_protocol
                if recreate_crops_captions
                else "source_crop_union"
            ),
            "projection_min_fraction": (
                projection_context.min_fraction
                if recreate_crops_captions
                else None
            ),
            "max_crops_per_component": (
                projection_context.max_crops_per_component
                if recreate_crops_captions
                else None
            ),
            "geometry_protocol": (
                "balanced_longest_aabb_axis" if operation == "split" else "aabb_union"
            ),
            "connected_components_materialized": False,
        }
        write_json(temp_dir / "scene_memory_perturbation.json", perturbation)
        temp_dir.rename(target_dir)
        return perturbation
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def reference_metadata(
    referenced_ids: set[int],
    perturbation: dict[str, Any],
) -> dict[str, Any]:
    operation = perturbation["operation"]
    affected = set(perturbation["affected_source_component_ids"])
    affected_references = sorted(referenced_ids & affected)
    if not referenced_ids:
        status = "no_referenced_components"
    elif not affected_references:
        status = "targets_unaffected"
    elif operation == "split":
        status = (
            "all_targets_split"
            if len(affected_references) == len(referenced_ids)
            else "some_targets_split"
        )
    elif operation == "merge":
        status = (
            "all_targets_merged"
            if len(affected_references) == len(referenced_ids)
            else "some_targets_merged"
        )
    else:
        status = "targets_unaffected"

    merge_pairing = "none"
    if operation == "merge" and affected_references:
        exposures: set[str] = set()
        for pair in perturbation["operations"]:
            pair_ids = set(pair["source_component_ids"])
            if not (pair_ids & referenced_ids):
                continue
            exposures.add(
                "target_with_target"
                if pair_ids <= referenced_ids
                else "target_with_distractor"
            )
        merge_pairing = next(iter(exposures)) if len(exposures) == 1 else "mixed"

    return {
        "referenced_component_ids": sorted(referenced_ids),
        "affected_referenced_component_ids": affected_references,
        "reference_status": status,
        "merge_pairing": merge_pairing,
    }


def clean_perturbation(dataset_name: str, component_ids: set[int]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "perturbation": "split_merge_components",
        "operation": "baseline",
        "sampling": "clean_baseline",
        "source_dataset_name": dataset_name,
        "dataset_name": dataset_name,
        "requested_event_rate": 0.0,
        "actual_event_rate": 0.0,
        "component_count_change_rate": 0.0,
        "affected_source_fraction": 0.0,
        "seed": None,
        "num_source_components": len(component_ids),
        "num_perturbed_components": len(component_ids),
        "num_events": 0,
        "affected_source_component_ids": [],
        "operations": [],
        "component_lineage": {
            str(component_id): [component_id] for component_id in sorted(component_ids)
        },
        "clean_to_perturbed_ids": {
            str(component_id): [component_id] for component_id in sorted(component_ids)
        },
    }


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic split/merge scene-memory variants without "
            "modifying existing outputs or benchmark artifacts."
        )
    )
    parser.add_argument(
        "--benchmark-data-dir",
        type=Path,
        default=repo_root / "benchmark" / "data",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root / "outputs",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=repo_root / "benchmark" / "split_merge_components",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=parse_rate,
        default=[parse_rate(rate) for rate in DEFAULT_RATES],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--dataset-prefix", default="scannetpp_")
    parser.add_argument("--alias-prefix", default="split_merge")
    parser.add_argument(
        "--scannetpp-data-root",
        type=Path,
        default=None,
        help=(
            "Optional ScanNet++ data/data root; uses <scene>/dslr/colmap and "
            "<scene>/dslr/resized_images for reprojection."
        ),
    )
    parser.add_argument(
        "--recreate-crops-captions",
        action="store_true",
        help=(
            "Reproject perturbed boxes into source images, create fresh crops, and "
            "caption affected components with segment3d instead of reusing source data."
        ),
    )
    parser.add_argument("--projection-min-fraction", type=float, default=0.3)
    parser.add_argument("--captioner-type", default="vllm")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--caption-device", type=int, default=0)
    parser.add_argument("--caption-n-images", type=int, default=1)
    parser.add_argument("--caption-batch-size", type=int, default=512)
    parser.add_argument("--max-crops-per-component", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and print the complete condition plan without writing.",
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
    if not benchmark_data_dir.is_dir():
        parser.error(f"Benchmark data directory does not exist: {benchmark_data_dir}")
    if not outputs_dir.is_dir():
        parser.error(f"Outputs directory does not exist: {outputs_dir}")
    if experiment_dir in {benchmark_data_dir, outputs_dir}:
        parser.error("Experiment directory cannot be a source directory")
    if scannetpp_data_root is not None and not scannetpp_data_root.is_dir():
        parser.error(f"ScanNet++ data root does not exist: {scannetpp_data_root}")

    questions = discover_questions(benchmark_data_dir, args.dataset_prefix)
    dataset_names = sorted({record["dataset_name"] for _, record in questions})
    scene_memories = {
        dataset_name: load_scene_memory(outputs_dir, dataset_name)
        for dataset_name in dataset_names
    }
    for question_path, question in questions:
        dataset_name = question["dataset_name"]
        referenced_ids = {
            int(component_id)
            for component_id in COMPONENT_TAG_RE.findall(question.get("expected_answer", ""))
        }
        unknown = referenced_ids - scene_memories[dataset_name]["component_ids"]
        if unknown:
            raise ValueError(
                f"{question_path} references unknown components {sorted(unknown)}."
            )

    nonzero_rates = [rate for rate in rates if rate != 0]
    conditions: list[tuple[str | None, Decimal, int | None]] = [(None, Decimal(0), None)]
    conditions.extend(
        (operation, rate, seed)
        for operation in OPERATIONS
        for rate in nonzero_rates
        for seed in seeds
    )

    plans: dict[tuple[str, str, int], list[Any]] = {}
    for dataset_name, scene_memory in scene_memories.items():
        maximum_events = max(
            (event_count(len(scene_memory["component_ids"]), rate) for rate in nonzero_rates),
            default=0,
        )
        for seed in seeds:
            split_plan = build_split_plan(dataset_name, scene_memory, seed)
            merge_plan = build_merge_plan(dataset_name, scene_memory, seed)
            if maximum_events > len(split_plan):
                raise ValueError(f"Not enough components to split in {dataset_name}.")
            if maximum_events > len(merge_plan):
                raise ValueError(
                    f"Not enough disjoint component pairs to merge in {dataset_name}: "
                    f"need {maximum_events}, have {len(merge_plan)}."
                )
            plans[("split", dataset_name, seed)] = split_plan
            plans[("merge", dataset_name, seed)] = merge_plan

    required_source_components: dict[str, set[int]] = {
        dataset_name: set() for dataset_name in dataset_names
    }
    maximum_rate = max(nonzero_rates, default=Decimal(0))
    for dataset_name, scene_memory in scene_memories.items():
        maximum_events = event_count(
            len(scene_memory["component_ids"]),
            maximum_rate,
        )
        for seed in seeds:
            required_source_components[dataset_name].update(
                int(component_id)
                for component_id in plans[("split", dataset_name, seed)][:maximum_events]
            )
            for pair in plans[("merge", dataset_name, seed)][:maximum_events]:
                required_source_components[dataset_name].update(
                    int(component_id) for component_id in pair["source_component_ids"]
                )

    aliases = {
        dataset_alias(args.alias_prefix, operation, dataset_name, rate, seed)
        for operation, rate, seed in conditions
        if operation is not None and seed is not None
        for dataset_name in dataset_names
    }
    if experiment_dir.exists() or experiment_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {experiment_dir}")
    for alias in sorted(aliases):
        path = outputs_dir / alias
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite scene-memory alias: {path}")

    print(
        f"Validated {len(questions)} questions across {len(dataset_names)} scenes; "
        f"{len(conditions)} conditions and {len(aliases)} new aliases."
    )
    print("Condition plan")
    for operation, rate, seed in conditions:
        total_events = sum(
            event_count(len(scene_memory["component_ids"]), rate)
            for scene_memory in scene_memories.values()
        )
        print(
            f"  {condition_id(operation, rate, seed)}: operation="
            f"{operation or 'baseline'}, rate={float(rate):.4f}, seed={seed}, "
            f"events={total_events}"
        )
    if args.recreate_crops_captions:
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from recreate_crops_captions import source_data_status

        recreated_captions = sum(
            total_events * (2 if operation == "split" else 1)
            for operation, rate, seed in conditions
            if operation is not None
            for total_events in [
                sum(
                    event_count(len(scene_memory["component_ids"]), rate)
                    for scene_memory in scene_memories.values()
                )
            ]
        )
        statuses = [
            source_data_status(
                repo_root,
                dataset_name,
                outputs_dir,
                scannetpp_data_root,
            )
            for dataset_name in dataset_names
        ]
        ready = sum(
            status["images_available"] and status["colmap_available"]
            for status in statuses
        )
        memberships = sum(
            status["point_membership_available"] for status in statuses
        )
        print("Crop/caption recreation plan")
        print(f"  Affected component captions to generate: {recreated_captions}")
        print(f"  Source images and COLMAP available: {ready}/{len(statuses)} scenes")
        print(
            "  COLMAP point-visibility memberships available: "
            f"{memberships}/{len(statuses)} scenes"
        )
        if ready != len(statuses):
            print(
                "  Source-data warning: a full run requires data/<scene>/ns_data/images "
                "and a COLMAP reconstruction for every scene."
            )
            if not args.dry_run:
                missing_scenes = [
                    status["dataset_name"]
                    for status in statuses
                    if not (
                        status["images_available"] and status["colmap_available"]
                    )
                ]
                raise FileNotFoundError(
                    "Crop/caption recreation source data is missing for scenes: "
                    + ", ".join(missing_scenes)
                )
    if args.dry_run:
        print("Dry run complete. No files were created.")
        return

    experiment_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_experiment_dir = Path(
        tempfile.mkdtemp(prefix=f".{experiment_dir.name}.tmp-", dir=experiment_dir.parent)
    )
    condition_records: list[dict[str, Any]] = []
    captioner = None
    try:
        if args.recreate_crops_captions:
            from recreate_crops_captions import create_captioner, load_projection_context

            captioner = create_captioner(
                repo_root,
                args.captioner_type,
                args.caption_model,
                args.caption_device,
            )

        condition_states: list[dict[str, Any]] = []
        for operation, rate, seed in conditions:
            current_id = condition_id(operation, rate, seed)
            questions_dir = temp_experiment_dir / "conditions" / current_id / "questions"
            questions_dir.mkdir(parents=True)
            condition_states.append(
                {
                    "condition_id": current_id,
                    "operation": operation,
                    "rate": rate,
                    "seed": seed,
                    "questions_dir": questions_dir,
                    "dataset_mapping": {},
                    "scene_records": {},
                }
            )

        for dataset_name, scene_memory in scene_memories.items():
            projection_context = None
            if args.recreate_crops_captions:
                print(f"Loading projection sources for {dataset_name}...", flush=True)
                projection_context = load_projection_context(
                    repo_root,
                    dataset_name,
                    scene_memory["manifest"],
                    args.projection_min_fraction,
                    scene_memory["scene_dir"],
                    scene_memory["bbox_map"],
                    required_source_components[dataset_name],
                    scannetpp_data_root,
                    args.max_crops_per_component,
                )
            for state in condition_states:
                operation = state["operation"]
                rate = state["rate"]
                seed = state["seed"]
                if operation is None:
                    alias = dataset_name
                    perturbation = clean_perturbation(
                        dataset_name,
                        scene_memory["component_ids"],
                    )
                else:
                    assert seed is not None
                    count = event_count(len(scene_memory["component_ids"]), rate)
                    selected = plans[(operation, dataset_name, seed)][:count]
                    alias = dataset_alias(
                        args.alias_prefix,
                        operation,
                        dataset_name,
                        rate,
                        seed,
                    )
                    perturbation = materialize_scene_alias(
                        outputs_dir,
                        dataset_name,
                        alias,
                        scene_memory,
                        operation,
                        selected,
                        rate,
                        seed,
                        repo_root=repo_root,
                        projection_context=projection_context,
                        captioner=captioner,
                        caption_n_images=args.caption_n_images,
                        caption_batch_size=args.caption_batch_size,
                    )
                state["dataset_mapping"][dataset_name] = alias
                state["scene_records"][dataset_name] = perturbation
            del projection_context

        for state in condition_states:
            current_id = state["condition_id"]
            operation = state["operation"]
            rate = state["rate"]
            seed = state["seed"]
            questions_dir = state["questions_dir"]
            dataset_mapping = state["dataset_mapping"]
            scene_records = state["scene_records"]
            status_counts: dict[str, int] = {}
            for source_path, source_record in questions:
                dataset_name = source_record["dataset_name"]
                perturbation = scene_records[dataset_name]
                referenced_ids = {
                    int(component_id)
                    for component_id in COMPONENT_TAG_RE.findall(
                        source_record.get("expected_answer", "")
                    )
                }
                references = reference_metadata(referenced_ids, perturbation)
                status = references["reference_status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                question_record = dict(source_record)
                question_record["dataset_name"] = dataset_mapping[dataset_name]
                question_record["sensitivity"] = {
                    "schema_version": 1,
                    "perturbation": "split_merge_components",
                    "operation": perturbation["operation"],
                    "condition_id": current_id,
                    "requested_event_rate": float(rate),
                    "actual_event_rate": perturbation["actual_event_rate"],
                    "component_count_change_rate": perturbation[
                        "component_count_change_rate"
                    ],
                    "affected_source_fraction": perturbation[
                        "affected_source_fraction"
                    ],
                    "seed": seed,
                    "source_dataset_name": dataset_name,
                    "perturbed_dataset_name": dataset_mapping[dataset_name],
                    **references,
                }
                write_json(questions_dir / source_path.name, question_record)

            total_source = sum(
                scene["num_source_components"] for scene in scene_records.values()
            )
            total_events = sum(scene["num_events"] for scene in scene_records.values())
            condition_records.append(
                {
                    "condition_id": current_id,
                    "operation": operation or "baseline",
                    "requested_event_rate": float(rate),
                    "actual_event_rate": total_events / total_source,
                    "seed": seed,
                    "questions_dir": str(Path("conditions") / current_id / "questions"),
                    "results_dir": str(Path("conditions") / current_id / "results"),
                    "metrics_dir": str(Path("conditions") / current_id / "metrics"),
                    "dataset_mapping": dataset_mapping,
                    "reference_status_counts": status_counts,
                    "scenes": scene_records,
                }
            )

        manifest = {
            "schema_version": 1,
            "experiment": "split_merge_components",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling": {
                "method": "nested_sha256_spatial_nearest_v1",
                "description": (
                    "Split targets and greedy disjoint spatial-nearest merge pairs "
                    "are planned once per scene and seed; rate conditions use prefixes."
                ),
                "severity_definition": "component-boundary events / clean components",
                "rates": [float(rate) for rate in rates],
                "seeds": seeds,
                "operations": list(OPERATIONS),
                "baseline_repetitions": 1,
            },
            "source_benchmark_data_dir": str(benchmark_data_dir),
            "outputs_dir": str(outputs_dir),
            "num_questions": len(questions),
            "num_scenes": len(dataset_names),
            "datasets": dataset_names,
            "artifact_protocol": {
                "recreate_crops_captions": args.recreate_crops_captions,
                "projection_min_fraction": (
                    args.projection_min_fraction
                    if args.recreate_crops_captions
                    else None
                ),
                "captioner_type": (
                    args.captioner_type if args.recreate_crops_captions else None
                ),
                "caption_model": (
                    args.caption_model if args.recreate_crops_captions else None
                ),
                "caption_device": (
                    args.caption_device if args.recreate_crops_captions else None
                ),
                "caption_n_images": (
                    args.caption_n_images if args.recreate_crops_captions else None
                ),
                "caption_batch_size": (
                    args.caption_batch_size if args.recreate_crops_captions else None
                ),
                "max_crops_per_component": (
                    args.max_crops_per_component
                    if args.recreate_crops_captions
                    else None
                ),
                "scannetpp_data_root": (
                    str(scannetpp_data_root)
                    if args.recreate_crops_captions and scannetpp_data_root is not None
                    else None
                ),
            },
            "conditions": condition_records,
        }
        write_json(temp_experiment_dir / "manifest.json", manifest)
        temp_experiment_dir.rename(experiment_dir)
    except Exception:
        if temp_experiment_dir.exists():
            shutil.rmtree(temp_experiment_dir)
        raise
    finally:
        if captioner is not None:
            try:
                captioner.cleanup()
            except Exception as exc:
                print(f"Warning: captioner cleanup failed: {exc}")

    print(f"Prepared experiment at {experiment_dir}")
    print("Existing scene memories and benchmark artifacts were not modified.")


if __name__ == "__main__":
    main()
