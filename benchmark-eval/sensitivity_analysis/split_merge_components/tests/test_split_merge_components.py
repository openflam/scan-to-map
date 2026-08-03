from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("split_merge_prepare", "prepare.py")
recreate = load_module("split_merge_recreate", "recreate_crops_captions.py")
experiment = load_module("split_merge_experiment", "experiment.py")
summarize = load_module("split_merge_summarize", "summarize.py")


def bbox_record(component_id: int, minimum: list[float], maximum: list[float]):
    return {
        "connected_comp_id": component_id,
        "bbox": prepare.axis_aligned_bbox(minimum, maximum),
    }


def test_balanced_split_reconstructs_source_aabb():
    source = bbox_record(7, [0.0, 0.0, 0.0], [4.0, 2.0, 1.0])

    first, second, axis = prepare.split_bbox_record(source, (7, 20), "fixture")

    assert axis == 0
    assert first["bbox"]["min"] == [0.0, 0.0, 0.0]
    assert first["bbox"]["max"] == [2.0, 2.0, 1.0]
    assert second["bbox"]["min"] == [2.0, 0.0, 0.0]
    assert second["bbox"]["max"] == [4.0, 2.0, 1.0]


def test_axis_aligned_bbox_uses_viewer_face_ring_corner_order():
    bbox = prepare.axis_aligned_bbox([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])

    assert bbox["corners"] == [
        [0.0, 1.0, 2.0],
        [3.0, 1.0, 2.0],
        [3.0, 4.0, 2.0],
        [0.0, 4.0, 2.0],
        [0.0, 1.0, 5.0],
        [3.0, 1.0, 5.0],
        [3.0, 4.0, 5.0],
        [0.0, 4.0, 5.0],
    ]


def test_merge_plan_is_deterministic_and_disjoint():
    component_ids = {0, 1, 2, 3}
    scene = {
        "component_ids": component_ids,
        "bbox_map": {
            0: bbox_record(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            1: bbox_record(1, [1.1, 0.0, 0.0], [2.1, 1.0, 1.0]),
            2: bbox_record(2, [10.0, 0.0, 0.0], [11.0, 1.0, 1.0]),
            3: bbox_record(3, [11.1, 0.0, 0.0], [12.1, 1.0, 1.0]),
        },
    }

    first = prepare.build_merge_plan("scene", scene, seed=3)
    second = prepare.build_merge_plan("scene", scene, seed=3)

    assert first == second
    flattened = [
        component_id
        for pair in first
        for component_id in pair["source_component_ids"]
    ]
    assert len(flattened) == len(set(flattened)) == len(component_ids)


def test_source_space_metrics_penalize_merge_contamination():
    record = {
        "expected_components": ["1"],
        "predicted_components": ["10"],
    }
    scene = {"component_lineage": {"10": [1, 2], "3": [3]}}

    metrics = summarize.derived_component_metrics(record, scene)

    assert metrics["Source Precision"] == pytest.approx(0.5)
    assert metrics["Source Recall"] == pytest.approx(1.0)
    assert metrics["Source F1 Score"] == pytest.approx(2 / 3)
    assert metrics["Topology Ceiling F1"] == pytest.approx(2 / 3)
    assert metrics["Merge Contamination"] == pytest.approx(0.5)


def test_source_space_metrics_collapse_split_children_and_record_redundancy():
    record = {
        "expected_components": ["1"],
        "predicted_components": ["1", "10"],
    }
    scene = {"component_lineage": {"1": [1], "10": [1], "2": [2]}}

    metrics = summarize.derived_component_metrics(record, scene)

    assert metrics["Source F1 Score"] == pytest.approx(1.0)
    assert metrics["Split Redundancy"] == pytest.approx(0.5)


def test_merge_reference_metadata_identifies_distractor_pairing():
    perturbation = {
        "operation": "merge",
        "affected_source_component_ids": [1, 2],
        "operations": [{"source_component_ids": [1, 2]}],
    }

    metadata = prepare.reference_metadata({1}, perturbation)

    assert metadata["reference_status"] == "all_targets_merged"
    assert metadata["merge_pairing"] == "target_with_distractor"


def test_pinhole_projection_matches_project_bbox_equations():
    image = SimpleNamespace(qvec=[1.0, 0.0, 0.0, 0.0], tvec=[0.0, 0.0, 0.0])
    camera = SimpleNamespace(model="PINHOLE", params=[100.0, 100.0, 50.0, 40.0])

    coordinates, valid = recreate.project_points(
        [[0.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 0.0, -1.0]],
        image,
        camera,
    )

    assert valid == [True, True, False]
    assert coordinates[0] == pytest.approx([50.0, 40.0])
    assert coordinates[1] == pytest.approx([100.0, 90.0])


def test_opencv_fisheye_projection_matches_colmap_equation():
    image = SimpleNamespace(qvec=[1.0, 0.0, 0.0, 0.0], tvec=[0.0, 0.0, 0.0])
    camera = SimpleNamespace(
        model="OPENCV_FISHEYE",
        params=[100.0, 100.0, 50.0, 40.0, 0.0, 0.0, 0.0, 0.0],
    )

    coordinates, valid = recreate.project_points([[1.0, 0.0, 1.0]], image, camera)

    assert valid == [True]
    assert coordinates[0] == pytest.approx([50.0 + 25.0 * 3.141592653589793, 40.0])


def test_split_point_memberships_follow_split_plane(tmp_path):
    context = recreate.ProjectionContext(
        dataset_name="scene",
        images_dir=tmp_path,
        cameras={},
        images={},
        points3d={
            10: SimpleNamespace(xyz=[-1.0, 0.0, 0.0]),
            11: SimpleNamespace(xyz=[0.0, 0.0, 0.0]),
            12: SimpleNamespace(xyz=[1.0, 0.0, 0.0]),
        },
        point_to_images={},
        image_ids_by_name={},
        component_point_ids={4: [10, 11, 12]},
        source_manifest={},
        min_fraction=0.3,
    )

    first, second = recreate.split_component_point_ids(context, 4, axis=0, midpoint=0.0)

    assert first == [10, 11]
    assert second == [12]


def test_recreated_crop_is_new_file_and_is_captioned(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_module.new("RGB", (100, 80), color=(20, 40, 60)).save(
        images_dir / "frame.jpg"
    )
    camera = SimpleNamespace(
        model="PINHOLE",
        params=[100.0, 100.0, 50.0, 40.0],
        width=100,
        height=80,
    )
    source_image = SimpleNamespace(
        qvec=[1.0, 0.0, 0.0, 0.0],
        tvec=[0.0, 0.0, 0.0],
        camera_id=3,
        name="frame.jpg",
    )
    context = recreate.ProjectionContext(
        dataset_name="scene",
        images_dir=images_dir,
        cameras={3: camera},
        images={7: source_image},
        points3d={},
        point_to_images={},
        image_ids_by_name={"frame.jpg": 7},
        component_point_ids=None,
        source_manifest={
            "1": {
                "crops": [
                    {
                        "source_image": "frame.jpg",
                        "image_id": 7,
                        "visible_points": 8,
                        "total_points": 10,
                    }
                ]
            }
        },
        min_fraction=0.3,
    )
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    bbox = bbox_record(9, [-0.5, -0.4, 2.0], [0.5, 0.4, 3.0])

    manifest, projections = recreate.recreate_component_crops(
        context,
        crops_dir,
        9,
        bbox,
        [1],
        None,
    )

    crop_path = crops_dir / "component_9" / manifest["crops"][0]["crop_filename"]
    assert crop_path.is_file()
    assert not crop_path.is_symlink()
    assert projections[0]["visibility_protocol"] == "source_manifest_visibility_fallback"

    class FakeCaptioner:
        def caption_batch(self, batch_data, supplied_crops_dir):
            assert supplied_crops_dir == crops_dir
            return [
                SimpleNamespace(
                    component_id=component_id,
                    caption="fresh visual caption",
                    error=None,
                )
                for component_id, _ in batch_data
            ]

    captions = recreate.caption_recreated_components(
        SCRIPT_DIR.parents[2],
        FakeCaptioner(),
        crops_dir,
        {9: manifest},
        {9},
        n_images=1,
        batch_size=4,
    )

    assert captions[9]["caption"] == "fresh visual caption"
    assert captions[9]["crop_filenames"] == [crop_path.name]


def test_experiment_forwards_recreation_settings(tmp_path):
    command = experiment.prepare_command(
        SCRIPT_DIR,
        tmp_path / "questions",
        tmp_path / "outputs",
        tmp_path / "experiment",
        [Decimal("0"), Decimal("0.1")],
        [0],
        "scene_",
        "alias",
        True,
        True,
        0.25,
        "vllm",
        "caption-model",
        2,
        3,
        8,
        5,
        None,
    )

    assert "--recreate-crops-captions" in command
    assert command[command.index("--projection-min-fraction") + 1] == "0.25"
    assert command[command.index("--caption-model") + 1] == "caption-model"
    assert command[-1] == "--dry-run"
