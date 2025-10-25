"""Runtime check that configuration paths are valid."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _prepare_imports() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root


PROJECT_ROOT = _prepare_imports()

from io_paths import (
    get_colmap_model_dir,
    get_device,
    get_images_dir,
    get_labels_dir,
    get_masks_dir,
    get_outputs_dir,
    get_sam_checkpoint,
    load_config,
)
from utils.logging import get_logger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate segment3d environment")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to the config.yaml file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logger = get_logger("segment3d.check_env")

    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    try:
        images_dir = get_images_dir(config)
        colmap_dir = get_colmap_model_dir(config)
        sam_ckpt = get_sam_checkpoint(config)
        masks_dir = get_masks_dir(config)
        labels_dir = get_labels_dir(config)
        outputs_dir = get_outputs_dir(config)
    except Exception as exc:
        logger.error("Validation error: %s", exc)
        return 1

    logger.info("images_dir: %s", images_dir)
    logger.info("colmap_model_dir: %s", colmap_dir)
    logger.info("sam_ckpt: %s", sam_ckpt)
    logger.info("masks_dir: %s", masks_dir)
    logger.info("labels_dir: %s", labels_dir)
    logger.info("outputs_dir: %s", outputs_dir)
    logger.info("device: %s", get_device(config))

    if not os.access(outputs_dir, os.W_OK):
        logger.error("outputs_dir is not writable: %s", outputs_dir)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
