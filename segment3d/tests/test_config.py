from __future__ import annotations

import sys
from pathlib import Path


SEGMENT3D_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SEGMENT3D_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import io_paths


def test_config_contains_required_keys() -> None:
    config = io_paths.load_config(SEGMENT3D_ROOT / "config.yaml")
    for key in io_paths.REQUIRED_KEYS:
        assert key in config, f"Missing config key: {key}"
