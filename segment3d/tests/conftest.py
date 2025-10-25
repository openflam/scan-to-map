"""
Pytest configuration for segment3d tests.
"""

import sys
from pathlib import Path

# Add the src directory to Python path for all tests
SEGMENT3D_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SEGMENT3D_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))