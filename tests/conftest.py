"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
