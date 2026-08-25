"""Thin CLI: run the full remake pipeline from repo root."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "uq_remake" / "run_full_pipeline.py")
runpy.run_path(str(ROOT / "src" / "uq_remake" / "run_full_pipeline.py"), run_name="__main__")
