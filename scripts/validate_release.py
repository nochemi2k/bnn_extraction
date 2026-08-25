"""Validate release checkout (wrapper)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402,F401  — before rdkit on Windows

from uq_remake.release.validate_release import main

if __name__ == "__main__":
    raise SystemExit(main())
