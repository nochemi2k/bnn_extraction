"""Re-infer fold predictions from checkpoints (wrapper)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uq_remake.training.reinfer_predictions import main

if __name__ == "__main__":
    raise SystemExit(main())
