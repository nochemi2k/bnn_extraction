"""Train the active DS1/DS2 grid into checkpoints/active/."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the shared DS1/DS2 active grid into checkpoints/active/.")
    p.add_argument("--config", default="configs/default.yaml", help="Training config pointer.")
    p.add_argument("--audit-only", action="store_true", help="Write model inventory only; do not train.")
    args, extra = p.parse_known_args(argv)
    forwarded = list(extra)
    if args.audit_only:
        forwarded.append("--audit-only")
    sys.argv = [str(ROOT / "src" / "uq_remake" / "training" / "run_active_grid.py"), *forwarded]
    runpy.run_path(str(ROOT / "src" / "uq_remake" / "training" / "run_active_grid.py"), run_name="__main__")


if __name__ == "__main__":
    main()
