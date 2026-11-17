"""Prepare M4 / DS1 / DS2 tables and fold-local features."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build or verify processed DS1/DS2 tables and fold features.")
    p.add_argument("--config", default="configs/data.yaml", help="Data path config (documentation pointer).")
    p.add_argument("--verify", action="store_true", help="Compare rebuilt features to saved fold files without writing.")
    args, extra = p.parse_known_args(argv)
    forwarded = list(extra)
    if args.verify:
        forwarded.append("--verify")
    target = ROOT / "src" / "uq_remake" / "data" / "prepare_data.py"
    sys.argv = [str(target), *forwarded]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
