"""Evaluate stored fold metrics, or recompute from fold prediction dumps."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Validate stored CV metrics (default) or, if fold predictions are present, "
            "recompute metrics and compare them to results/cv_metrics/."
        )
    )
    p.add_argument("--config", default="configs/evaluation.yaml")
    p.add_argument(
        "--from-predictions",
        action="store_true",
        help="Recompute from results/active_grid or results/cv_predictions if those trees exist.",
    )
    args, extra = p.parse_known_args(argv)
    target = ROOT / "src" / "uq_remake" / "metrics" / "evaluate_cv.py"
    forwarded = list(extra)
    if args.from_predictions:
        forwarded.append("--compare-stored")
    else:
        forwarded.append("--stored-only")
    sys.argv = [str(target), *forwarded]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
