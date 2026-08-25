"""Canonical retention analyses use the 10-100% 11-point grid only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.downstream.protocol import RETENTION_FRACTIONS

CANONICAL = {0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00}
ACTIVE_TEXT_ROOTS = [
    REPO / "configs",
    REPO / "src" / "uq_remake",
    REPO / "scripts",
]


class TestRetentionGrid(unittest.TestCase):
    def test_protocol_grid(self):
        self.assertEqual(set(RETENTION_FRACTIONS), CANONICAL)
        self.assertEqual(len(RETENTION_FRACTIONS), 11)

    def test_ad_retention_spec_grid(self):
        spec = yaml.safe_load((REPO / "configs" / "experiments" / "ad_retention_spec.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(float(x) for x in spec["retention_grid"]), CANONICAL)

    def test_fold_retention_uses_canonical_grid(self):
        path = REPO / "results" / "downstream_cv" / "ad_retention" / "fold_retention.csv"
        if not path.exists():
            self.skipTest("fold_retention.csv not yet rebuilt")
        df = pd.read_csv(path)
        fracs = set(pd.to_numeric(df["retention_fraction"], errors="coerce").dropna().round(2))
        self.assertEqual(fracs, CANONICAL)

    def test_no_legacy_50_100_in_active_code(self):
        banned = ("AURC_50_100", "ad_fold_risk_coverage.csv")
        hits = []
        for root in ACTIVE_TEXT_ROOTS:
            for path in root.rglob("*"):
                if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md"}:
                    continue
                if "archive" in path.parts or "__pycache__" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for token in banned:
                    if token in text:
                        hits.append(f"{path.relative_to(REPO)}:{token}")
        self.assertEqual(hits, [], "legacy 50-100% tokens remain in active files")


if __name__ == "__main__":
    unittest.main()
