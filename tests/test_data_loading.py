"""Load ready-to-use DS1/DS2 tables from a fresh checkout."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.training.protocol import load_table


class TestDataLoading(unittest.TestCase):
    def test_ds1_loads(self):
        df = load_table("DS1")
        self.assertGreater(len(df), 1000)
        self.assertIn("record_id", df.columns)

    def test_ds2_loads(self):
        df = load_table("DS2")
        self.assertGreater(len(df), 1000)
        self.assertIn("pair_id", df.columns)
        self.assertIn("selectivity_logK", df.columns)


if __name__ == "__main__":
    unittest.main()
