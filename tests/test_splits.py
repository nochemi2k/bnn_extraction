"""Ligand-disjoint project folds."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.training.protocol import load_folds, load_table, record_ids_for_split


class TestSplits(unittest.TestCase):
    def test_five_folds_and_disjoint_test_ids(self):
        for ds in ("DS1", "DS2"):
            table = load_table(ds)
            folds = load_folds(ds)
            self.assertEqual(sorted(folds), ["fold0", "fold1", "fold2", "fold3", "fold4"])
            seen = []
            for fold in sorted(folds):
                ids = list(record_ids_for_split(table, ds, folds[fold]["test_idx"]))
                self.assertTrue(ids, fold)
                seen.append(set(map(str, ids)))
            for i, a in enumerate(seen):
                for j, b in enumerate(seen):
                    if i >= j:
                        continue
                    self.assertFalse(a & b, f"{ds} fold{i} ∩ fold{j}")


if __name__ == "__main__":
    unittest.main()
