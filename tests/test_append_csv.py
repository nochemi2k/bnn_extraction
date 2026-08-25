"""Guard append_csv against pandas FutureWarning on mixed-NA frames."""
from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uq_remake.training.write_model_audit import append_csv


class TestAppendCsv(unittest.TestCase):
    def test_mixed_keys_no_futurewarning(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runs.csv"
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                append_csv(path, {"model_id": "A", "status": "trained"})
                append_csv(path, {"model_id": "B", "status": "trained", "lambda_reg": 0.2, "val_nll": 1.1})
                append_csv(path, {"model_id": "C", "status": "aggregated", "n_members": 5})
            import pandas as pd

            df = pd.read_csv(path)
            self.assertEqual(len(df), 3)
            self.assertEqual(df.iloc[1]["lambda_reg"], 0.2)


if __name__ == "__main__":
    unittest.main()
