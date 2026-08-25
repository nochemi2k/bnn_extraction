"""Tests for native Chemprop evidential validation (no clipping, no repair)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uq_remake.models.chemprop_evidential_validate import (
    NATIVE_VAR_ALEATORIC,
    NATIVE_VAR_EPISTEMIC,
    fold_uq_status,
    validate_native_evidential_output,
)


class TestValidateNativeEvidential(unittest.TestCase):
    def test_valid_row_uses_chemprop_formulas(self):
        out = validate_native_evidential_output(
            {"mu": [1.0], "nu": [2.0], "alpha": [3.0], "beta": [4.0]}
        )
        self.assertTrue(out["point_valid"][0])
        self.assertTrue(out["uq_valid"][0])
        self.assertEqual(out["uncertainty_status"][0], "valid_native_nig")
        self.assertAlmostEqual(out["var_aleatoric"][0], 4.0 / (3.0 - 1.0))
        self.assertAlmostEqual(out["var_epistemic"][0], 4.0 / (2.0 * (3.0 - 1.0)))
        self.assertEqual(NATIVE_VAR_ALEATORIC, "beta / (alpha - 1)")
        self.assertEqual(NATIVE_VAR_EPISTEMIC, "beta / (nu * (alpha - 1))")

    def test_invalid_params_preserved_variances_nan(self):
        native = {
            "mu": [1.0, 2.0, 3.0, np.nan],
            "nu": [1.0, 0.0, 2.0, 1.0],
            "alpha": [2.0, 3.0, 1.0, 2.0],
            "beta": [0.5, 1.0, 1.0, 1.0],
        }
        out = validate_native_evidential_output(native)
        np.testing.assert_array_equal(out["nu"], native["nu"])
        np.testing.assert_array_equal(out["alpha"], native["alpha"])
        np.testing.assert_array_equal(out["beta"], native["beta"])
        self.assertTrue(out["uq_valid"][0])
        self.assertFalse(out["uq_valid"][1])
        self.assertFalse(out["uq_valid"][2])
        self.assertFalse(out["point_valid"][3])
        self.assertFalse(np.isfinite(out["var_total"][1]))
        self.assertFalse(np.isfinite(out["var_total"][2]))
        self.assertEqual(out["n_uq_invalid"], 3)
        self.assertEqual(out["uncertainty_status"][1], "invalid_native_nig")
        self.assertEqual(out["uncertainty_status"][3], "invalid_point")

    def test_does_not_clip_alpha(self):
        out = validate_native_evidential_output(
            {"mu": [0.0], "nu": [1.0], "alpha": [0.5], "beta": [1.0]}
        )
        self.assertEqual(out["alpha"][0], 0.5)
        self.assertFalse(out["uq_valid"][0])
        self.assertTrue(np.isnan(out["var_aleatoric"][0]))

    def test_fold_status(self):
        self.assertEqual(fold_uq_status(n_point_invalid=0, n_uq_invalid=0, n_rows=10, aligned=True), "complete")
        self.assertEqual(fold_uq_status(n_point_invalid=0, n_uq_invalid=3, n_rows=10, aligned=True), "partial_invalid_rows")
        self.assertEqual(fold_uq_status(n_point_invalid=0, n_uq_invalid=0, n_rows=10, aligned=False), "invalid_fold")


if __name__ == "__main__":
    unittest.main()
