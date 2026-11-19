"""Verify package integrity, key reported values, and optional full recomputation."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import PACKAGE_ROOT, REPOSITORY_ROOT, load_fixed_predictions, read_json, sha256, validate_prediction_indices
from evaluate import evaluate_all


def check_checksums() -> int:
    checksum_file = PACKAGE_ROOT / "SHA256SUMS"
    if not checksum_file.exists():
        raise FileNotFoundError("SHA256SUMS is missing")
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = PACKAGE_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def check_weight_manifest() -> int:
    """Check distributed checkpoint presence and metadata without importing torch."""
    expected = []
    for dataset in ("bigsoldb", "lanthanide"):
        spec = read_json(PACKAGE_ROOT / "data" / dataset / "training_spec.json")
        if spec.get("weights_saved") is not True or "weight_format" not in spec:
            raise ValueError(f"{dataset}: training specification does not describe distributed weights")
        for fold in ("fold0", "fold1", "fold2", "fold3", "fold4"):
            for member in range(5):
                relative = Path("weights") / dataset / fold / f"member_{member}.pt"
                path = PACKAGE_ROOT / relative
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing or empty checkpoint: {relative}")
                expected.append(relative.as_posix())
    actual = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "weights").rglob("*.pt")
    )
    if actual != sorted(expected):
        raise ValueError("Checkpoint directory contains an unexpected file manifest")
    return len(expected)


def check_predictions() -> dict:
    expected_counts = {"bigsoldb": 119043, "lanthanide": 6298}
    result = {}
    for dataset, expected_count in expected_counts.items():
        frames = load_fixed_predictions(dataset)
        validate_prediction_indices(dataset, frames)
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != expected_count:
            raise ValueError(f"{dataset}: expected {expected_count} rows, found {len(combined)}")
        required = ["y_true", "y_pred", "dummy_pred", "sigma_total"]
        if dataset == "lanthanide":
            required += ["tanimoto_nn_distance", "mahalanobis_feature_centroid"]
        if not np.isfinite(combined[required]).all().all() or not (combined["sigma_total"] > 0).all():
            raise ValueError(f"{dataset}: non-finite or invalid prediction fields")
        result[dataset] = {"rows": len(combined), "folds": len(frames), "indices_and_values_valid": True}
    return result


def close(observed: float, expected: float) -> None:
    if not np.isclose(observed, expected, rtol=0, atol=1e-12):
        raise ValueError(f"Reference-value mismatch: {observed} != {expected}")


def check_reference_values() -> None:
    big = read_json(PACKAGE_ROOT / "reference/bigsoldb/main_text_summary.json")
    close(big["full_model"]["row_rmse_fold_mean"], 0.8755598812714676)
    close(big["training_solvent_mean_dummy"]["row_rmse_fold_mean"], 1.1479619412450037)
    close(big["half_retention"]["uncertainty"]["row_rmse_fold_mean"], 0.7547895002453189)
    close(big["half_retention"]["random"]["row_rmse_fold_mean"], 0.8755218163785399)
    big_boot = pd.read_csv(PACKAGE_ROOT / "reference/bigsoldb/retention_cluster_bootstrap.csv")
    row = big_boot[(big_boot["retention"] == 0.5) & (big_boot["comparator"] == "random")].iloc[0]
    close(row["point_uncertainty_minus_comparator_rmse"], -0.120732316133221)
    close(row["ci95_low"], -0.143217539095152)
    close(row["ci95_high"], -0.0972499016339854)

    ln = read_json(PACKAGE_ROOT / "reference/lanthanide/main_text_evaluation/main_text_summary.json")
    close(ln["full_model"]["metal_pair_balanced_rmse_fold_mean"], 0.6829936481201775)
    close(ln["training_metal_pair_mean_dummy"]["metal_pair_balanced_rmse_fold_mean"], 0.8886776505225977)
    mop = pd.read_csv(PACKAGE_ROOT / "reference/lanthanide/mop_inference/mop_cluster_bootstrap.csv")
    row = mop[(mop["retention"] == 0.5) & (mop["comparator"] == "random")].iloc[0]
    close(row["point_uncertainty_minus_comparator_mop_rmse"], -0.1966233193941016)
    close(row["ci95_low"], -0.22693212884002553)
    close(row["ci95_high"], -0.16388237123717014)
    common = pd.read_csv(PACKAGE_ROOT / "reference/lanthanide/mop_inference/common_eligibility_summary.csv").set_index("strategy")
    close(common.loc["uncertainty", "mean"], 0.4591612432508549)
    close(common.loc["random", "mean"], 0.5699444760571745)


def assert_json_close(observed, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or observed.keys() != expected.keys():
            raise ValueError(f"JSON structure mismatch at {path}")
        for key in expected:
            assert_json_close(observed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"JSON list mismatch at {path}")
        for index, (left, right) in enumerate(zip(observed, expected)):
            assert_json_close(left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(observed, (int, float)) or not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise ValueError(f"JSON numeric mismatch at {path}: {observed} != {expected}")
    elif observed != expected:
        raise ValueError(f"JSON value mismatch at {path}: {observed} != {expected}")


def compare_file(candidate: Path, reference: Path) -> None:
    if candidate.suffix == ".json":
        assert_json_close(
            json.loads(candidate.read_text(encoding="utf-8")),
            json.loads(reference.read_text(encoding="utf-8")),
        )
    elif candidate.suffix in {".csv", ".gz"}:
        observed = pd.read_csv(candidate)
        expected = pd.read_csv(reference)
        try:
            pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
        except AssertionError as exc:
            raise ValueError(f"Table mismatch beyond 1e-12 absolute tolerance: {candidate}") from exc
    else:
        if candidate.read_bytes() != reference.read_bytes():
            raise ValueError(f"File mismatch: {candidate}")


def compare_reproduction(candidate_root: Path) -> int:
    reference_root = PACKAGE_ROOT / "reference"
    reference_files = sorted(path.relative_to(reference_root) for path in reference_root.rglob("*") if path.is_file())
    candidate_files = sorted(path.relative_to(candidate_root) for path in candidate_root.rglob("*") if path.is_file())
    if candidate_files != reference_files:
        raise ValueError(f"Reproduction file manifest differs: {candidate_files} != {reference_files}")
    for relative in reference_files:
        compare_file(candidate_root / relative, reference_root / relative)
    return len(reference_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="also recompute and compare all reference tables at 1e-12 tolerance")
    args = parser.parse_args()
    checksum_count = check_checksums()
    checkpoint_count = check_weight_manifest()
    predictions = check_predictions()
    check_reference_values()
    compared = 0
    if args.recompute:
        with tempfile.TemporaryDirectory(prefix=".external_validation_check_", dir=REPOSITORY_ROOT) as temporary:
            candidate = Path(temporary)
            evaluate_all(candidate)
            compared = compare_reproduction(candidate)
    print(json.dumps({
        "status": "passed", "checksum_files": checksum_count,
        "weight_checkpoints": checkpoint_count,
        "predictions": predictions, "key_reference_values": "passed",
        "fully_recomputed_reference_files": compared,
    }, indent=2))


if __name__ == "__main__":
    main()
