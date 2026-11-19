"""Verify that packaged external checkpoints reproduce the fixed ensemble predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FOLDS, PACKAGE_ROOT, REPOSITORY_ROOT, read_json

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # before other numerical libraries on Windows

from uq_remake.models.uq_tabular import aggregate_members, predict_hetero_mc
from weight_io import FORMAT_VERSION, load_weight_checkpoint

ENSEMBLE_FIELDS = [
    "y_pred", "var_aleatoric", "var_epistemic_within", "var_epistemic_between",
    "var_epistemic_total", "var_total", "sigma_aleatoric",
    "sigma_epistemic_total", "sigma_total",
]


def verify_dataset(dataset: str, work_dir: Path, weights_root: Path, device: str, atol: float) -> dict:
    feature_names = read_json(PACKAGE_ROOT / "data" / dataset / "feature_names.json")["feature_names"]
    maxima = {field: 0.0 for field in ENSEMBLE_FIELDS}
    rows = 0
    for fold in FOLDS:
        arrays = np.load(work_dir / fold / "arrays.npz")
        expected = pd.read_csv(PACKAGE_ROOT / "predictions" / dataset / f"{fold}.csv.gz")
        if len(expected) != len(arrays["X_test"]):
            raise ValueError(f"{dataset}/{fold}: test-array and packaged-prediction lengths differ")
        manifest = read_json(work_dir / fold / "manifest.json")
        member_outputs = []
        for member in range(5):
            path = weights_root / dataset / fold / f"member_{member}.pt"
            model, checkpoint = load_weight_checkpoint(path, device=device)
            if checkpoint["fold"] != fold or checkpoint["member"] != member:
                raise ValueError(f"{path}: fold/member metadata mismatch")
            if checkpoint["format_version"] != FORMAT_VERSION or checkpoint["mc_test_passes"] != 30:
                raise ValueError(f"{path}: inference-protocol metadata mismatch")
            if checkpoint["feature_names"] != feature_names or checkpoint["in_dim"] != arrays["X_test"].shape[1]:
                raise ValueError(f"{path}: feature schema mismatch")
            for key in ("scaler_mean", "scaler_scale"):
                saved = checkpoint[key].detach().cpu().numpy()
                reference = np.asarray(manifest[key], dtype=np.float32)
                if not np.array_equal(saved, reference):
                    raise ValueError(f"{path}: {key} differs from the fold manifest")
            output = predict_hetero_mc(
                model, arrays["X_test"], n_mc=checkpoint["mc_test_passes"],
                device=device, seed=checkpoint["seed"] + 11,
            )
            member_outputs.append(output)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        aggregated = aggregate_members(member_outputs)
        for field in ENSEMBLE_FIELDS:
            output_field = "mu" if field == "y_pred" else field
            observed = np.asarray(aggregated[output_field], dtype=float)
            reference = expected[field].to_numpy(dtype=float)
            maximum = float(np.max(np.abs(observed - reference), initial=0.0))
            maxima[field] = max(maxima[field], maximum)
            if not np.isfinite(maximum) or maximum > atol:
                raise ValueError(
                    f"{dataset}/{fold}/{field}: checkpoint inference differs from fixed predictions "
                    f"by {maximum:.3e} (tolerance {atol:.1e})"
                )
        rows += len(expected)
        print(f"[verified] {dataset}/{fold}: {len(expected)} rows; all fields within {atol:.1e}", flush=True)
    return {"members": 25, "rows": rows, "max_absolute_difference": maxima, "tolerance": atol}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bigsoldb-work-dir", type=Path, default=REPOSITORY_ROOT / "external_validation_work/bigsoldb")
    parser.add_argument("--lanthanide-work-dir", type=Path, default=REPOSITORY_ROOT / "external_validation_work/lanthanide")
    parser.add_argument("--weights-root", type=Path, default=PACKAGE_ROOT / "weights")
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="canonical MC-dropout inference device")
    parser.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance for float32 save/reload inference")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Numerical checkpoint verification requires the original CUDA inference protocol")
    result = {
        "bigsoldb": verify_dataset("bigsoldb", args.bigsoldb_work_dir, args.weights_root, args.device, args.atol),
        "lanthanide": verify_dataset("lanthanide", args.lanthanide_work_dir, args.weights_root, args.device, args.atol),
    }
    print(json.dumps({"status": "passed", "checkpoints": 50, "device": args.device, "datasets": result}, indent=2))


if __name__ == "__main__":
    main()
