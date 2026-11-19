"""Shared paths, constants, and I/O for the external-validation release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
FOLDS = [f"fold{i}" for i in range(5)]
RETENTIONS = (0.25, 0.50, 0.75)
RANDOM_REPEATS = 20
RANDOM_SEED_BASE = 20261117
BOOTSTRAP_REPLICATES = 2000
BIGSOLDB_BOOTSTRAP_SEED = 20261118
LANTHANIDE_MOP_BOOTSTRAP_SEED = 20261119
Z90 = 1.6448536269514722


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_predictions(dataset: str) -> list[pd.DataFrame]:
    """Load the five enriched ensemble tables shipped with this package."""
    if dataset not in {"bigsoldb", "lanthanide"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    frames = []
    for fold in FOLDS:
        path = PACKAGE_ROOT / "predictions" / dataset / f"{fold}.csv.gz"
        frame = pd.read_csv(path)
        for column in ("training_solute_support_bin", "metal_pair_training_support_bin", "diluent_training_support_bin"):
            if column in frame:
                frame[column] = pd.Categorical(
                    frame[column], categories=["0", "1-4", "5-19", "20-99", ">=100"], ordered=True,
                )
        if frame["fold_id"].astype(str).unique().tolist() != [fold]:
            raise ValueError(f"{path}: fold identifier mismatch")
        frames.append(frame)
    return frames


def validate_prediction_indices(dataset: str, frames: list[pd.DataFrame]) -> None:
    """Check exact row order and once-only outer-test coverage from packaged indices."""
    splits = read_json(PACKAGE_ROOT / "data" / dataset / "splits.json")
    all_test: list[int] = []
    for fold, frame in zip(FOLDS, frames):
        expected = splits[fold]["test_idx"]
        observed = frame["cohort_index"].astype(int).tolist()
        if observed != expected:
            raise ValueError(f"{dataset}/{fold}: packaged prediction order differs from test_idx")
        all_test.extend(observed)
    if sorted(all_test) != list(range(len(all_test))):
        raise ValueError(f"{dataset}: outer-test predictions do not cover the cohort exactly once")
