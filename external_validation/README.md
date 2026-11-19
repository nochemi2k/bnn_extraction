# External-validation reproduction package

This directory reproduces the two external case studies added to the revised article. It contains preparation/training protocols, all 50 trained ensemble-member checkpoints, fixed-prediction evaluation, compact reference outputs, exact split definitions, and integrity checks.

## Source data

Download the original data separately:

- **BigSolDB v2.2:** https://doi.org/10.5281/zenodo.22648301. Required files: `BigSolDBv2.2.csv`, `BigSolDBv2.2_methods.csv`, and `BigSolDBv2.2_solid_forms.csv`.
- **Lanthanide separation:** supporting workbook `au2c00122_si_002.xlsx` from Liu et al., https://doi.org/10.1021/jacsau.2c00122. Only the historical `training set` and `validation set` sheets are used here.

The fixed prediction tables in `predictions/` are sufficient for all reported evaluations; downloading source data is necessary only to repeat curation, training, or checkpoint-inference verification.

## Environment

Exact versions recorded for the completed analyses are in `environment.json`. Fixed-prediction evaluation and verification require Python, NumPy, and pandas. Preparation additionally uses SciPy, scikit-learn, RDKit, and openpyxl as applicable. Training and checkpoint inference use PyTorch and the repository's existing `src/uq_remake/` implementation plus `configs/active_grid.json` and `configs/shared_dmpnn.json`.

Run commands from the repository root.

## Fast verification

```bash
python external_validation/verify.py
```

This checks file checksums, the presence of all 50 checkpoints, prediction counts and values, exact outer-test indices, and the principal reported metrics. It does not require PyTorch.

To recompute every retention table and all 2,000-replicate bootstrap analyses, then compare every generated reference file with a 1e-12 absolute numerical tolerance:

```bash
python external_validation/verify.py --recompute
```

To write a fresh set of analysis outputs without running the comparison:

```bash
python external_validation/evaluate.py all --output results/external_validation_reproduction
```

## Trained weights

`weights/bigsoldb/` and `weights/lanthanide/` each contain five folds with five members per fold: 25 checkpoints per dataset and 50 in total. Each checkpoint is a portable, CPU `state_dict` in `weight_io` format v1 and includes:

- the complete `HeteroMLP` architecture and strictly loadable model parameters;
- fold, member, training seed, and 30-pass MC-dropout inference settings;
- target definition and ordered feature names;
- the outer-training-fold scaler mean and scale;
- validation-based training metadata and the log-variance clamp.

The scaler arrays are part of each checkpoint. Preparation writes `arrays.npz` files whose `X_train`, `X_val`, and `X_test` matrices are already standardized with these fold-specific arrays. To featurize new raw inputs instead, construct features in the checkpoint's `feature_names` order and apply `(X - scaler_mean) / scaler_scale` before inference.

Load a member and run its specified MC-dropout inference as follows:

```python
import sys
sys.path.insert(0, "external_validation")
sys.path.insert(0, "src")

from weight_io import load_weight_checkpoint
from uq_remake.models.uq_tabular import predict_hetero_mc

model, checkpoint = load_weight_checkpoint(
    "external_validation/weights/bigsoldb/fold0/member_0.pt",
    device="cpu",
)
# X_scaled follows checkpoint["feature_names"] and uses the saved fold scaler.
result = predict_hetero_mc(
    model,
    X_scaled,
    n_mc=checkpoint["mc_test_passes"],
    device="cpu",
    seed=checkpoint["seed"] + 11,
)
```

For a five-member ensemble, aggregate the member outputs with `aggregate_members` from `uq_remake.models.uq_tabular`. The fixed ensemble tables shipped in `predictions/` remain the canonical inputs for exact numerical reproduction of the article's analyses.

## Checkpoint-inference verification

After repeating preparation, this command loads every packaged checkpoint, runs its saved inference protocol, aggregates each five-member ensemble, and compares all reported prediction and uncertainty fields against the fixed ensemble tables:

```bash
python external_validation/verify_weights.py
```

Numerical verification uses the original CUDA inference protocol. MC-dropout random streams are device-dependent, so CPU inference is valid for using a member but is not expected to reproduce the canonical CUDA ensemble draws. The default absolute tolerance is `1e-6`, accommodating float32 save/reload and hardware-level roundoff without altering the canonical prediction tables. `--bigsoldb-work-dir`, `--lanthanide-work-dir`, and `--weights-root` can override the default locations. This test requires the prepared `arrays.npz` and fold `manifest.json` files, but not model retraining.

## Repeat preparation and training

Place the three BigSolDB files in one directory and run:

```bash
python external_validation/bigsoldb.py all --source-dir /path/to/bigsoldb_v2_2
```

For the lanthanide workbook, run:

```bash
python external_validation/lanthanide.py all --source-file /path/to/au2c00122_si_002.xlsx
```

Optional `--work-dir` and `--output-dir` arguments control bulky intermediates and newly trained predictions and weights. Their defaults are `external_validation_work/` and `results/external_validation/`; they do not overwrite this compact release. Existing predictions are reused only when their corresponding checkpoint exists and row identities align. Add `--retrain` to force retraining.

Both protocols use five outer `GroupKFold` folds, grouped validation drawn from outer training, five heteroscedastic MLP ensemble members, and 30 MC-dropout passes. No new hyperparameter search is performed. Validation splitting uses seed `42 + fold`; model seeds use the repository's fixed protocol. Random retention uses 20 permutations with seed `20261117 + fold`. Retention fractions are 0.25, 0.50, and 0.75. BigSolDB bootstrap seeding starts at 20261118; lanthanide MoP bootstrap seeding starts at 20261119; each reported interval uses 2,000 replicates.

Retraining can produce small numerical differences across hardware and library builds. Checkpoint loading and the fixed prediction tables provide direct artifact-level reproduction without repeating optimization.

## Metric scope

- **BigSolDB:** primary retention metric is row-weighted RMSE. The dummy is the outer-training solvent mean, falling back to the global training mean. Tanimoto uses nearest training-solute Morgan fingerprints; Mahalanobis distance uses standardized joint solute/solvent descriptors and excludes temperature. Bootstrap intervals resample solutes within folds.
- **Lanthanide log separation factor:** the target is `logD_x - logD_y` in increasing-atomic-number orientation. Primary retention metric is mean-of-metal-pairs RMSE (MoP RMSE), with at least three retained rows per pair. The dummy is the outer-training metal-pair mean, falling back to the global training mean. Tanimoto uses released ligand ECFP bits; Mahalanobis distance excludes ECFP and uses the standardized joint ligand-RDKit, condition, and both-metal descriptor block. MoP intervals resample metal pairs within folds. A 50% common-eligibility sensitivity uses the intersection of pairs eligible under all criteria and random replicates.

All intervals condition on fixed selected sets and exclude model-selection and retraining variability. The analyses quantify dataset-specific risk ordering and calibration.

## Directory map

- `bigsoldb.py`, `lanthanide.py`: cleaned source-data preparation and fixed-protocol training with checkpoint saving.
- `weight_io.py`: portable checkpoint serialization and strict, safe loading.
- `evaluate.py`: all fixed-prediction metrics, retention analyses, and bootstrap procedures.
- `verify.py`: lightweight integrity and checkpoint-manifest checks plus optional exact full recomputation.
- `verify_weights.py`: checkpoint loading, inference, aggregation, and numerical comparison with fixed predictions.
- `common.py`: shared constants, paths, and prediction-loading checks.
- `data/`: exact splits, feature schemas, compact fold manifests, preparation audits, and training specifications.
- `weights/`: 50 trained member checkpoints, including feature and preprocessing metadata.
- `predictions/`: five compressed ensemble tables per dataset, enriched with cohort indices, dummy predictions, and analysis fields. Member-level predictions are not needed for the reported analyses.
- `reference/`: compact reference fold metrics, complete random-replicate retention metrics, bootstrap outputs, conditional diagnostics, eligibility accounting, and common-population sensitivities.
- `environment.json`: actual package versions used for the analyses.
- `SHA256SUMS`: SHA-256 hashes for every other file in this directory.
