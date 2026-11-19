# Applicability Domains as Empirical Claims

## Testing Similarity and Uncertainty Against Prediction Error

**Ivan Zlobin, Ivan Bespalov, Mark Schneider, Maksim Shandybo, Nikolai Rekut, and Alexander Pavlov**

This repository accompanies the manuscript *Applicability Domains as Empirical Claims: Testing Similarity and Uncertainty Against Prediction Error*. It provides the models, data-processing protocols, held-out evaluations, trained external-model weights, fixed predictions, and reference outputs needed to examine and reproduce the study.

### Preprint and current version

The original preprint is available as [ChemRxiv v1](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15010241/v1).

**The current repository contains improvements made for the resubmission and therefore extends the v1 preprint.** In particular, it adds focused experimental validation on solute-disjoint BigSolDB solubility and ligand-disjoint lanthanide log-separation-factor prediction, together with all 50 external ensemble-member checkpoints, exact splits, fixed predictions, and independent reproduction checks. The revised study also clarifies the empirical risk-ordering criterion, the eligibility-varying mean-of-pairs baseline, and the scope of the prospective-style ligand-selection replay.

## Study overview

An applicability domain is commonly defined by resemblance to a training set. This study instead treats an applicability criterion as a testable claim about prediction error: predictions ranked as safer should have lower held-out error than an equally sized random subset.

The primary analysis compares 23 similarity-, distance-, support-, and uncertainty-based criteria using a fixed direct-selectivity model and five ligand-disjoint folds. Focused external case studies examine whether the resulting risk ordering transfers to two independently published, context-dependent experimental datasets. A prospective-style replay then tests how predictions and uncertainties can be combined to select ligands for a specified selectivity objective.

**Primary model:** `DS2_RDKIT_MLPENS5_HETERO`, a five-member heteroscedastic MLP ensemble using RDKit descriptors and Monte-Carlo dropout at inference. The model was selected by the prespecified protocol and fixed before the applicability-criterion analysis.

**Primary validation:** five-fold ligand-disjoint `GroupKFold`. Reported fold summaries use the sample standard deviation (`ddof=1`).

## Repository contents

| Path | Role |
|------|------|
| `src/uq_remake/models/` | Heteroscedastic, evidential, and MC-dropout MLPs; variational BNN; SNGP; sparse GPR; Chemprop DMPNN wrapper |
| `src/uq_remake/data/` | M4/DS1/DS2 construction from the raw CMI tables, RDKit features, and fold definitions |
| `src/uq_remake/training/` | Training protocol and active-grid trainer |
| `src/uq_remake/metrics/` | Point and uncertainty metrics, MoP aggregation, cross-validation, and model-selection rule |
| `scripts/` | Command-line entry points for data preparation, training, evaluation, and artifact unpacking |
| `configs/` | Experiment grid, model registry, and shared architecture/evaluation configurations |
| `data/raw_m4/`, `data/splits/` | Raw CMI tables and ligand-disjoint fold definitions |
| `results/` | Canonical tables and JSON summaries underlying the reported analyses |
| `artifacts/` | Git LFS archives containing trained primary-study weights, predictions, processed tables, and row-level scores |
| `external_validation/` | External-data protocols, 50 trained checkpoints, fixed predictions, exact splits, reference results, and verification commands |
| `checkpoints/cmi_published/` | Provenance record for the published CMI model checkpoint |

## Quick verification

Run commands from the repository root.

Verify the primary-study checkpoint registry and stored metrics:

```bash
python scripts/evaluate.py
```

Verify the complete external-validation package, including file checksums, all 50 checkpoints, fixed predictions, split identities, and principal metrics:

```bash
python external_validation/verify.py
```

Recompute every external retention table and all 2,000-replicate bootstrap analyses, then compare them with the packaged reference outputs:

```bash
python external_validation/verify.py --recompute
```

See [`external_validation/README.md`](external_validation/README.md) for checkpoint inference, source-data preparation, and retraining protocols.

## Environment

Python **3.11** is recommended. Import **PyTorch before RDKit**.

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt   # training
# or: python -m pip install -r requirements.txt  # evaluation only

$env:PYTHONPATH = "$PWD\src"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"
```

Primary-study trained weights, fold predictions, processed tables, and row-level scores are stored in Git LFS archives under `artifacts/`:

```powershell
git lfs install
git lfs pull
powershell -File scripts/unpack_artifacts.ps1
```

The external-validation checkpoints and compact fixed-prediction tables are provided directly in `external_validation/`.

## Reproduction

### Primary study from raw CMI data

Training the model grid requires a GPU:

```powershell
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
```

### Primary study from trained weights

Unpack `artifacts/checkpoints.7z` and, optionally, `cv_predictions.7z`, then run:

```powershell
python scripts/evaluate.py
python scripts/evaluate.py --from-predictions
```

### External case studies

The fixed prediction tables are sufficient to reproduce all reported external evaluations. Source-data downloads are needed only to repeat curation, training, or checkpoint-inference verification:

- **BigSolDB v2.2:** https://doi.org/10.5281/zenodo.22648301
- **Lanthanide separation:** supporting workbook from Liu et al., https://doi.org/10.1021/jacsau.2c00122

Preparation, training, weight loading, and inference commands are documented in [`external_validation/README.md`](external_validation/README.md).

## Locating reported quantities

| Quantity | Artifact |
|----------|----------|
| Published-split ligand overlap | `results/canonical/cmi_split_ligand_overlap.json` |
| Published-model reproduction | `results/stage1/cmi_reproduction_results.json` |
| Workhorse MoP RMSE, PICP, and ACE | `results/cv_metrics/fold_metric_summary.csv` |
| Workhorse selection rule and audit | `results/model_audit/` |
| Primary-study retention summary | `results/downstream_cv/ad_retention/retention_summary.csv` |
| Prospective-style pair-acquisition summary | `results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv` |
| BigSolDB main-text metrics | `external_validation/reference/bigsoldb/main_text_summary.json` |
| Lanthanide main-text metrics | `external_validation/reference/lanthanide/main_text_evaluation/main_text_summary.json` |
| Lanthanide MoP retention and bootstrap analysis | `external_validation/reference/lanthanide/mop_inference/` |

Row-level primary-study material is listed in `artifacts/README.md`; external reference outputs are indexed in `external_validation/SHA256SUMS`.

## Citation

Preprint: [ChemRxiv v1, DOI 10.26434/chemrxiv.15010241/v1](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15010241/v1).

The ChemRxiv record describes the original version. The current repository includes the subsequent resubmission improvements described above. See `CITATION.cff` for repository citation metadata.
