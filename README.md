# Applicability domains as testable claims

Code and data behind *Applicability domains as testable claims: predictive uncertainty
against molecular similarity for metal–ligand selectivity*.

An applicability domain is usually a region of resemblance to the training set. This
repository holds the model, data and evaluation stack used to test it as a claim about
error instead: candidate predictive-uncertainty scores are checked against the held-out
error of a single frozen model, on ligand-disjoint folds.

**Workhorse:** `DS2_RDKIT_MLPENS5_HETERO`, a five-member heteroscedastic MLP ensemble on
RDKit descriptors with Monte-Carlo dropout at inference, selected and frozen before any
applicability-domain analysis was designed.
**Validation:** five-fold ligand-disjoint `GroupKFold`; every statistic is a fold mean ± s.d.
(`ddof=1`).

## Repository contents

| Path | Role |
|------|------|
| `src/uq_remake/models/` | Model zoo: heteroscedastic / evidential / MC-dropout MLPs (single and 5-member ensembles), full variational BNN, SNGP, sparse GPR, Chemprop DMPNN wrapper |
| `src/uq_remake/data/` | M4 / DS1 / DS2 dataset construction from the raw CMI tables, RDKit feature matrices, fold splits |
| `src/uq_remake/training/` | Training protocol and the active-grid trainer (`run_active_grid.py`) |
| `src/uq_remake/metrics/` | Fold-level point and UQ metrics (RMSE, MoP RMSE, PICP, ACE, …), CV protocol, workhorse selection rule |
| `scripts/` | Thin CLIs: `prepare_data.py`, `train.py`, `evaluate.py`, plus `unpack_artifacts.ps1` for the LFS archives |
| `configs/` | Grid definition, experiment registry, shared architecture and evaluation configs |
| `data/raw_m4/`, `data/splits/` | Raw CMI tables and the ligand-disjoint fold definitions (shipped in git) |
| `results/` | Canonical summary tables and JSONs behind the paper's numbers |
| `artifacts/` | Git LFS archives: trained weights, fold predictions, processed tables, row-level scores. See `artifacts/README.md` |
| `checkpoints/cmi_published/` | Provenance record for the published CMI model checkpoint |

## Environment

Python **3.11**. Import **torch before RDKit**.

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

Trained weights, fold dumps, processed tables and row-level scores live in Git LFS
archives under `artifacts/`, not unpacked in the repository:

```powershell
git lfs install
git lfs pull
powershell -File scripts/unpack_artifacts.ps1
```

## Reproduction

### From the raw CMI tables (needs a GPU)

```powershell
python scripts/prepare_data.py     # M4 / DS1 / DS2 tables, folds, feature matrices
python scripts/train.py            # train the active grid (configs/active_grid.json)
python scripts/evaluate.py         # fold-level metrics -> results/cv_metrics/
```

### From the trained weights

Unpack `artifacts/checkpoints.7z` (and optionally `cv_predictions.7z` to skip
re-inference), then:

```powershell
python scripts/evaluate.py                 # validate stored metrics
python scripts/evaluate.py --from-predictions   # recompute metrics from fold predictions and compare
```

`scripts/evaluate.py` with no arguments checks the stored `results/cv_metrics/` against
the checkpoint registry and exits non-zero on any mismatch — that is the checkout check.

## Where is this number?

| Quantity | Artifact |
|----------|----------|
| Published-split ligand overlap, 93.9% / 97.5% | `results/canonical/cmi_split_ligand_overlap.json` |
| Published-model reproduction | `results/stage1/cmi_reproduction_results.json` |
| H+ inclusion decision | `results/canonical/hplus_decision.json` |
| Workhorse MoP RMSE, PICP, ACE | `results/cv_metrics/fold_metric_summary.csv` |
| Workhorse selection rule and audit | `results/model_audit/` |
| Grid risk ordering (σ_total) | `results/canonical/grid_sigma_total_risk_ordering.csv` |
| Workhorse versus runner-up paired difference | `results/canonical/workhorse_fold_paired_difference_summary.csv` |
| Pooled retention summary | `results/downstream_cv/ad_retention/retention_summary.csv` |
| Batch (pair-acquisition) summary | `results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv` |

Row-level material behind every summary (fold scores, prediction dumps) is in the LFS
archives listed in `artifacts/README.md`.

## Citation

See `CITATION.cff`.
