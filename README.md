# Applicability domains as testable claims

Code and data behind *Applicability domains as testable claims: predictive uncertainty
against molecular similarity for metal–ligand selectivity*.

An applicability domain is usually a region of resemblance to the training set. This
repository treats it as a claim about error instead, and tests twenty-three candidate
criteria — fingerprint, descriptor-space, learned-embedding and model-uncertainty — against
the held-out error of a single frozen model, on ligand-disjoint folds.

**Workhorse:** `DS2_RDKIT_MLPENS5_HETERO`, a five-member heteroscedastic MLP ensemble on
RDKit descriptors with Monte-Carlo dropout at inference, selected and frozen before any
applicability-domain analysis was designed.
**Validation:** five-fold ligand-disjoint `GroupKFold`; every statistic is a fold mean ± s.d.
(`ddof=1`).

A folder map is in [`REPO_COMPOSITION.md`](REPO_COMPOSITION.md).

## The submission

| Path | Role |
|------|------|
| `manuscripts/natcompsci/main_manuscript_natcompsci.md` | **Main text. Hand-maintained; edit directly.** |
| `manuscripts/natcompsci/supplementary_information.md` | **Supplementary Information. Hand-maintained; edit directly.** S1 carries every definition, protocol and formula. |
| `manuscripts/natcompsci/figures/` | Four main-text figures and ten supplementary figures, with the two scripts that build them |
| `manuscripts/natcompsci/SUBMISSION_CHECKLIST.md` | What has been verified, and what still needs an author |
| `manuscripts/derived/tables/` | The derived CSVs every figure and validator reads |
| `manuscripts/final/` | **Frozen.** The earlier JCIM submission bundle. Never edited. Five of its tables are still read as inputs — see below. |

`manuscripts/final/` is kept because the current Supplementary Information reads
`iz_risk_score_spearman.csv`, `iz_duplicate_measurement_summary.csv`,
`iz_mahalanobis_feature_median_scores.csv` and the four pair-acquisition tables straight out
of it, and the validators assert against them. Treat it as read-only input, not as a second
manuscript.

## Rebuilding the paper

Set `PYTHONPATH=src` first. Nothing below trains or rescores a model.

```powershell
python manuscripts\natcompsci\figures\build_figures.py       # Figs 1-4 + graphical abstract
python manuscripts\natcompsci\figures\build_si_figures.py    # Figs S1-S10
```

Both read `manuscripts/derived/tables/` and, for Figs S1–S2, the deposited fold-level
prediction files. Override the table location with `BNN_TABLES` if you keep it elsewhere.

Every number quoted in the manuscript is re-derived from the frozen fold-level predictions
along a code path independent of the one that produced it, and asserted to three decimals:

```powershell
python src\uq_remake\manuscript\validate_core_analyses.py        #  56 checks
python src\uq_remake\manuscript\validate_scalefree_central.py    #  22 checks
python src\uq_remake\manuscript\validate_manuscript_values.py    # 186 checks
```

All three exit 0. If one fails it names the quantity and prints both values.

To regenerate the derived tables themselves, rather than trusting the committed copies:

```powershell
python -m uq_remake.manuscript.run_manuscript_rebuild --derived
```

`pooled_risk_ordering.py` and `decision_level_statistics.py` each take several minutes.

## Environment

Python **3.11**. Import **torch before RDKit**.

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt   # training
# or: python -m pip install -r requirements.txt  # evaluate / figures only

$env:PYTHONPATH = "$PWD\src"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"
```

Weights, fold dumps, the processed DS1/DS2 tables, the fold-local feature matrices and the
row-level downstream scores live in Git LFS archives under `artifacts/`, not unpacked in the
repository. A fresh clone must unpack before running the pipeline; the figures and
validators do not need them. Details: [`artifacts/README.md`](artifacts/README.md).

```powershell
git lfs install
git lfs pull
powershell -File scripts/unpack_artifacts.ps1            # everything the pipeline needs
powershell -File scripts/unpack_artifacts.ps1 -Rebuild   # tables + figures only, no weights
```

## Three reproduction paths

### 1) Fresh start (raw CMI tables → splits/features → train → evaluate → paper)

Needs `data/raw_m4/` (shipped in git, ~5 MB). Rebuilds DS1/DS2, ligand-disjoint folds,
RDKit/Chemprop features, the published-split and H+ Chemprop audits, the published-model
reproduction, trains the active grid, scores CV and downstream studies (including the
acquisition threshold sensitivity), then regenerates tables and figures.

```powershell
python scripts/run_pipeline.py
```

Slow, and needs a GPU. To resume after the data is rebuilt: `--from active_grid`.

### 2) Same splits, from trained weights

Unpack `artifacts/checkpoints.7z`, and optionally `cv_predictions.7z` to skip re-inference.
The fold-matched ad hoc weights are a separate archive (`checkpoints_ad_hoc.7z`).

```powershell
python scripts/run_pipeline.py --from cv_metrics
# or, if fold predictions are missing:
python scripts/reinfer_fold_predictions.py
python scripts/run_pipeline.py --from cv_metrics
```

### 3) Tables and figures from completed results

Uses stored `results/cv_metrics/` and `results/downstream_cv/`, no training. Unpack the
derived trees first with `unpack_artifacts.ps1 -Rebuild`, because the retention row scores
and the fold-local feature matrices are LFS archives rather than tracked files.

```powershell
python scripts/run_pipeline.py --dry --from manuscript
```

Checkout check:

```powershell
python scripts/validate_release.py --profile full --skip-checkpoint-load
python -m unittest discover -s tests -v
```

## What the one-command pipeline runs

| Step | Module | Live | `--dry` |
|------|--------|------|---------|
| `prepare_data` | `data/prepare_data.py` | rebuild tables, folds, features | `--verify` |
| `cmisplit` | `data/run_cmisplit.py` | published-split Chemprop audit | skip |
| `cmi_repro` | `data/run_cmi_reproduction.py` | released weights + architecture-matched retrain | skip |
| `hplus` | `data/run_hplus_decision.py` | H+ inclusion decision | skip |
| `finalize_ds1` | `data/finalize_ds1.py` | write DS1 from the H+ policy | skip |
| `active_grid` | `training/run_active_grid.py` | train | `--audit-only` |
| `cv_metrics` | `metrics/evaluate_cv.py` | score held-out folds | `--stored-only` |
| `downstream_cv` | `downstream/run_downstream_cv.py` | retention, ad hoc, pair acquisition + sensitivity | `--from-existing-scores` |
| `ad_hoc_foldmatched` | `downstream/run_ad_hoc_foldmatched.py` | retrain fold-matched DS1 and patch its rows back in | `--stage eval` |
| `manuscript` | `manuscript/run_manuscript_rebuild.py` | derived tables, figures, validators | same |
| `validate` | `release/validate_release.py` | path/figure/checkpoint gate | same |

**`ad_hoc_foldmatched` must stay where it is.** `run_downstream_cv` rewrites
`ad_hoc_selectivity_{summary,comparison,fold_metrics}.csv` from the independently folded DS1
checkpoints; the fold-matched protocol the composed-versus-direct comparison actually
reports then patches its rows back in. Run it earlier, or not at all, and that comparison
silently reverts to the superseded alignment.

## Where is this number?

| Quantity | Artifact |
|----------|----------|
| Published-split ligand overlap, 93.9% / 97.5% | `results/canonical/cmi_split_ligand_overlap.json` |
| Published-model reproduction (Fig. 1a) | `results/stage1/cmi_reproduction_results.json`; tabulated in `manuscripts/derived/tables/benchmark_reproduction.csv` |
| Composed versus direct construction (Fig. 1b) | `manuscripts/derived/tables/selectivity_construction.csv` |
| Measurement floor (SI S2.7) | `manuscripts/derived/tables/noise_floor.csv` |
| Workhorse MoP RMSE, PICP, ACE | `results/cv_metrics/fold_metric_summary.csv` |
| Pooled retention and AUC (Fig. 2a) | `manuscripts/derived/tables/pooled_retention.csv` |
| Matched-population control (SI S7.2) | `manuscripts/derived/tables/pooled_retention_matched.csv`, `pooled_contrasts.csv` |
| Score–error associations, 23 scores (Fig. 2b) | `manuscripts/final/tables/iz_risk_score_spearman.csv` (frozen input) |
| Uncertainty × support grids (Fig. 3c,d) | `manuscripts/derived/tables/scale_free_coverage_grid.csv`, `coverage_override_grid.csv` |
| Batch selection (Fig. 4) | `manuscripts/final/tables/pair_acquisition_*.csv` (frozen input) |
| Threshold sensitivity (Fig. S10) | `manuscripts/final/tables/pair_acquisition_tau2_half_*.csv` (frozen input) |
| Duplicate-measurement counts (SI S2.6) | `manuscripts/final/tables/iz_duplicate_measurement_summary.csv` (frozen input, no regenerator) |

Every deposited table carries a `.provenance.json` sidecar naming its inputs, the git commit
and the aggregation rule applied.

## Maintenance

[`MAINTENANCE.md`](MAINTENANCE.md) records the standing conditions of this checkout: the
Git LFS filter state, what the packed archives cost against a GitHub quota, and how to
reclaim the local object cache.

## Retired material

Superseded figures, tables and snapshots from earlier revisions are in
[`_deprecated/`](_deprecated/MANIFEST.md), with a manifest and a guarded removal script.
Material retired during the Nature Computational Science rework is in
`_quarantine_20260905/`, with its own manifest. Both are gitignored; read the manifests
before deleting either.

## Citation

See `CITATION.cff`.
