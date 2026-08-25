# Uncertainty-Based Risk Ordering and Objective-Specific Ligand Ranking for Metal–Ligand Selectivity

Ligand-disjoint models of metal-complex stability constants (**DS1**) and metal–metal selectivity (**DS2**). The selected DS2 workhorse is the heteroscedastic MLP ensemble `DS2_RDKIT_MLPENS5_HETERO`.

This repository **does retrain**. Stored metrics and figures are also enough to rebuild every table and figure without training.

A folder map is in [`REPO_COMPOSITION.md`](REPO_COMPOSITION.md).

## Manuscript files

| File | Status |
|------|--------|
| `manuscripts/final/main_manuscript_iz.md` | **Submission source. Hand-maintained — edit directly.** |
| `manuscripts/final/supplementary_information.md` | **Submission source. Hand-maintained — edit directly.** |
| `manuscripts/final/_generated/*.md` | Generated draft, for diffing only. Not the submission. |

The prose generators (`src/uq_remake/manuscript/write_manuscript.py`,
`write_main_short.py`) are **superseded for prose**. The submitted text has diverged
from them: it carries four main figures rather than five, a numbered ACS reference
list, an Associated Content section, and Methods/SI corrections made after those
modules were last touched. To stop a rebuild silently discarding that work, the
writers now emit to `manuscripts/final/_generated/`; they overwrite the curated files
only when `UQ_REMAKE_OVERWRITE_PROSE=1` is set, which discards the curated version.

Tables, figures, the claim registry, and the validators are **not** superseded and
remain fully generated from `results/`.

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
row-level downstream scores all live in Git LFS archives under `artifacts/`, not unpacked in
the repository. A fresh clone must unpack before running anything. Details:
[`artifacts/README.md`](artifacts/README.md).

```powershell
git lfs install
git lfs pull
powershell -File scripts/unpack_artifacts.ps1            # everything the pipeline needs
powershell -File scripts/unpack_artifacts.ps1 -Rebuild   # tables + figures only, no weights
```

## Three reproduction paths

Set `PYTHONPATH=src` in every session.

### 1) Fresh start (raw CMI tables → splits/features → train → evaluate → paper)

Needs `data/raw_m4/` (shipped in git, ~5 MB). Rebuilds DS1/DS2, ligand-disjoint folds, RDKit/Chemprop features, CMI-split and H+ Chemprop audits, CMI Table 1 reproduction, trains the active grid, scores CV and downstream studies (including pair-acquisition threshold sensitivity and SI diagnostics), then regenerates tables and figures.

```powershell
python scripts/run_pipeline.py
# equivalent: python -m uq_remake.run_full_pipeline
```

This is slow (GPU). To resume after data is already rebuilt:

```powershell
python scripts/run_pipeline.py --from active_grid
```

### 2) Same splits, from trained weights

Unpack `artifacts/checkpoints.7z` (and optionally `cv_predictions.7z` to skip re-inference). Fold-matched Table 2 / S13 ad hoc weights are a separate archive (`checkpoints_ad_hoc.7z`). See [`artifacts/README.md`](artifacts/README.md). Then evaluate and rescore downstream:

```powershell
python scripts/run_pipeline.py --from cv_metrics
# or, if fold predictions are missing:
python scripts/reinfer_fold_predictions.py
python scripts/run_pipeline.py --from cv_metrics
```

Training only (writes `checkpoints/active/`):

```powershell
python scripts/train.py
```

### 3) Tables and figures from completed results

Uses stored `results/cv_metrics/` and `results/downstream_cv/` (no training). Unpack the
derived trees first — `unpack_artifacts.ps1 -Rebuild` — because the retention row scores and
the fold-local feature matrices are LFS archives, not tracked files:

```powershell
python scripts/run_pipeline.py --dry --from manuscript
# or:
python -m uq_remake.manuscript.run_manuscript_rebuild
```

`--dry` from the start verifies processed features, checks the model inventory, re-aggregates stored fold metrics, and rebuilds downstream from existing fold scores.

```powershell
python scripts/run_pipeline.py --dry
```

Checkout check:

```powershell
python scripts/validate_release.py --profile full --skip-checkpoint-load
python -m unittest discover -s tests -v
```

Use `--profile manuscript` only if `results/cv_predictions/` is absent.

## What the one-command pipeline runs

| Step | Module | Live | `--dry` |
|------|--------|------|---------|
| `prepare_data` | `data/prepare_data.py` | rebuild tables, folds, features | `--verify` |
| `cmisplit` | `data/run_cmisplit.py` | official-split Chemprop audit | skip |
| `cmi_repro` | `data/run_cmi_reproduction.py` | released CMI weights + architecture-matched retrain | skip |
| `hplus` | `data/run_hplus_decision.py` | H+ inclusion decision | skip |
| `finalize_ds1` | `data/finalize_ds1.py` | write DS1 from H+ policy | skip |
| `active_grid` | `training/run_active_grid.py` | train | `--audit-only` |
| `cv_metrics` | `metrics/evaluate_cv.py` | score held-out folds | `--stored-only` |
| `downstream_cv` | `downstream/run_downstream_cv.py` | retention, ad hoc, pair acquisition + sensitivity, SI diagnostics (`--rescore`) | `--from-existing-scores` |
| `ad_hoc_foldmatched` | `downstream/run_ad_hoc_foldmatched.py` | retrain fold-matched DS1 and patch Table 2 / SI S13 rows (`--stage all`) | `--stage eval` |
| `manuscript` | `manuscript/run_manuscript_rebuild.py` | tables, figures + finalize, generated draft, Table 3 check, validators | same |
| `validate` | `release/validate_release.py` | path/figure/checkpoint gate | same |

`cmi_repro` is skipped when `results/stage1/cmi_reproduction_results.json` is already complete (`--skip-if-complete`).

**`ad_hoc_foldmatched` must stay where it is.** `run_downstream_cv` rewrites
`ad_hoc_selectivity_{summary,comparison,fold_metrics}.csv` from the independently folded DS1
checkpoints; the fold-matched protocol that Table 2 and SI S13 actually report then patches
its rows back in. Run it earlier, or not at all, and Table 2 silently reverts to the
superseded alignment. In `--dry` it runs `--stage eval`, which re-applies the patch from
stored predictions without retraining, and skips with a message if those predictions were
never unpacked.

## Figures

**Main text (four figures).** The manuscript's numbering is the committed filename:

| File | Manuscript |
|------|------------|
| `manuscripts/final/figures/fig01_workhorse_selection` | Figure 1 — all-pair direct-UQ model context and workhorse choice |
| `manuscripts/final/figures/fig02_risk_ordering` | Figure 2 — all-pair fixed-retention risk ordering, five scores |
| `manuscripts/final/figures/fig03_predicted_magnitude` | Figure 3 — predicted magnitude, error, and uncertainty |
| `manuscripts/final/figures/fig04_pair_acquisition` | Figure 4 — objective-specific pair acquisition |

`generate_figures.py` is frozen and still writes the historical ids
(`fig02_workhorse_selection`, `fig03_risk_ordering`, `fig04_predicted_magnitude`,
`fig05_pair_acquisition`) plus a study-design schematic the manuscript no longer uses.
Run the finalize step after it to produce the committed names:

```powershell
python -m uq_remake.manuscript.generate_figures
python -m uq_remake.manuscript.finalize_main_figures
```

`run_manuscript_rebuild` calls `finalize_main_figures` for you, then
`redraw_figures`; run either by hand only if you invoked `generate_figures` directly.

**`redraw_figures` must run last.** Six figures — main-text 1, 3 and 4, and Supplementary
S10, S11 and S12 — are rebuilt from the same stored summary CSVs because the frozen
generator's layout obscured the result each one exists to show: Figure 1 had no error bars
on a panel whose argument is that the models are not separable; Figures 3 and S11 plotted
error and uncertainty as two curves on separate axes when they are the same quantity in the
same units; Figure 4 coloured a monotone penalty as three unrelated hues; S10 gave three
different series near-identical colours; S12 was unreadable on linear axes and spent a
colourbar on a variable already carried by its own y-axis. No displayed number changes.

```powershell
python -m uq_remake.manuscript.redraw_figures            # all six
python -m uq_remake.manuscript.redraw_figures --only 1 4 # selected
```
 It also regenerates `fig02_risk_ordering` via
`iz_risk_ordering_figure.py`, the five-score version the manuscript displays (the
`generate_figures` panel plots three scores), and it maps the supplementary ids: the SI
dropped the high-selectivity pair-membership panel, so `figS7`–`figS19` become
`figS6`–`figS18`.

**Supporting Information:** `figS1`–`figS18`, no gaps. Figures S15–S18 belong to the
restricted rare-earth-element diagnostic in SI Section S13.4.

## Canonical outputs

**Headline tables:** `manuscripts/final/tables/iz_combined_model_comparison.csv` (Table 2), `iz_risk_score_spearman.csv` (Table 3), `pair_acquisition_hit_and_precision.csv` and `pair_acquisition_regret.csv` (Table 4), `ad_retention_50pct.csv`.

Table 3 had no generator until now. `manuscript/iz_risk_score_table.py` rebuilds all 23 rows
from the frozen workhorse fold scores, the registered distance–residual summary and the
post-hoc feature-median cache, and **verifies** rather than overwrites:

```powershell
python -m uq_remake.manuscript.iz_risk_score_table          # verify (runs inside the rebuild)
python -m uq_remake.manuscript.iz_risk_score_table --write  # regenerate
```

**Science trees:** `results/cv_metrics/`, `results/downstream_cv/`, `results/canonical/`, `results/manuscript_final/`.

## Retired artifacts

Superseded figures, tables, and snapshots from earlier revisions were moved to
[`_deprecated/`](_deprecated/MANIFEST.md), which explains each entry and ships a
guarded removal script for the originals. Read the manifest before running it.

## Citation

See `CITATION.cff`.
