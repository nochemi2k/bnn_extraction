# Repository composition

Map of the checkout used to produce `manuscripts/final/main_manuscript_iz.md` and
`manuscripts/final/supplementary_information.md`.

**Workhorse:** `DS2_RDKIT_MLPENS5_HETERO`.
**DS1 retention workhorse:** `DS1_RDKIT_MLPENS5_EVID`.
**Validation:** five-fold ligand-disjoint GroupKFold; fold mean ± SD (`ddof=1`).

Retraining is supported. Canonical numbers live in `results/`. Tables and figures are
generated; **the two manuscript `.md` files are hand-maintained and are the submission
source** (see README, "Manuscript files").

## Pipeline

```
data/raw_m4
  → prepare_data (tables, folds, fold-local features)
  → cmisplit + cmi_repro + hplus + finalize_ds1
  → run_active_grid (checkpoints/active + optional cv_predictions)
  → evaluate_cv (results/cv_metrics)
  → run_downstream_cv (retention, ad hoc, pair acquisition + sensitivity, SI diagnostics)
  → run_ad_hoc_foldmatched (fold-matched Table 2 / SI S13 rows; MUST follow run_downstream_cv)
  → run_manuscript_rebuild (tables, figures, finalize ids, Table 3 check, generated draft, validators)
```

One command: `python scripts/run_pipeline.py`
From stored results only: `python scripts/run_pipeline.py --dry`
Tables and figures only: `python -m uq_remake.manuscript.run_manuscript_rebuild`

## Top level

| Path | Role |
|------|------|
| `README.md` | Reproduction paths |
| `configs/` | Grid, specs, evaluation pointers |
| `data/` | Raw CMI + fold splits in git; processed tables and feature matrices from LFS |
| `src/uq_remake/` | Package (data, models, training, metrics, downstream, manuscript) |
| `scripts/` | Thin CLIs for the three paths |
| `results/` | Summary metrics in git; row-level score dumps from LFS |
| `checkpoints/active/` | Weights (gitignored; unpack `artifacts/checkpoints.7z`) |
| `artifacts/` | Git LFS 7z packs (9 archives; see `artifacts/README.md`) |
| `manuscripts/final/` | Manuscript bundle: curated `.md`, generated tables and figures |
| `_deprecated/` | Retired artifacts + manifest + guarded removal script |
| `tests/` | Unit/integration tests |

## `scripts/` (keep these)

| Script | Calls |
|--------|--------|
| `run_pipeline.py` | `run_full_pipeline.py` |
| `prepare_data.py` | `data/prepare_data.py` |
| `train.py` | `training/run_active_grid.py` |
| `evaluate.py` | `metrics/evaluate_cv.py` |
| `reinfer_fold_predictions.py` | `training/reinfer_predictions.py` |
| `validate_release.py` | `release/validate_release.py` |
| `unpack_artifacts.ps1` | unpack LFS 7z archives; `-Rebuild` for a tables-and-figures-only checkout |
| `pack_artifacts.ps1` | repack gitignored trees into LFS 7z (temp under `_archive_tmp/`); `-Weights` / `-Bulk` |

## `src/uq_remake/` (needed for reproduction)

**Data:** `prepare_data.py`, `build_m4.py`, `build_ds2.py`, `run_cmisplit.py`, `run_cmi_reproduction.py`, `cmi_reproduction.py`, `run_hplus_decision.py`, `finalize_ds1.py`, `rdkit_features.py`, `feature_schema.py`, `descriptor_audit.py`, `ligand_parse.py`, `metal_props.py`, `species.py`.

**Training:** `run_active_grid.py`, `protocol.py`, `write_model_audit.py` (eligibility inventory used by workhorse selection), `reinfer_predictions.py`.

**Metrics:** `evaluate_cv.py`, `fold_metrics.py`, `cv_protocol.py`, `workhorse.py`, `uq_metrics.py`, `point_metrics.py`, `reeval_chemprop_evidential_v2.py`.

**Downstream:** `run_downstream_cv.py`, `run_ad_study.py`, `run_ad_hoc_selectivity.py`, `run_ad_hoc_foldmatched.py`, `run_pair_acquisition.py`, `si_diagnostics.py`, `ad_scores.py`, `analyses.py`, `ad_hoc_*.py`, `pair_acquisition.py`, `context_diagnostics.py`, `chemprop_embeddings.py`, `magnitude_adjusted_risk.py`, `ad_distance_error_corr.py`, `protocol.py`.

**Manuscript:** `run_manuscript_rebuild.py`, `build_final_assets.py`, `generate_figures.py`, `iz_risk_ordering_figure.py`, `finalize_main_figures.py`, `iz_risk_score_table.py`, `figure_specs.py`, `manuscript_paths.py`, `markdown.py`, validators. `write_main_short.py` and `write_manuscript.py` are **superseded for prose** — they write a generated draft to `manuscripts/final/_generated/`, not the submission text.

**Models:** `point_tabular.py`, `uq_tabular.py`, `variational_bnn.py`, `sngp.py`, `gpr.py`, `chemprop_dmpnn.py`, `chemprop_cli.py`, `chemprop_evidential_validate.py`.

**Release:** `validate_release.py`.

## `results/` (do not overwrite on a tables-only rebuild)

| Tree | Used for |
|------|----------|
| `cv_metrics/` | All model tables |
| `manuscript_final/workhorse_selection_accuracy_calibration.csv` | Figure 1 |
| `downstream_cv/ad_retention/` | Figure 2, Table 3, SI S7–S8 |
| `downstream_cv/pair_acquisition/` | Figure 3, Figure 4, Table 4, SI S10 |
| `downstream_cv/pair_acquisition_tau_2_0p5/` | Figure S14, Table S10.7 |
| `downstream_cv/ad_hoc_selectivity/` | Figures S15 and S17, Tables S13.4–S13.5 (patched by `run_ad_hoc_foldmatched`) |
| `downstream_cv/pair_event_support/` | Figure S12, Table S8.5 |
| `downstream_cv/pair_density/`, `sign_stability/`, `ad/`, `ad_distance_error_corr/` | SI S8–S9 |
| `canonical/` | Composition, H+, CMI overlap |
| `model_audit/workhorse_eligibility.csv` + `workhorse_selection_rule.json` | Methods 2.4, SI S6 |
| `manuscript_final/` | `numbers.json`, workhorse selection, validation reports |
| `stage1/cmi_reproduction_results.json` | Table 1 CMI columns, Tables S1.6–S1.8 |
| `cv_predictions/` | Optional OOF dumps (unpack LFS archive) |

## `manuscripts/final/`

| Path | Role |
|------|------|
| `main_manuscript_iz.md` | Submission main text (curated) |
| `supplementary_information.md` | Supporting Information (curated) |
| `figures/fig01`–`fig04` | Main-text figures, in manuscript numbering |
| `figures/figS1`–`figS18` | Supporting Information figures |
| `tables/` | Generated CSV tables |
| `_generated/` | Generated prose draft, for diffing only |

## Where is this number?

| Quantity | Artifact |
|----------|----------|
| Official overlap 93.9% / 97.5% | `results/canonical/cmi_split_ligand_overlap.json` |
| Table 1 CMI columns | `results/stage1/cmi_reproduction_results.json`, `results/stage1/stage1a_cmisplit_results.json` |
| Table 2 (ad hoc vs direct, with training time) | `manuscripts/final/tables/iz_combined_model_comparison.csv` |
| Table 3 (score–residual Spearman) | `manuscripts/final/tables/iz_risk_score_spearman.csv`; rebuilt and verified by `manuscript/iz_risk_score_table.py` |
| Duplicate-measurement counts (Methods 2.1.4) | `manuscripts/final/tables/iz_duplicate_measurement_summary.csv` (frozen input, no regenerator — see SI S12) |
| Table 4 (pair acquisition) | `results/downstream_cv/pair_acquisition/pair_acquisition_hit_and_precision.csv` + `pair_acquisition_regret.csv` |
| Workhorse MoP RMSE, PICP, ACE | `results/cv_metrics/fold_metric_summary.csv` |
| 50% retention comparison | `results/downstream_cv/ad_retention/retention_50pct.csv` |
| Retention-curve AUC | `results/downstream_cv/ad_retention/retention_auc_summary.csv` |
| Feature-median Mahalanobis sensitivity | `manuscripts/final/tables/iz_mahalanobis_feature_median_*.csv` (post hoc; see SI S7) |
| Ad hoc vs direct (SI S13) | `results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_comparison.csv` |
| Threshold sensitivity | `results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv` |
