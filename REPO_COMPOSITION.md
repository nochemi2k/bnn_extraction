# Repository composition

Map of the checkout that produces `manuscripts/natcompsci/main_manuscript_natcompsci.md`
and `manuscripts/natcompsci/supplementary_information.md`.

**Workhorse:** `DS2_RDKIT_MLPENS5_HETERO`.
**DS1 retention workhorse:** `DS1_RDKIT_MLPENS5_EVID`.
**Validation:** five-fold ligand-disjoint `GroupKFold`; fold mean ± s.d. (`ddof=1`).

Retraining is supported but not required. Canonical numbers live in `results/`; the derived
tables the manuscript reads live in `manuscripts/derived/tables/`. **The two manuscript
`.md` files are hand-maintained and are the submission source** — no generator writes them.

## Pipeline

```
data/raw_m4
  → prepare_data (tables, folds, fold-local features)
  → cmisplit + cmi_repro + hplus + finalize_ds1
  → run_active_grid (checkpoints/active + optional cv_predictions)
  → evaluate_cv (results/cv_metrics)
  → run_downstream_cv (retention, ad hoc, pair acquisition + threshold sensitivity)
  → run_ad_hoc_foldmatched (fold-matched composed-model rows; MUST follow run_downstream_cv)
  → run_manuscript_rebuild (derived tables, figures, validators)
```

One command: `python scripts/run_pipeline.py`
From stored results only: `python scripts/run_pipeline.py --dry`
Derived tables only: `python -m uq_remake.manuscript.run_manuscript_rebuild --derived`

## Top level

| Path | Role |
|------|------|
| `README.md` | Reproduction paths and a number-to-artifact index |
| `configs/` | Grid, experiment specs, evaluation pointers |
| `data/` | Raw CMI tables and fold splits in git; processed tables and feature matrices from LFS |
| `src/uq_remake/` | Package (data, models, training, metrics, downstream, manuscript) |
| `scripts/` | Thin CLIs for the three reproduction paths |
| `results/` | Summary metrics in git; row-level score dumps from LFS |
| `checkpoints/active/` | Weights (gitignored; unpack `artifacts/checkpoints.7z`) |
| `artifacts/` | Git LFS 7z packs, nine archives; see `artifacts/README.md` |
| `manuscripts/natcompsci/` | **The submission**: text, SI, figures, checklist |
| `manuscripts/derived/` | Derived tables and the prototype sanity-check JSON the analyses verify against |
| `manuscripts/final/` | **Frozen** earlier JCIM bundle; read-only, still the source of five input tables |
| `_deprecated/`, `_quarantine_20260905/` | Retired material with manifests (both gitignored) |
| `tests/` | Unit and integration tests |

## `manuscripts/`

| Path | Role |
|------|------|
| `natcompsci/main_manuscript_natcompsci.md` | Main text, curated |
| `natcompsci/supplementary_information.md` | Supplementary Information, curated; S1 is Methods |
| `natcompsci/figures/build_figures.py` | Figs 1–4 and the graphical abstract |
| `natcompsci/figures/build_si_figures.py` | Figs S1–S10 |
| `natcompsci/figures/fig*.png` / `.pdf` | The rendered display items |
| `natcompsci/SUBMISSION_CHECKLIST.md` | Verification status and outstanding author items |
| `derived/tables/` | 53 derived CSVs plus `.provenance.json` sidecars |
| `derived/analysis/chemical_sanity_checks.json` | Prototype values the analyses are checked against |
| `final/` | Frozen JCIM bundle: curated `.md`, its tables and figures |

Five files in `manuscripts/final/tables/` are live inputs rather than history:
`iz_risk_score_spearman.csv`, `iz_duplicate_measurement_summary.csv`,
`iz_mahalanobis_feature_median_scores.csv`, and the `pair_acquisition_*` tables including
the `tau2_half` threshold-sensitivity set. They are read, never rewritten.

## `scripts/`

| Script | Calls |
|--------|--------|
| `run_pipeline.py` | `run_full_pipeline.py` |
| `prepare_data.py` | `data/prepare_data.py` |
| `train.py` | `training/run_active_grid.py` |
| `evaluate.py` | `metrics/evaluate_cv.py` |
| `reinfer_fold_predictions.py` | `training/reinfer_predictions.py` |
| `validate_release.py` | `release/validate_release.py` |
| `unpack_artifacts.ps1` | Unpack the LFS 7z archives; `-Rebuild` for a tables-and-figures-only checkout |
| `pack_artifacts.ps1` | Repack gitignored trees into LFS 7z; `-Weights` / `-Bulk` |

## `src/uq_remake/manuscript/`

Three groups, and the distinction matters when deciding what may be changed.

**Emit the derived tables the current submission reads.** `benchmark_context.py`
(published-model reproduction, composed versus direct), `noise_floor.py` (the measurement
floor), `pooled_risk_ordering.py` (pooled retention and the matched-population control),
`decision_level_statistics.py` (within-cell retention, both units),
`embedding_retention.py` (second model and its own learned embedding),
`coverage_override_grid.py` (uncertainty × support grids),
`robustness_threshold_and_model.py`, `sigma_chemical_axes.py`, `scale_free_error.py`,
`uncertainty_by_regime.py`, `logk_censoring.py`, `within_pair_ordering.py`.
`external_set_emulation.py` and `single_pair_campaign.py` emit the two analyses reported as
conducted-and-not-reported.

**Validate.** `validate_manuscript_values.py` (186 checks), `validate_core_analyses.py`
(56), `validate_scalefree_central.py` (22), plus `validate_pair_acquisition.py`,
`validate_ad_retention.py`, `validate_ad_hoc_selectivity.py`, `validate_manuscript.py`,
`validate_no_spearman.py`.

**Build the frozen JCIM bundle.** `generate_figures.py`, `finalize_main_figures.py`,
`iz_risk_ordering_figure.py`, `iz_risk_score_table.py`, `redraw_figures.py`,
`build_final_assets.py`, `figure_specs.py`, `write_manuscript.py`, `write_main_short.py`.
These are the provenance for the frozen tables the current SI still reads, which is why they
are kept; they do not touch `manuscripts/natcompsci/`.

Shared: `manuscript_common.py` (constants, fold loaders, provenance sidecars),
`manuscript_paths.py`, `markdown.py`, `run_manuscript_rebuild.py`.

## `src/uq_remake/`, the rest

**Data:** `prepare_data.py`, `build_m4.py`, `build_ds2.py`, `run_cmisplit.py`,
`run_cmi_reproduction.py`, `cmi_reproduction.py`, `run_hplus_decision.py`,
`finalize_ds1.py`, `rdkit_features.py`, `feature_schema.py`, `descriptor_audit.py`,
`ligand_parse.py`, `metal_props.py`, `species.py`.

**Training:** `run_active_grid.py`, `protocol.py`, `write_model_audit.py`,
`reinfer_predictions.py`.

**Metrics:** `evaluate_cv.py`, `fold_metrics.py`, `cv_protocol.py`, `workhorse.py`,
`uq_metrics.py`, `point_metrics.py`, `reeval_chemprop_evidential_v2.py`.

**Downstream:** `run_downstream_cv.py`, `run_ad_study.py`, `run_ad_hoc_selectivity.py`,
`run_ad_hoc_foldmatched.py`, `run_pair_acquisition.py`, `si_diagnostics.py`, `ad_scores.py`,
`analyses.py`, `ad_hoc_*.py`, `pair_acquisition.py`, `context_diagnostics.py`,
`chemprop_embeddings.py`, `magnitude_adjusted_risk.py`, `ad_distance_error_corr.py`,
`protocol.py`.

**Models:** `point_tabular.py`, `uq_tabular.py`, `variational_bnn.py`, `sngp.py`, `gpr.py`,
`chemprop_dmpnn.py`, `chemprop_cli.py`, `chemprop_evidential_validate.py`.

**Release:** `validate_release.py`.

## `results/` — do not overwrite on a tables-only rebuild

| Tree | Feeds |
|------|-------|
| `cv_metrics/` | Model comparison; the row-weighted against mean-of-pairs contrast |
| `downstream_cv/ad_retention/` | Figure 2, SI S6–S7 |
| `downstream_cv/pair_acquisition/` | Figure 4, SI S9 |
| `downstream_cv/pair_acquisition_tau_2_0p5/` | Figure S10, SI S9.4 |
| `downstream_cv/ad_hoc_selectivity/` | Figure 1b, SI S4.5 (patched by `run_ad_hoc_foldmatched`) |
| `downstream_cv/pair_density/`, `sign_stability/`, `ad/`, `ad_distance_error_corr/` | SI S7–S8 |
| `canonical/` | Composition, H+ decision, published-split ligand overlap |
| `stage1/cmi_reproduction_results.json` | Figure 1a, SI S3.3 |
| `model_audit/` | Model-selection screen, SI S4.2 |
| `cv_predictions/` | Optional out-of-fold dumps (unpack the LFS archive) |
| `manuscript_final/` | Frozen-bundle validation reports |
