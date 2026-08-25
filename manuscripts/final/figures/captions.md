# Figure captions

Generated from the manuscript text and the canonical fold-level artifacts.
Main-text figure ids match the numbering in `main_manuscript_iz.md` (Figures 1-4).
Retention curves are empirical ranking tools, not coverage guarantees.

## fig01_workhorse_selection

Figure 1. All-pair direct-UQ model context and representative-model choice. The heteroscedastic MLP ensemble is fixed as the workhorse for downstream analyses because it provides an operationally suitable direct-UQ model under the study’s interval-usability convention; it is not presented as the universally optimal architecture. The plotted coordinates are pair-equal MoP diagnostics. Eligibility is determined separately from unrounded five-fold row-level PICP@90 and ACE values, so neither shaded region is the operational eligibility gate. Star = selected heteroscedastic ensemble. Diamond = Chemprop evidential partial comparator. Crosses = models excluded by the row-level screen.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2 UQ models`
- aggregation: `fold_mean_plus_sd`
- source: `results/manuscript_final/workhorse_selection_accuracy_calibration.csv`
- status: `canonical`

## fig02_risk_ordering

Figure 2. DS2 selected-workhorse all-pair fixed-retention risk ordering. Each score is used to retain a nominal fraction of held-out DS2 predictions, after which MoP RMSE is evaluated on the resulting all-pair subset. In addition to $\sigma_{\mathrm{total}}$, RDKit feature-centroid Mahalanobis distance, and Tanimoto nearest-neighbor distance, the figure shows Tanimoto median distance (median Tanimoto distance to all outer-training ligands) and RDKit feature-median Mahalanobis distance (median Ledoit–Wolf Mahalanobis distance to all outer-training rows in the same StandardScaled RDKit+metal+medium space). The feature-median Mahalanobis curve is dashed because it nearly coincides with the feature-centroid curve; both are plotted. Panel A: retained-subset all-pair MoP RMSE. Panel B counts species-level pairs with at least three retained rows. It contextualizes MoP support and is not a performance metric. The black dashed curve is the fold-local random-retention baseline. Different scores retain different pair populations at the same nominal retention fraction. Markers show 10%, 25%, 50%, and 100% retention. The result tests empirical error ordering for the stated all-pair dataset and validation regime.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_local_then_mean_sd`
- source: `results/downstream_cv/ad_retention/fold_retention.csv`; `results/downstream_cv/ad_retention/retention_summary.csv`; `manuscripts/final/tables/iz_mahalanobis_feature_median_retention_summary.csv` (post-hoc sensitivity score, SI S7)
- generator: `src/uq_remake/manuscript/iz_risk_ordering_figure.py` (five scores; the `generate_figures.py` panel plots three)
- status: `canonical`

## fig03_predicted_magnitude

Figure 3. Predicted magnitude, error, and uncertainty. Panel A: held-out row-level RMSE by predicted-magnitude bin. Panel B: mean $\sigma_{\mathrm{total}}$ over the same bins. Panel C: fold-mean Spearman rank associations among $\sigma_{\mathrm{total}}$, $|\widehat y|$, $|y|$, and the absolute residual $|e|=|y-\widehat y|$. Bars and points are fold means; error bars are fold SD. Binned RMSE is a within-bin row-level quantity and is not pair-level MoP RMSE. The magnitude-adjusted partial rank correlation $\rho(\sigma_{\mathrm{total}},|e| \mid |\widehat y|)=0.428 \pm 0.026$ is reported in the text and in Supplementary Table S10.7 rather than plotted here.

- dataset: `DS2`
- scope: `all_heldout_rows`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_mean_plus_sd`
- source: `results/downstream_cv/pair_acquisition/pair_acquisition_calibration_bin_summary.csv`
- status: `canonical`

## fig04_pair_acquisition

Figure 4. Objective-specific pair acquisition. Ranking rules are evaluated separately for high-selectivity and near-zero objectives using true-hit rate, event precision, and retrospective regret at $k=1$, $5$, and $10$. High-selectivity events are target-direction $\Delta\log K>1$; near-zero events are $|\Delta\log K|<1$. The left column (A, C, E) is high selectivity and the right column (B, D, F) is near-zero selectivity. Rows are true hit@$k$, event precision@$k$, and retrospective regret@$k$. Panels E and F report retrospective regret@$k$ (lower is better) on the same pair-task shortlists. Table 4 reports the main-text subset, including retrospective regret@$k$; Table S10.4 reports the complete retrospective regret inventory. These panels summarize the retrospective held-out ranking evaluation defined in Methods 2.6. Ranking-function and retrospective regret@$k$ definitions are in Methods 2.6.

- dataset: `DS2`
- scope: `heldout_fold`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `pair_equal_then_fold_mean_plus_sd`
- source: `results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv`
- status: `canonical`

## figS1_ds1_complete

Supplementary Figure S1. Complete DS1 metal-complex RMSE and $R^2$ for active models.

- dataset: `DS1`
- scope: `metal_complex`
- models: `all active DS1 models`
- aggregation: `fold_mean_plus_sd`
- source: `manuscripts/final/tables/supplementary_complete_model_results.csv`
- status: `canonical`

## figS2_ds1_point

Supplementary Figure S2. DS1 point-model metal-complex RMSE and $R^2$.

- dataset: `DS1`
- scope: `metal_complex`
- models: `DS1 point models`
- aggregation: `fold_mean_plus_sd`
- source: `manuscripts/final/tables/main_ds1_point_results.csv`
- status: `canonical`

## figS3_ds1_uq

Supplementary Figure S3. DS1 uncertainty-aware models on held-out metal-complex rows. RMSE and $R^2$ summarize absolute $\log K$ prediction; PICP@90 and ACE summarize row-level calibration of nominal 90% intervals for absolute $\log K$. These are not DS2 pair-level selectivity metrics.

- dataset: `DS1`
- scope: `metal_complex`
- models: `DS1 UQ models`
- aggregation: `fold_mean_plus_sd`
- source: `manuscripts/final/tables/main_ds1_uq_results.csv`
- status: `canonical`

## figS4_ds2_all_pairs

Supplementary Figure S4. DS2 all-pair pair-level comparison.

- dataset: `DS2`
- scope: `all_pairs`
- models: `all active DS2 models`
- aggregation: `fold_mean_plus_sd`
- source: `manuscripts/final/tables/ds2_all_pairs.csv`
- status: `canonical`

## figS5_ds2_highsel_row

Supplementary Figure S5. DS2 high-selectivity row-scope comparison.

- dataset: `DS2`
- scope: `high_selectivity_row`
- models: `all active DS2 models`
- aggregation: `fold_mean_plus_sd`
- source: `manuscripts/final/tables/ds2_highsel_row.csv`
- status: `canonical`

## figS6_workhorse_fold_stability

Supplementary Figure S6. All-pair fold stability of the selected workhorse.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_level`
- source: `manuscripts/final/tables/supplementary_fold_breakdowns.csv`
- status: `canonical`

## figS7_chemprop_embedding_retention

Supplementary Figure S7. All-pair representation-space retention sensitivity analysis.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_GRAPH_DMPNN_EVID`
- aggregation: `fold_local_then_mean_sd`
- source: `results/downstream_cv/ad_retention/fold_retention.csv`
- status: `canonical`

## figS8_uncertainty_decomposition

Supplementary Figure S8. Workhorse uncertainty components; standard-deviation components and variance components.

- dataset: `DS2`
- scope: `heldout_fold`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_mean_of_row_scores`
- source: `results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold0.csv`
- status: `canonical`

## figS9_ad_orientation

Supplementary Figure S9. Risk-score orientation audit; higher values indicate greater predicted risk.

- dataset: `DS2`
- scope: `workhorse_fold_local`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_level`
- source: `results/downstream_cv/ad/ad_orientation_audit.csv`
- status: `canonical`

## figS10_retention_all_pairs

Supplementary Figure S10. All-pair MoP retention error curves; primary scores emphasized.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_local_then_mean_sd`
- source: `results/downstream_cv/ad_retention/fold_retention.csv`
- status: `canonical`

## figS11_pair_density

Supplementary Figure S11. All-pair pair-support density and workhorse diagnostics.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_mean_plus_sd`
- source: `results/downstream_cv/pair_density/pair_density_summary.csv`
- status: `canonical`

## figS12_pair_event_support

Supplementary Figure S12. Held-out pair support versus high-selectivity event support.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `out_of_fold_pooled`
- source: `results/downstream_cv/pair_event_support/pair_event_support_pooled.csv`
- status: `canonical`

## figS13_sign_stability_detail

Supplementary Figure S13. All-pair interval sign stability.

- dataset: `DS2`
- scope: `all_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_mean_plus_sd`
- source: `manuscripts/final/tables/sign_stability_summary.csv`
- status: `canonical`

## figS14_pair_acquisition_threshold_sensitivity

Supplementary Figure S14. Supporting event-threshold sensitivity for pair acquisition.

- dataset: `DS2`
- scope: `heldout_fold`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `pair_equal_then_fold_mean_plus_sd`
- source: `results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv`
- status: `canonical`

## figS15_ad_hoc_interval_calibration

Supplementary Figure S15. Direct DS2 intervals versus propagated ad hoc intervals on the restricted REE–REE diagnostic.

- dataset: `DS2`
- scope: `ree_pairs`
- models: `direct_ds2_and_ad_hoc_ds1`
- aggregation: `fold_mean_plus_sd`
- source: `results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_comparison.csv`
- status: `canonical`

## figS16_retention_ree

Supplementary Figure S16. Restricted REE–REE diagnostic MoP retention curves.

- dataset: `DS2`
- scope: `ree_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_local_then_mean_sd`
- source: `results/downstream_cv/ad_retention/fold_retention.csv`
- status: `canonical`

## figS17_direct_vs_ad_hoc_selectivity

Supplementary Figure S17. Direct versus propagated intervals on the restricted REE–REE diagnostic.

- dataset: `DS2`
- scope: `ree_pairs`
- models: `direct_ds2_and_ad_hoc_ds1`
- aggregation: `fold_mean_plus_sd`
- source: `results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv`
- status: `canonical`

## figS18_ree_fold_stability

Supplementary Figure S18. Fold stability on the restricted REE–REE diagnostic.

- dataset: `DS2`
- scope: `ree_pairs`
- models: `DS2_RDKIT_MLPENS5_HETERO`
- aggregation: `fold_level`
- source: `manuscripts/final/tables/supplementary_fold_breakdowns.csv`
- status: `canonical`
