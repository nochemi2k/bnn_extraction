# Figure captions

Main-text figures are Figure 0 (graphical abstract) and Figures 1-5, with filenames `fig00_`-`fig05_`.
Supplementary figures are S1-S18, filenames `figS1_`-`figS18_`. Regenerated from the manuscript sources;
do not edit by hand.

## Figure 0. Graphical abstract.

Metal–ligand selectivity is modeled directly on the pairwise target $\Delta\log K$ rather than reconstructed from two absolute-binding predictions, which attaches the uncertainty estimate to the quantity the experiment consumes and more than halves its calibration error. A single model carrying direct uncertainty quantification (UQ) is then fixed and asked two questions. First, which predictions can be trusted: retaining the half of held-out rows the model is surest of cuts pair-equal error by 65%, whereas the same retention driven by molecular distance recovers substantially less — and is beaten, over most of the retention range, by a bare count of how many training ligands the metal pair carries, which uses no molecular structure at all. Second, how uncertainty should enter a ranking rule: subtracting it as a lower confidence bound helps a near-zero objective and steadily damages a high-selectivity one, because the candidates such a campaign wants are the ones the model is least sure about, whereas dividing by it as a standardized threshold margin leads the high-selectivity objective outright and is level with the best rule on the near-zero one. The retention curves are the four scores of Figure 2 and the ranking bars a subset of Figure 5; every plotted value is a fold mean reported there.

## Figure 1. All-pair direct-UQ model context and workhorse choice.

Both panels are pair-equal (MoP) on both axes: MoP RMSE horizontally, against MoP ACE (A) and MoP PICP@90 (B), with fold-SD error bars ($n=5$, $\mathrm{ddof}=1$) on both. The horizontal bars overlap heavily, which is the point — neighboring models differ in accuracy by well under one fold standard deviation, so this is not a leaderboard. They separate vertically instead, and the heteroscedastic MLP ensemble is fixed as the workhorse as the Pareto-front member closest to the ideal point (Methods 2.4). The dashed line in panel B marks nominal coverage. Star = selected workhorse; open circles = the other direct-UQ models; diamond = Chemprop evidential, retained as a comparator with partial native uncertainty.

## Figure 2. DS2 workhorse all-pair fixed-retention risk ordering.

Within each held-out fold, rows are ranked by a risk score, the lowest-risk fraction is retained, and MoP RMSE is recomputed on pairs keeping at least three rows. Four scores are compared against the fold-local random-retention baseline (black dashed), chosen to separate *what each score can see* rather than to inventory aggregation variants: $\sigma_{\mathrm{total}}$ (red), the model's own predictive uncertainty; **pair-density risk** (amber), a count of how many training ligands the row's metal pair carries, using no molecular structure and no model output; **RDKit feature-median Mahalanobis distance** (blue), a distance in the joint ligand–metal–medium descriptor space; and **Tanimoto median distance** (green), a distance in ligand-fingerprint space. Bands are fold SD on every score ($n=5$, $\mathrm{ddof}=1$), so each curve can be read against its own fold spread. The two distances converge above 30% retention and separate from random by only a fraction of what the other two scores achieve, while pair-density risk — which knows nothing about chemistry — sits below both of them at every retention fraction from 30% upward and is the only one of the four to cross above the random baseline in the extreme tail at 10%. Panel B is what keeps that reading honest: pair-density risk reaches its curve in A while retaining only $109.4 \pm 4.5$ supported pairs at 50% retention, a third of the next-narrowest score, and on the pairs every score keeps it is the worst of the six rather than the second best (Table S7.6). A retention curve cannot by itself distinguish a score that orders error from one that selects an easy population. The centroid aggregation of the descriptor distance and the nearest-neighbor aggregation of both distances behave like the median curves shown; all are tabulated in Supplementary Tables S7.1 and S7.4, and the RDKit Mahalanobis nearest-neighbor variant in Tables S7.5 and S8.4. Panel B counts the species-level pairs still supported at each fraction; it makes the changing pair population explicit and is not a performance metric.

## Figure 3. Score--residual association across the full score inventory.

Fold-mean Spearman $\rho$ between each of the twenty-three candidate risk scores and the absolute held-out residual $|y-\widehat y|$, sorted, coloured by score family. Points are fold means, whiskers fold SD ($n=5$, $\mathrm{ddof}=1$), on 17,705 held-out DS2 rows per fold. Every score is oriented so that larger values mean greater predicted risk, so a positive $\rho$ is the sign a usable risk score must have and the vertical rule at zero is the line that matters. Learned-representation scores use the Chemprop evidential checkpoint; all others use the workhorse. Exact values are Supplementary Table S8.6.

## Figure 4. Predicted magnitude, error, and uncertainty.

Panel A plots held-out row-level RMSE and mean $\sigma_{\mathrm{total}}$ on a common axis across predicted-magnitude bins; both are in $\Delta\log K$, and the shaded ribbon is the gap between them. Both rise by more than fivefold across the bins — RMSE from $0.906$ to $5.112$ and mean $\sigma_{\mathrm{total}}$ from $0.676$ to $5.224$ — and the ribbon changes sign near $|\widehat y|=4$, where uncertainty overtakes error. This coupling is what makes the magnitude-adjusted analysis necessary: the region a high-selectivity campaign most wants to explore is the region where the model is least accurate and least confident. Panel B asks whether the uncertainty signal survives that coupling. The two solid bars are raw fold-mean Spearman rank correlations with the absolute residual $|e|=|y-\widehat y|$; the hatched bar is the fold-local partial rank correlation controlling for $|\widehat y|$, which remains positive at $0.428 \pm 0.026$ (Table S10.5). Points and bars are fold means, error bars fold SD ($n=5$). Binned RMSE is a within-bin row-level quantity, not pair-level MoP RMSE.

## Figure 5. Objective-specific pair acquisition.

Six ranking rules at $k=1$, $5$ and $10$, on two experimental objectives: high selectivity (left column, target-direction $\Delta\log K>1$) and near-zero selectivity (right column, $|\Delta\log K|<1$). Rows are the three metrics, which answer three different questions about the same shortlist — will it contain anything at all (true hit@$k$, A–B), what fraction of it qualifies (event precision@$k$, C–D), and how close its best member comes to the best candidate available (retrospective regret@$k$, E–F, lower is better). LCB-1, LCB-2 and LCB-3 are one rule at increasing uncertainty penalty, shaded light to dark, so its effect reads as a gradient within each group. That gradient runs downward in the left column and is flat in the right: subtracting uncertainty costs a high-selectivity campaign heavily and buys a near-zero campaign a gain that does not depend on the penalty strength. The threshold-margin rule (amber) divides by $\sigma_{\mathrm{total}}$ rather than subtracting it, and is the only uncertainty-aware rule that never falls materially below the value-only baseline on an event metric in either objective — it leads panels A and C outright and matches the best rule in B and D, its single shortfall being a $0.001$ deficit on near-zero true hit@10 (Table S10.2). Panel A carries the one place where the two event metrics disagree: at $k\geq5$ the strongest penalty recovers an event in fewer pair-tasks than random ranking, which event precision does not show. Regret is the one axis where the value-only rule wins outright, and only on the high-selectivity objective (E). Bars are fold means, error bars fold SD ($n=5$), after a pair-equal mean within each fold. Definitions are in Methods 2.6; complete inventories in Tables S10.2 and S10.3.

## Supplementary Figure S1. Complete DS1 metal-complex RMSE and $R^2$ for active models.

Primary full-data DS2 all-pair scope. All-pair MoP RMSE is the primary accuracy metric. All-pair PICP@90 and ACE here are pair-equal MoP summaries on the same supported pairs. Row-level PICP@90 and ACE used for the operational screen are Table S4.2. All-pair MoP $R^2$ is auxiliary. Scope = `all_pairs`. Source = `manuscripts/final/tables/ds2_all_pairs.csv`.

## Supplementary Figure S2. DS1 point-model metal-complex RMSE and $R^2$.

Primary full-data DS2 all-pair scope. All-pair MoP RMSE is the primary accuracy metric. All-pair PICP@90 and ACE here are pair-equal MoP summaries on the same supported pairs. Row-level PICP@90 and ACE used for the operational screen are Table S4.2. All-pair MoP $R^2$ is auxiliary. Scope = `all_pairs`. Source = `manuscripts/final/tables/ds2_all_pairs.csv`.

## Supplementary Figure S3. DS1 uncertainty-aware models on held-out metal-complex rows. RMSE and $R^2$ summarize absolute $\log K$ prediction; PICP@90 and ACE summarize row-level calibration of nominal 90% intervals for absolute $\log K$. These are not DS2 pair-level selectivity metrics.

Primary full-data DS2 all-pair scope. All-pair MoP RMSE is the primary accuracy metric. All-pair PICP@90 and ACE here are pair-equal MoP summaries on the same supported pairs. Row-level PICP@90 and ACE used for the operational screen are Table S4.2. All-pair MoP $R^2$ is auxiliary. Scope = `all_pairs`. Source = `manuscripts/final/tables/ds2_all_pairs.csv`.

## Supplementary Figure S4. DS2 all-pair pair-level comparison.

MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. Negative values indicate poor and unstable within-pair explanatory performance under this aggregation.

## Supplementary Figure S5. DS2 high-selectivity scope comparison.

Row-scope values. MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. Negative values indicate poor and unstable within-pair explanatory performance under this aggregation.

## Supplementary Figure S6. All-pair fold stability of the selected workhorse.

Fold-local all-pair MoP RMSE and auxiliary MoP $R^2$ are shown for the five held-out folds of the selected heteroscedastic MLP ensemble. MoP $R^2$ is auxiliary and does not determine workhorse selection.

## Supplementary Figure S7. All-pair representation-space retention sensitivity analysis.

Eight of the registered scores are drawn: the four uncertainty terms, pair-density risk, the RDKit feature-centroid Mahalanobis distance, and the two Tanimoto aggregations. The Chemprop embedding distances and the post-hoc RDKit Mahalanobis median and nearest-neighbor variants are omitted for legibility; their retention behaviour is in Tables S7.1, S7.3 and S7.4, and the per-fraction values for every score are in `results/downstream_cv/ad_retention/retention_summary.csv`. Thick lines are $\sigma_{\mathrm{total}}$, Tanimoto nearest-neighbor distance, and random retention. Chemprop embedding distances are muted sensitivity analyses and are not pooled with the selected workhorse uncertainty score. The two RDKit descriptor-space Mahalanobis variants differ by $0.006$ $\Delta\log K$ at 50% retention, so main-text Figure 2 plots only the median variant and Table S7.4 carries both.

## Supplementary Figure S8. Workhorse uncertainty components; standard-deviation components and variance components.

Selected workhorse = Heteroscedastic MLP ensemble (`DS2_RDKIT_MLPENS5_HETERO`). Bars are fold means. The variance identity is applied at the row level before averaging. Panel A standard deviations do not add linearly. Panel B variances add to total predictive variance as defined in Section S2.4.

## Supplementary Figure S9. Risk-score orientation audit; higher values indicate greater predicted risk.

All score orientations were verified before retention analysis; higher scores indicate greater predicted risk.

## Supplementary Figure S10. All-pair MoP retention error curves; primary scores emphasized.

Every registered score on one axis. Colour encodes score family: red for the predictive-uncertainty components, amber for pair-density support, blue for descriptor-space distance, green for fingerprint distance. Emphasized lines carry a fold-SD band. The random-retention baseline is the black dashed curve; the fingerprint nearest-neighbor curve lies **above** it from 10% to about 70% retention before converging on it, which is the negative result reported in main-text Section 3.4. All scores are oriented so that larger values mean greater predicted risk, and retention keeps the lowest-scoring rows. Empirical error-ordering scores, not out-of-distribution (OOD) detectors.

## Supplementary Figure S11. All-pair pair-support density and workhorse diagnostics.

All-pair DS2 workhorse, binned by the number of training ligands available for the held-out row's species-level metal pair. Panel A plots held-out row-level RMSE and mean $\sigma_{\mathrm{total}}$ on a common $\Delta\log K$ axis: both fall by roughly a factor of three across the bins, so the model's own uncertainty tracks pair support although support is never supplied to it as a feature. Panel B plots row-level PICP@90 against nominal coverage; the shaded area is the shortfall. Coverage degrades as support thins — monotonically in the point estimates, though the 11–25 and 26–50 bins differ by only $0.001$ — reaching $0.803 \pm 0.036$ where the metal pair has no training ligands at all. Bin 0 is a pair-context extrapolation regime and is not complete chemical novelty. Fold means with fold SD ($n=5$).

## Supplementary Figure S12. Held-out pair support versus high-selectivity event support.

Out-of-fold held-out labels only; both axes are logarithmic because the pair inventory spans three orders of magnitude in support. Colour is the observed high-selectivity fraction of each pair, so the tight upper diagonal in Panel A is the population of pairs in which nearly every held-out row is an event, and the separate lower cloud is the near-degenerate pairs. Panel B shows per-pair error falling and tightening toward $1$–$2$ $\Delta\log K$ as support grows; the low-error purple cluster at high support is the near-zero pair family. The plotted pooled per-pair RMSE is not MoP RMSE. Descriptive and label-informed; it describes pair-specific data support and is not a chemical mechanism analysis or a selection rule.

## Supplementary Figure S13. All-pair interval sign stability.

Categories are based on whether the nominal 90% interval is entirely negative, entirely positive, or overlaps zero. The overlap-zero category dominates the all-pair scope.

## Supplementary Figure S14. Supporting event-threshold sensitivity for pair acquisition.

High-selectivity events are target-direction $\Delta\log K>2$; near-zero events are $|\Delta\log K|<1/2$. The plotted ranking families match Figure 5 and the columns carry the same two objectives (left high selectivity, right near-zero). This supporting figure uses the same three-row layout as the redrawn main-text Figure 5: true hit@$k$, event precision@$k$, and retrospective regret@$k$. This supporting sensitivity does not replace Figure 5 or Table 3. Table S10.6 reports the compact $k=1,5$ subset.

## Supplementary Figure S15. Direct DS2 intervals versus propagated ad hoc intervals on the restricted REE–REE diagnostic.

Direct intervals are native intervals for $\Delta\log K$; ad hoc intervals are propagated diagnostics from two absolute-binding predictions under an independent-error assumption. This restricted diagnostic is descriptive and is not used for workhorse selection or the primary all-pair conclusions. Ad hoc series use the same DS2 fold-matched DS1 protocol as Table 2.

## Supplementary Figure S16. Restricted REE–REE diagnostic MoP retention curves.

Restricted diagnostic; not the primary all-pair retention result and

## Supplementary Figure S17. Direct versus propagated intervals on the restricted REE–REE diagnostic.

Direct DS2 intervals and propagated ad hoc intervals are compared here as a restricted diagnostic. The analysis is not the primary performance evaluation, does not replace all-pair MoP metrics, and is not used to select the workhorse or establish the all-pair risk-ordering result. Ad hoc series use the same DS2 fold-matched DS1 protocol as Table 2.

## Supplementary Figure S18. Fold stability on the restricted REE–REE diagnostic.

Fold-local MoP RMSE and auxiliary MoP $R^2$ for the selected workhorse. This restricted diagnostic is not used for workhorse selection or for the primary all-pair conclusions.

