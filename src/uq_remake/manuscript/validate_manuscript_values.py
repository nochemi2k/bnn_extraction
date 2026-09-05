"""Every number quoted in the Nature Computational Science main text, re-derived.

Complements validate_core_analyses.py (aggregation conventions, model selection)
and validate_scalefree_central.py (the central retention statistic). This module
closes the remaining gaps: per-score rank associations, the flagged-region
chemistry, the scale-free half contrast, scale-free partial correlations,
block-level censoring, the axis-asymmetry counts and the regime decomposition.
"""
import json, os, sys
import numpy as np
import pandas as pd

T = "manuscripts/derived/tables/"
FOLDS = [f"results/downstream_cv/ad_retention/fold_scores/DS2_RDKIT_MLPENS5_HETERO/fold{i}.csv"
         for i in range(5)]
checks = []
import math

def _erfinv_scalar(y):
    """Inverse error function, Winitzki start refined by Newton steps."""
    a = 0.147
    ln = math.log(1 - y * y)
    t1 = 2 / (math.pi * a) + ln / 2
    x = math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)
    for _ in range(4):
        x = x - (math.erf(x) - y) / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x

def chk(n, ok, d=""): checks.append(dict(check=n, passed=bool(ok), detail=str(d)))
def near(a, b, tol=6e-4): return abs(a - b) <= tol

# ---------------------------------------------------------------- A. dataset counts
ds2 = pd.read_csv("data/processed/ds2.csv")
chk("dataset_rows_88525", len(ds2) == 88525, len(ds2))
chk("dataset_pairs_2503", ds2.metal_species_pair.nunique() == 2503, ds2.metal_species_pair.nunique())
chk("dataset_ligands_2187", ds2.ligand_smiles.nunique() == 2187, ds2.ligand_smiles.nunique())
chk("dataset_endpoints_177050", 2 * len(ds2) == 177050, 2 * len(ds2))

# ---------------------------------------------------------------- B. axis asymmetry (Fig 1a)
def share(col, k):
    vc = ds2[col].value_counts(normalize=True)
    return vc.head(k).sum()
chk("axis_ligand_top10_share_14pct", near(share("ligand_smiles", 10), 0.1419, 5e-4),
    f"{share('ligand_smiles',10):.4f}")
chk("axis_electrolyte_six_distinct", ds2.electrolyte.nunique() == 6, ds2.electrolyte.nunique())
chk("axis_electrolyte_top5_share_97pct", near(share("electrolyte", 5), 0.9717, 5e-4),
    f"{share('electrolyte',5):.4f}")
chk("axis_temperature_top1_share_74pct", near(share("temperature_C", 1), 0.7407, 5e-4),
    f"{share('temperature_C',1):.4f}")
chk("axis_ionic_strength_top5_share_93pct", near(share("ionic_strength_M", 5), 0.9271, 5e-4),
    f"{share('ionic_strength_M',5):.4f}")

# ---------------------------------------------------------------- C. per-score rank associations
wp = pd.read_csv(T + "within_pair_score_residual.csv")
wp = wp[wp.unit == "pair_fold_cell"].set_index("score_name")
QUOTED = {"sigma_total": (0.436, 0.029), "sigma_aleatoric": (0.433, 0.028),
          "euclidean_feature_median": (0.225, 0.060),
          "tanimoto_nn_distance": (-0.093, 0.076),
          "tanimoto_knn_mean_5": (-0.108, 0.082),
          "tanimoto_knn_mean_10": (-0.110, 0.095)}
for s, (m, sd) in QUOTED.items():
    chk(f"rho_quoted:{s}", near(wp.loc[s, "mean_rho"], m, 6e-4) and near(wp.loc[s, "fold_sd"], sd, 6e-4),
        f"rho={wp.loc[s,'mean_rho']:.4f}+/-{wp.loc[s,'fold_sd']:.4f} manuscript={m}+/-{sd}")
mh = [wp.loc[k, "mean_rho"] for k in wp.index if k.startswith("mahalanobis")]
chk("all_mahalanobis_variants_between_0.07_and_0.08",
    all(0.065 <= v <= 0.085 for v in mh), f"range {min(mh):.3f}-{max(mh):.3f}")
chk("all_three_fingerprint_nn_variants_negative",
    all(wp.loc[k, "mean_rho"] < 0 for k in
        ["tanimoto_nn_distance", "tanimoto_knn_mean_5", "tanimoto_knn_mean_10"]),
    [round(wp.loc[k, "mean_rho"], 3) for k in
     ["tanimoto_nn_distance", "tanimoto_knn_mean_5", "tanimoto_knn_mean_10"]])

# ---------------------------------------------------------------- D. flagged-region chemistry
fc = pd.read_csv(T + "flagged_region_composition.csv")
ss = fc[fc.row_type == "summary_stat"]
g = lambda c, k: ss[(ss.cell == c) & (ss.key == k)]["mean"].iloc[0]
for cell, key, exp, tol in [("dense_support_high_sigma", "favoured_logK", 12.9, 0.05),
                            ("dense_support_low_sigma", "favoured_logK", 2.9, 0.05),
                            ("dense_support_high_sigma", "abs_dlogK", 4.15, 0.01),
                            ("dense_support_low_sigma", "abs_dlogK", 0.43, 0.01)]:
    chk(f"flagged_region:{cell}:{key}", near(g(cell, key), exp, tol), f"{g(cell,key):.3f} vs {exp}")
ov = fc[fc.row_type == "overlap"]
chk("flagged_region_overlap_33_of_34",
    int(ov[ov.key == "n_pairs_overlap"].n_rows.iloc[0]) == 33 and
    int(ov[(ov.key == "n_pairs_in_cell") & (ov.cell == "dense_support_high_sigma")].n_rows.iloc[0]) == 34,
    "33/34")
chk("flagged_region_isovalent_both_halves",
    g("dense_support_high_sigma", "d_oxidation_state") == 0 and
    g("dense_support_low_sigma", "d_oxidation_state") == 0, "delta oxidation state = 0")

# ------------------------------------------- E. scale-free quantities recomputed from fold scores
def prep(d):
    d = d.copy()
    d["abs_error"] = (d.y_true - d.y_pred).abs()
    gg = d.groupby("metal_species_pair")["y_true"]
    d["sd_p"] = gg.transform(lambda x: x.std(ddof=1)); d["n_p"] = gg.transform("size")
    d = d[(d.n_p >= 10) & (d.sd_p > 0)].copy()
    d["e_norm"] = d.abs_error / d.sd_p
    d["sigma_norm"] = d.sigma_total / d.sd_p
    d["supp"] = (d["n_training_ligands_for_pair"] if "n_training_ligands_for_pair" in d
                 else d.groupby("metal_species_pair")["ligand_smiles"].transform("nunique"))
    return d

def rank(x): return pd.Series(x).rank().values
def pcorr(a, b, c):
    A, B, C = rank(a), rank(b), rank(c)
    def res(y, X):
        X = np.c_[np.ones(len(X)), X]
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(A, C), res(B, C))[0, 1])

hi, lo, sfav, serr = [], [], [], []
for path in FOLDS:
    d = prep(pd.read_csv(path))
    fav = np.maximum(d.logK_x, d.logK_y) if "logK_x" in d else None
    if fav is None:
        m = pd.read_csv("data/processed/ds2.csv")[
            ["ligand_smiles", "metal_species_pair", "logK_x", "logK_y"]].drop_duplicates(
            ["ligand_smiles", "metal_species_pair"])
        d = d.merge(m, on=["ligand_smiles", "metal_species_pair"], how="left")
        fav = np.maximum(d.logK_x, d.logK_y)
    d["favoured"] = fav
    d = d.dropna(subset=["favoured"])
    sfav.append(pcorr(d.sigma_norm, d.favoured, d.supp))
    serr.append(pcorr(d.e_norm, d.supp, d.favoured))
    d["pq"] = pd.qcut(d.supp, 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    d["sq"] = pd.qcut(d.sigma_total, 4, labels=[1, 2, 3, 4]).astype(int)
    dense = d[d.pq == 4]
    hi.append(dense[dense.sq == 4].e_norm.mean()); lo.append(dense[dense.sq == 1].e_norm.mean())
hi, lo = np.array(hi), np.array(lo)
chk("scalefree_half_contrast_high_0.87", near(hi.mean(), 0.87, 0.01), f"{hi.mean():.3f}")
chk("scalefree_half_contrast_low_0.20", near(lo.mean(), 0.20, 0.01), f"{lo.mean():.3f}")
chk("scalefree_partial_sigma_favoured_0.552", near(np.mean(sfav), 0.552, 0.002),
    f"{np.mean(sfav):.4f}+/-{np.std(sfav,ddof=1):.4f}")
chk("scalefree_partial_error_support_is_zero", abs(np.mean(serr)) < 0.02,
    f"{np.mean(serr):+.4f}+/-{np.std(serr,ddof=1):.4f} (manuscript -0.009+/-0.057)")

# ---------------------------------------------------------------- F. block-level censoring
cz = pd.read_csv(T + "logk_censoring.csv")
d2 = cz[cz.dataset == "DS2_endpoints_pooled"].set_index("metal_block")
chk("censoring_lanthanide_2.41pct", near(d2.loc["4f", "frac_le_1"] * 100, 2.41, 0.01),
    f"{d2.loc['4f','frac_le_1']*100:.2f}")
chk("censoring_actinide_24.45pct", near(d2.loc["5f", "frac_le_1"] * 100, 24.45, 0.01),
    f"{d2.loc['5f','frac_le_1']*100:.2f}")
chk("censoring_n_endpoints_177050", int(d2.loc["all", "n"]) == 177050, int(d2.loc["all", "n"]))

# ---------------------------------------------------------------- G. regime decomposition
ur = pd.read_csv(T + "uncertainty_by_regime.csv").set_index(["regime", "metric_name"])
for reg, met, exp, tol in [("high_selectivity", "sigma_total", 2.28, 0.01),
                           ("near_zero", "sigma_total", 0.95, 0.01),
                           ("high_selectivity", "aleatoric_variance_share", 0.894, 0.001),
                           ("near_zero", "aleatoric_variance_share", 0.917, 0.001)]:
    v = ur.loc[(reg, met), "fold_mean"]
    chk(f"regime:{reg}:{met}", near(v, exp, tol), f"{v:.4f} vs {exp}")

# ---------------------------------------------------------------- H. robustness (S8.5)
rb = pd.read_csv(T + "robustness_threshold_and_model.csv")
W = "DS2_RDKIT_MLPENS5_HETERO"
thr = rb[rb.model == W].set_index("min_cell_n")
for n_, cells, unf, rs, rf in [(20, 225, 0.974, 0.436, -0.093),
                               (10, 369, 0.920, 0.395, -0.074),
                               (5, 588, 0.936, 0.355, -0.081)]:
    r = thr.loc[n_]
    chk(f"threshold_sensitivity_n{n_}",
        near(r.unfiltered, unf, 6e-4) and near(r.rho_sigma_total, rs, 6e-4)
        and near(r.rho_tanimoto_nn_distance, rf, 6e-4) and round(r.cells_per_fold) == cells,
        f"cells={r.cells_per_fold:.0f} unf={r.unfiltered:.3f} rho_sig={r.rho_sigma_total:+.3f} rho_fp={r.rho_tanimoto_nn_distance:+.3f}")
chk("threshold_sensitivity_fingerprint_negative_at_every_threshold",
    bool((thr.rho_tanimoto_nn_distance < 0).all()),
    thr.rho_tanimoto_nn_distance.round(3).tolist())
chk("threshold_sensitivity_sigma_positive_at_every_threshold",
    bool((thr.rho_sigma_total > 0.3).all()), thr.rho_sigma_total.round(3).tolist())

mods = rb[rb.min_cell_n == 20].set_index("model")
chk("model_robustness_four_models_present", len(mods) == 4, mods.index.tolist())
chk("model_robustness_skill_range_0.833_to_0.974",
    near(mods.unfiltered.min(), 0.833, 6e-4) and near(mods.unfiltered.max(), 0.974, 6e-4),
    f"{mods.unfiltered.min():.3f}-{mods.unfiltered.max():.3f}")
chk("model_robustness_sigma_positive_in_all_models",
    bool((mods.rho_sigma_total > 0).all()) and near(mods.rho_sigma_total.min(), 0.362, 6e-4)
    and near(mods.rho_sigma_total.max(), 0.436, 6e-4),
    f"range {mods.rho_sigma_total.min():+.3f} to {mods.rho_sigma_total.max():+.3f}")
chk("model_robustness_fingerprint_negative_in_all_models",
    bool((mods.rho_tanimoto_nn_distance < 0).all()),
    mods.rho_tanimoto_nn_distance.round(3).to_dict())
best = mods.unfiltered.idxmin()
chk("model_robustness_best_model_still_shows_negative_fingerprint",
    mods.loc[best, "rho_tanimoto_nn_distance"] < 0,
    f"best={best} unfiltered={mods.loc[best,'unfiltered']:.3f} rho_fp={mods.loc[best,'rho_tanimoto_nn_distance']:+.3f}")

# ------------------------------------- I. the model's held-out behaviour (Fig. S1)
_PRED = os.path.join("results", "cv_predictions", "fold_test_predictions",
                     "DS2_RDKIT_MLPENS5_HETERO")
_fr = []
for _k in range(5):
    _d = pd.read_csv(os.path.join(_PRED, f"fold{_k}", "fold_test_predictions.csv"),
                     usecols=["fold_id", "y_true", "y_pred", "sigma_total",
                              "uncertainty_status"])
    _fr.append(_d[_d.uncertainty_status == "valid"])
_p = pd.concat(_fr, ignore_index=True).dropna(subset=["y_true", "y_pred", "sigma_total"])
_res = (_p.y_pred - _p.y_true).to_numpy()
chk("figS1_rows_pooled_over_folds", len(_p) == 88525, len(_p))
_rmse = float(np.sqrt((_res ** 2).mean()))
chk("figS1_row_level_rmse_2.096", near(_rmse, 2.096, 6e-4), f"{_rmse:.3f}")
_lv = np.arange(0.1, 0.95, 0.1)
_zc = np.array([math.sqrt(2) * _erfinv_scalar(l) for l in _lv])
_z = np.abs(_res) / _p.sigma_total.to_numpy()
_emp = np.array([(_z <= t).mean() for t in _zc])
_ace = float(np.abs(_emp - _lv).mean())
_c90 = float(np.interp(0.9, _lv, _emp))
chk("figS1_row_level_ace_0.049", near(_ace, 0.049, 6e-4), f"{_ace:.3f}")
chk("figS1_row_level_coverage_at_90_is_0.894", near(_c90, 0.894, 6e-4), f"{_c90:.3f}")
chk("figS1_row_level_coverage_inside_screen_band_0.85_0.95",
    0.85 <= _c90 <= 0.95, f"{_c90:.3f}")
chk("figS1_row_level_ace_inside_screen_limit_0.10", _ace <= 0.10, f"{_ace:.3f}")

# ---------------- J. decision-level statistics quoted in the rewritten main text
_ret = pd.read_csv(T + "decision_level_retention.csv")
_con = pd.read_csv(T + "decision_level_contrasts.csv")
_rho = pd.read_csv(T + "decision_level_rho.csv").set_index("score")
_dec = pd.read_csv(T + "aggregation_decomposition.csv")

def _r(unit, score, r=0.50):
    row = _ret[(_ret.unit == unit) & (_ret.score == score) & (_ret.retention == r)]
    return float(row.fold_mean.iloc[0])

def _c(a, b, unit="absolute_dlogK", r=0.50):
    return _con[(_con.unit == unit) & (_con.score_a == a) & (_con.score_b == b)
                & (_con.retention == r)].iloc[0]

chk("absolute_unfiltered_1.323", near(_r("absolute_dlogK", "random", 1.00), 1.323),
    f"{_r('absolute_dlogK', 'random', 1.00):.3f}")
for name, sc, exp in (("sigma", "sigma_total", 0.771),
                      ("descriptor_euclid_median", "euclidean_feature_median", 0.909),
                      ("fingerprint_nn", "tanimoto_nn_distance", 1.276),
                      ("random", "random", 1.288),
                      ("predicted_magnitude", "yhat_mag", 1.509)):
    got = _r("absolute_dlogK", sc)
    chk(f"absolute_r50_{name}_{exp}", near(got, exp), f"{got:.3f}")
chk("absolute_sigma_reduction_is_42pct",
    round(100 * (1 - _r("absolute_dlogK", "sigma_total") / 1.323)) == 42,
    f"{100 * (1 - _r('absolute_dlogK', 'sigma_total') / 1.323):.1f}%")
chk("absolute_best_distance_reduction_is_31pct",
    round(100 * (1 - _r("absolute_dlogK", "euclidean_feature_median") / 1.323)) == 31,
    f"{100 * (1 - _r('absolute_dlogK', 'euclidean_feature_median') / 1.323):.1f}%")

_x = _c("sigma_total", "unfiltered")
chk("contrast_sigma_vs_unfiltered_0.552_ci_excludes_zero",
    near(_x.paired_diff, -0.552, 1.1e-3) and bool(_x.ci_excludes_zero)
    and round(_x.cell_win_fraction, 2) == 0.96,
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}] win {_x.cell_win_fraction:.3f}")
_x = _c("sigma_total", "euclidean_feature_median")
chk("contrast_sigma_beats_best_distance_0.138_ci_excludes_zero",
    near(_x.paired_diff, -0.138, 1.1e-3) and bool(_x.ci_excludes_zero)
    and _x.folds_same_sign and round(_x.cell_win_fraction, 2) == 0.76,
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}] "
    f"same_sign={_x.folds_same_sign} win {_x.cell_win_fraction:.3f}")
_x = _c("euclidean_feature_median", "random")
chk("contrast_descriptor_beats_random_0.380", near(_x.paired_diff, -0.380, 1.1e-3)
    and bool(_x.ci_excludes_zero) and round(_x.cell_win_fraction, 2) == 0.90,
    f"{_x.paired_diff:+.3f} win {_x.cell_win_fraction:.3f}")
_x = _c("tanimoto_nn_distance", "random")
chk("contrast_fingerprint_vs_random_is_null_at_r50",
    near(_x.paired_diff, -0.012, 1.1e-3) and not bool(_x.ci_excludes_zero)
    and 0.46 <= _x.cell_win_fraction <= 0.48,
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}] win {_x.cell_win_fraction:.3f}")
_x = _c("tanimoto_nn_distance", "random", r=0.30)
chk("contrast_fingerprint_worse_than_random_at_r30",
    near(_x.paired_diff, 0.070, 1.1e-3) and bool(_x.ci_excludes_zero),
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}]")
_x = _c("yhat_mag", "unfiltered")
chk("contrast_predicted_magnitude_harmful_0.187",
    near(_x.paired_diff, 0.187, 1.1e-3) and bool(_x.ci_excludes_zero)
    and round(1 - _x.cell_win_fraction, 2) == 0.84,
    f"{_x.paired_diff:+.3f} worse in {1 - _x.cell_win_fraction:.0%} of cells")

_ka = _con[_con.unit == "absolute_dlogK"].set_index(["score_a", "score_b", "retention"]).sort_index()
_kb = _con[_con.unit == "over_cell_sd"].set_index(["score_a", "score_b", "retention"]).sort_index()
_flip = _ka.index[np.sign(_ka.paired_diff.values) != np.sign(_kb.paired_diff.values)].tolist()
chk("unit_change_preserves_every_cell_win_fraction",
    bool(np.allclose(_ka.cell_win_fraction.values, _kb.cell_win_fraction.values)),
    f"{len(_ka)} contrasts, win fractions identical")
chk("unit_change_preserves_every_contrast_sign_except_one_null",
    len(_flip) == 1 and _flip[0] == ("tanimoto_nn_distance", "unfiltered", 0.30)
    and abs(_ka.loc[_flip[0], "paired_diff"]) < 0.01
    and not bool(_ka.loc[_flip[0], "ci_excludes_zero"]),
    f"only {_flip} flips; absolute value "
    f"{_ka.loc[_flip[0], 'paired_diff']:+.3f} with an interval spanning zero")

for name, sc, exp, frac in (("sigma", "sigma_total", 0.436, 0.02),
                            ("descriptor_euclid_median", "euclidean_feature_median", 0.225, 0.11),
                            ("fingerprint_nn", "tanimoto_nn_distance", -0.093, 0.69)):
    r = _rho.loc[sc]
    chk(f"rho_{name}_{exp}_ci_excludes_zero",
        near(r.fold_mean, exp) and (r.ci_low > 0) == (r.ci_high > 0)
        and round(r.frac_cells_negative, 2) == frac,
        f"{r.fold_mean:+.3f} CI[{r.ci_low:+.3f},{r.ci_high:+.3f}] "
        f"negative in {r.frac_cells_negative:.0%}")
chk("rho_training_frequency_undefined_within_cell",
    _rho.loc["pair_density_risk", "status"] == "undefined_within_cell",
    _rho.loc["pair_density_risk", "status"])

_A = _dec[_dec.protocol == "A_pooled_all_pairs"].set_index("score")
chk("pooled_protocol_reproduces_published_unfiltered_2.078",
    near(_A.loc["unfiltered", "fold_mean"], 2.078, 1.1e-3),
    f"{_A.loc['unfiltered', 'fold_mean']:.3f}")
chk("pooled_protocol_reproduces_published_sigma_0.730",
    near(_A.loc["sigma_total", "fold_mean"], 0.730, 1.1e-3),
    f"{_A.loc['sigma_total', 'fold_mean']:.3f}")
chk("pooled_protocol_reproduces_published_mahalanobis_1.586",
    near(_A.loc["mahalanobis_feature_centroid", "fold_mean"], 1.586, 1.1e-3),
    f"{_A.loc['mahalanobis_feature_centroid', 'fold_mean']:.3f}")
chk("pooled_protocol_population_differs_between_scores",
    round(_A.loc["unfiltered", "pairs_retained"]) == 802
    and round(_A.loc["sigma_total", "pairs_retained"]) == 328
    and round(_A.loc["euclidean_feature_median", "pairs_retained"]) == 531,
    f"unfiltered {_A.loc['unfiltered','pairs_retained']:.1f}, "
    f"sigma {_A.loc['sigma_total','pairs_retained']:.1f}, "
    f"euclid {_A.loc['euclidean_feature_median','pairs_retained']:.1f}")
_C = _dec[_dec.protocol == "C_within_cell_decision_level"].set_index("score")
chk("decision_level_population_identical_for_every_score",
    _C.pairs_retained.nunique() == 1, f"{_C.pairs_retained.iloc[0]:.1f} pairs for all scores")
chk("removing_population_selection_improves_distances_and_worsens_sigma",
    _C.loc["euclidean_feature_median", "fold_mean"] < _A.loc["euclidean_feature_median", "fold_mean"]
    and _C.loc["sigma_total", "fold_mean"] > _dec[
        (_dec.protocol == "B_pooled_eligible_cells") & (_dec.score == "sigma_total")].fold_mean.iloc[0],
    "descriptor 1.404->0.909, sigma 0.528->0.771")

# ------------- K. why the pooled protocol changes the pair population (S7.4)
_L = pd.read_csv(T + "pooled_retention_pair_loss.csv").set_index("score")
chk("pair_loss_accounting_adds_up_for_every_score",
    bool((np.abs(_L.pairs_losing_every_row + _L.pairs_below_mop_minimum
                 + _L.pairs_counted - _L.pairs_eligible_unfiltered) < 1e-6).all()),
    f"{len(_L)} scores, counted + below-minimum + zero-row = {_L.pairs_eligible_unfiltered.iloc[0]:.1f}")
for sc, zero, counted in (("pair_density_risk", 692.2, 109.4),
                          ("sigma_total", 321.8, 328.2),
                          ("euclidean_feature_median", 73.6, 530.6),
                          ("tanimoto_nn_distance", 49.2, 606.6),
                          ("random", 21.4, 563.6)):
    r = _L.loc[sc]
    chk(f"pair_loss_{sc}", near(r.pairs_losing_every_row, zero, 0.05)
        and near(r.pairs_counted, counted, 0.05),
        f"{r.pairs_losing_every_row:.1f} lose every row, {r.pairs_counted:.1f} counted")
chk("training_frequency_count_is_constant_within_a_pair",
    near(_L.loc["pair_density_risk", "between_pair_rank_variance_share"], 1.0, 1e-9),
    f"{_L.loc['pair_density_risk', 'between_pair_rank_variance_share']:.3f}")
chk("sigma_varies_mostly_between_pairs_distances_mostly_within",
    _L.loc["sigma_total", "between_pair_rank_variance_share"] > 0.6
    and all(_L.loc[d, "between_pair_rank_variance_share"] < 0.2
            for d in ("euclidean_feature_median", "mahalanobis_feature_centroid",
                      "tanimoto_median_distance", "tanimoto_nn_distance")),
    f"sigma {_L.loc['sigma_total','between_pair_rank_variance_share']:.3f}; distances "
    + ", ".join(f"{_L.loc[d,'between_pair_rank_variance_share']:.3f}"
                for d in ("euclidean_feature_median", "mahalanobis_feature_centroid",
                          "tanimoto_median_distance", "tanimoto_nn_distance")))
chk("distances_keep_about_half_of_each_pair_sigma_keeps_a_fifth",
    near(_L.loc["sigma_total", "median_retained_fraction"], 0.21, 0.005)
    and all(0.48 <= _L.loc[d, "median_retained_fraction"] <= 0.54
            for d in ("euclidean_feature_median", "tanimoto_nn_distance")),
    f"sigma {_L.loc['sigma_total','median_retained_fraction']:.2f}, "
    f"euclid {_L.loc['euclidean_feature_median','median_retained_fraction']:.2f}, "
    f"fp NN {_L.loc['tanimoto_nn_distance','median_retained_fraction']:.2f}")

# ------------------- L. pooled risk ordering and the matched control (S7.1-6.2)
_P = pd.read_csv(T + "pooled_retention.csv")
_P = _P[_P.population == "free"]
_M = pd.read_csv(T + "pooled_retention_matched.csv")
_PC = pd.read_csv(T + "pooled_contrasts.csv")

def _p(score, r=0.50, col="fold_mean"):
    return float(_P[(_P.score == score) & (_P.retention == r)][col].iloc[0])

def _m(score, label="matched_five_scores", r=0.50, col="fold_mean"):
    return float(_M[(_M.population == label) & (_M.score == score)
                    & (_M.retention == r)][col].iloc[0])

chk("pooled_unfiltered_2.078_over_801.6_pairs",
    near(_p("unfiltered", 1.00), 2.078, 1.1e-3)
    and near(_p("unfiltered", 1.00, "pairs_retained"), 801.6, 0.05),
    f"{_p('unfiltered', 1.00):.3f} over {_p('unfiltered', 1.00, 'pairs_retained'):.1f} pairs")
for name, sc, exp in (("sigma", "sigma_total", 0.730),
                      ("training_frequency_count", "pair_density_risk", 0.955),
                      ("descriptor_euclid_median", "euclidean_feature_median", 1.404),
                      ("descriptor_mahalanobis_centroid", "mahalanobis_feature_centroid", 1.586),
                      ("random", "random", 1.823),
                      ("fingerprint_nn", "tanimoto_nn_distance", 1.888),
                      ("predicted_magnitude", "yhat_mag", 2.361)):
    chk(f"pooled_r50_{name}_{exp}", near(_p(sc), exp, 1.1e-3), f"{_p(sc):.3f}")
chk("pooled_sigma_reduction_is_65pct",
    round(100 * (1 - _p("sigma_total") / _p("unfiltered", 1.00))) == 65,
    f"{100 * (1 - _p('sigma_total') / _p('unfiltered', 1.00)):.1f}%")
chk("pooled_best_distance_reduction_is_32pct",
    round(100 * (1 - _p("euclidean_feature_median") / _p("unfiltered", 1.00))) == 32,
    f"{100 * (1 - _p('euclidean_feature_median') / _p('unfiltered', 1.00)):.1f}%")
chk("pooled_fingerprint_nn_is_worse_than_random",
    _p("tanimoto_nn_distance") > _p("random"),
    f"fp NN {_p('tanimoto_nn_distance'):.3f} vs random {_p('random'):.3f}")
chk("pooled_count_beats_every_molecular_distance",
    all(_p("pair_density_risk") < _p(d) for d in
        ("euclidean_feature_median", "euclidean_feature_nn",
         "mahalanobis_feature_centroid", "mahalanobis_feature_nn",
         "tanimoto_median_distance", "tanimoto_nn_distance")),
    f"count {_p('pair_density_risk'):.3f} vs best distance "
    f"{_p('euclidean_feature_median'):.3f}")

# the AUC convention reproduces the published values for this dataset
for sc, exp in (("sigma_total", 0.835), ("mahalanobis_feature_centroid", 1.441),
                ("tanimoto_median_distance", 1.499), ("tanimoto_nn_distance", 1.717)):
    got = float(_P[_P.score == sc].auc.iloc[0])
    chk(f"pooled_auc_{sc}_{exp}", near(got, exp, 2.1e-3), f"{got:.3f}")

chk("matched_population_is_296_pairs",
    near(_m("sigma_total", col="pairs_shared"), 296.4, 0.05),
    f"{_m('sigma_total', col='pairs_shared'):.1f}")
chk("matched_sigma_barely_moves_0.730_to_0.720",
    near(_m("sigma_total"), 0.720, 1.1e-3), f"{_m('sigma_total'):.3f}")
chk("matched_distances_improve_sharply",
    near(_m("euclidean_feature_median"), 0.843, 1.1e-3)
    and near(_m("mahalanobis_feature_centroid"), 1.045, 1.1e-3)
    and near(_m("tanimoto_nn_distance"), 1.249, 1.1e-3),
    f"euclid {_m('euclidean_feature_median'):.3f}, mahal "
    f"{_m('mahalanobis_feature_centroid'):.3f}, fp NN {_m('tanimoto_nn_distance'):.3f}")
chk("matched_including_count_makes_the_count_the_worst_of_seven",
    _m("pair_density_risk", "matched_including_count") ==
    max(_m(x, "matched_including_count") for x in
        ("sigma_total", "euclidean_feature_median", "mahalanobis_feature_centroid",
         "tanimoto_median_distance", "tanimoto_nn_distance", "random",
         "pair_density_risk"))
    and near(_m("pair_density_risk", "matched_including_count"), 0.949, 1.1e-3)
    and near(_m("sigma_total", "matched_including_count"), 0.557, 1.1e-3),
    f"count {_m('pair_density_risk', 'matched_including_count'):.3f} is the worst; "
    f"sigma {_m('sigma_total', 'matched_including_count'):.3f} the best, on "
    f"{_m('sigma_total', 'matched_including_count', col='pairs_shared'):.1f} pairs")

def _pc(a, b, r=0.50):
    return _PC[(_PC.score_a == a) & (_PC.score_b == b) & (_PC.retention == r)].iloc[0]

_x = _pc("sigma_total", "euclidean_feature_median")
chk("matched_contrast_sigma_beats_best_distance_0.123",
    near(_x.paired_diff, -0.123, 1.1e-3) and bool(_x.ci_excludes_zero),
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}] "
    f"pairs won {_x.pair_win_fraction:.2f}")
_x = _pc("sigma_total", "random")
chk("matched_contrast_sigma_beats_random_0.527",
    near(_x.paired_diff, -0.527, 1.1e-3) and bool(_x.ci_excludes_zero)
    and round(_x.pair_win_fraction, 2) == 0.85,
    f"{_x.paired_diff:+.3f} pairs won {_x.pair_win_fraction:.2f}")
_x = _pc("tanimoto_nn_distance", "random")
chk("matched_contrast_fingerprint_is_null_at_r50",
    abs(_x.paired_diff) < 0.005 and not bool(_x.ci_excludes_zero),
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}]")
_x = _pc("tanimoto_nn_distance", "random", 0.30)
chk("matched_contrast_fingerprint_worse_than_random_at_r30",
    near(_x.paired_diff, 0.153, 1.1e-3) and bool(_x.ci_excludes_zero),
    f"{_x.paired_diff:+.3f} CI[{_x.ci_low:+.3f},{_x.ci_high:+.3f}]")

# ------------- M. the coverage x uncertainty grid in absolute units (main-text Fig. 2)
_G = pd.read_csv(T + "coverage_override_grid.csv")
_G = _G[_G.aggregation == "fold_mean_plus_sd"]

def _gs(section, sup=None, sig=None):
    d = _G[_G.table_section == section]
    if sup is not None:
        d = d[d.support_quartile == sup]
    if sig is not None:
        d = d[d.sigma_quartile == sig]
    return float(d.mean_abs_error.iloc[0])

_sig = [_gs("sigma_marginal", sig=q) for q in (1, 2, 3, 4)]
_sup = [_gs("support_marginal", sup=q) for q in (1, 2, 3, 4)]
chk("fig2_sigma_marginal_0.25_to_2.57",
    near(_sig[0], 0.252, 1.1e-3) and near(_sig[-1], 2.569, 1.1e-3)
    and all(_sig[i] < _sig[i + 1] for i in range(3)),
    " / ".join(f"{v:.3f}" for v in _sig) + f"  span {_sig[-1] / _sig[0]:.1f}x")
chk("fig2_support_marginal_2.09_to_0.94_not_monotone",
    near(_sup[0], 2.087, 1.1e-3) and near(_sup[-1], 0.936, 1.1e-3)
    and not all(_sup[i] > _sup[i + 1] for i in range(3)),
    " / ".join(f"{v:.3f}" for v in _sup) + "  (rises again in Q4)")
chk("fig2_best_covered_row_0.28_to_1.82",
    near(_gs("grid", 4, 1), 0.275, 1.1e-3) and near(_gs("grid", 4, 4), 1.817, 1.1e-3),
    f"{_gs('grid', 4, 1):.3f} -> {_gs('grid', 4, 4):.3f} across sigma quartiles "
    "within the densest support quartile")

# ------------- N. pair acquisition (main-text Fig. 3 and S9)
_A = pd.read_csv("manuscripts/final/tables/pair_acquisition_hit_and_precision.csv")
_R = pd.read_csv("manuscripts/final/tables/pair_acquisition_regret.csv")
_HS = _A[(_A.task == "high_selectivity") & (_A.orientation == "both")].set_index("strategy")
_NZ = _A[_A.task == "low_selectivity"].set_index("strategy")
_RHS = _R[(_R.task == "high_selectivity") & (_R.orientation == "both")].set_index("strategy")
_RNZ = _R[_R.task == "low_selectivity"].set_index("strategy")

chk("s8_pair_task_counts_945_and_463",
    near(_HS.n_tasks_fold_mean.iloc[0], 945.0, 0.05)
    and near(_NZ.n_tasks_fold_mean.iloc[0], 463.0, 0.05)
    and near(_HS.n_tasks_fold_sd.iloc[0], 66.63, 0.01)
    and near(_NZ.n_tasks_fold_sd.iloc[0], 56.18, 0.01),
    f"high {_HS.n_tasks_fold_mean.iloc[0]:.1f}+-{_HS.n_tasks_fold_sd.iloc[0]:.1f}, "
    f"near-zero {_NZ.n_tasks_fold_mean.iloc[0]:.1f}+-{_NZ.n_tasks_fold_sd.iloc[0]:.1f}")

chk("fig3_standardised_margin_high_selectivity_0.831_0.716_0.596",
    near(_HS.loc["expected_improvement", "event_precision_at_1"], 0.831, 1.1e-3)
    and near(_HS.loc["expected_improvement", "event_precision_at_5"], 0.716, 1.1e-3)
    and near(_HS.loc["expected_improvement", "event_precision_at_10"], 0.596, 1.1e-3),
    "k=1/5/10 " + " / ".join(
        f"{_HS.loc['expected_improvement', f'event_precision_at_{k}']:.3f}" for k in (1, 5, 10)))

chk("fig3_standardised_margin_leads_high_selectivity_at_every_k",
    all(_HS.loc["expected_improvement", f"event_precision_at_{k}"]
        == _HS[f"event_precision_at_{k}"].max() for k in (1, 5, 10)),
    "; ".join(f"k={k}: best is {_HS[f'event_precision_at_{k}'].idxmax()}" for k in (1, 5, 10)))

chk("fig3_lower_bound_costs_more_as_penalty_grows_0.795_to_0.669",
    near(_HS.loc["lcb_1sigma", "event_precision_at_1"], 0.795, 1.1e-3)
    and near(_HS.loc["lcb_3sigma", "event_precision_at_1"], 0.669, 1.1e-3)
    and _HS.loc["lcb_1sigma", "event_precision_at_1"]
        > _HS.loc["lcb_2sigma", "event_precision_at_1"]
        > _HS.loc["lcb_3sigma", "event_precision_at_1"],
    " > ".join(f"{_HS.loc[f'lcb_{z}sigma', 'event_precision_at_1']:.3f}" for z in (1, 2, 3)))

chk("fig3_near_zero_lower_bounds_lead_0.806_to_0.811_over_value_only_0.748",
    near(_NZ.loc["lcb_1sigma", "event_precision_at_1"], 0.806, 1.1e-3)
    and near(_NZ.loc["lcb_2sigma", "event_precision_at_1"], 0.811, 1.1e-3)
    and near(_NZ.loc["y_pred", "event_precision_at_1"], 0.748, 1.1e-3)
    and near(_NZ.loc["expected_improvement", "event_precision_at_1"], 0.799, 1.1e-3)
    and _NZ[[f"event_precision_at_1"]].idxmax().iloc[0] == "lcb_2sigma",
    f"margin {_NZ.loc['expected_improvement', 'event_precision_at_1']:.3f}, "
    f"lcb1 {_NZ.loc['lcb_1sigma', 'event_precision_at_1']:.3f}, "
    f"lcb2 {_NZ.loc['lcb_2sigma', 'event_precision_at_1']:.3f}, "
    f"value {_NZ.loc['y_pred', 'event_precision_at_1']:.3f}")

chk("fig3_margin_within_one_fold_sd_of_best_near_zero",
    abs(_NZ.loc["expected_improvement", "event_precision_at_1"]
        - _NZ["event_precision_at_1"].max())
    <= _NZ.loc["lcb_2sigma", "event_precision_at_1_sd"],
    f"gap {_NZ['event_precision_at_1'].max() - _NZ.loc['expected_improvement', 'event_precision_at_1']:+.3f} "
    f"vs fold s.d. {_NZ.loc['lcb_2sigma', 'event_precision_at_1_sd']:.3f}")

chk("fig3_random_shortlist_0.639_and_0.569",
    near(_HS.loc["random", "event_precision_at_1"], 0.639, 1.1e-3)
    and near(_NZ.loc["random", "event_precision_at_1"], 0.569, 1.1e-3)
    and _HS["event_precision_at_1"].min() == _HS.loc["random", "event_precision_at_1"]
    and _NZ["event_precision_at_1"].min() == _NZ.loc["random", "event_precision_at_1"],
    f"high {_HS.loc['random', 'event_precision_at_1']:.3f}, "
    f"near-zero {_NZ.loc['random', 'event_precision_at_1']:.3f}, worst rule in both")

chk("s8_any_hit_saturates_above_0.95_for_model_rules",
    near(_HS.loc["expected_improvement", "any_hit_at_5"], 0.962, 1.1e-3)
    and near(_HS.loc["random", "any_hit_at_5"], 0.852, 1.1e-3)
    and _NZ.loc[["expected_improvement", "lcb_1sigma", "lcb_2sigma", "lcb_3sigma",
                 "y_pred"], "any_hit_at_5"].min() > 0.975
    and _NZ.loc["random", "any_hit_at_5"] < 0.92,
    f"high margin {_HS.loc['expected_improvement', 'any_hit_at_5']:.3f} vs random "
    f"{_HS.loc['random', 'any_hit_at_5']:.3f}; near-zero min over model rules "
    f"{_NZ.loc[['expected_improvement', 'lcb_1sigma', 'lcb_2sigma', 'lcb_3sigma', 'y_pred'], 'any_hit_at_5'].min():.3f} "
    f"vs random {_NZ.loc['random', 'any_hit_at_5']:.3f}")

chk("s8_regret_is_won_by_the_value_only_rule",
    _RHS["regret_at_1"].idxmin() == "y_pred"
    and near(_RHS.loc["y_pred", "regret_at_1"], 1.738, 1.1e-3)
    and near(_RHS.loc["expected_improvement", "regret_at_1"], 2.508, 1.1e-3)
    and near(_RHS.loc["random", "regret_at_1"], 4.211, 1.1e-3)
    and near(_RHS.pool_optimum.iloc[0], 7.535, 1.1e-3),
    f"value {_RHS.loc['y_pred', 'regret_at_1']:.3f} < margin "
    f"{_RHS.loc['expected_improvement', 'regret_at_1']:.3f} < random "
    f"{_RHS.loc['random', 'regret_at_1']:.3f}; pool optimum "
    f"{_RHS.pool_optimum.iloc[0]:.3f}")

chk("s8_near_zero_regret_follows_event_precision",
    _RNZ["regret_at_1"].idxmin() == "lcb_1sigma"
    and near(_RNZ.loc["lcb_1sigma", "regret_at_1"], 0.515, 1.1e-3)
    and near(_RNZ.loc["y_pred", "regret_at_1"], 0.708, 1.1e-3)
    and near(_RNZ.loc["random", "regret_at_1"], 1.585, 1.1e-3),
    f"lcb1 {_RNZ.loc['lcb_1sigma', 'regret_at_1']:.3f}, value "
    f"{_RNZ.loc['y_pred', 'regret_at_1']:.3f}, random "
    f"{_RNZ.loc['random', 'regret_at_1']:.3f}")

# ------------- O. dataset breakdown quoted in S2.5
_F = pd.concat([pd.read_csv(f) for f in FOLDS], ignore_index=True)
_ay = _F.y_true.abs()
chk("s1_5_near_zero_share_49.21_and_median_1.02",
    near((_ay < 1).mean() * 100, 49.21, 5e-3) and near(_ay.median(), 1.022, 1.1e-3),
    f"{(_ay < 1).mean() * 100:.2f}% within |dlogK|<1, median {_ay.median():.3f}")
_cnt = _F.metal_species_pair.value_counts()
_top5 = _cnt.sort_values(ascending=False).head(int(len(_cnt) * 0.05)).sum() / _cnt.sum()
chk("s1_5_records_per_pair_median_4_iqr_2_21_max_1878_top5_54.2",
    _cnt.median() == 4 and _cnt.quantile(0.25) == 2 and _cnt.quantile(0.75) == 21
    and _cnt.max() == 1878 and near(_top5 * 100, 54.23, 0.05),
    f"median {_cnt.median():.0f}, IQR {_cnt.quantile(0.25):.0f}-{_cnt.quantile(0.75):.0f}, "
    f"max {_cnt.max()}, top 5% carry {_top5 * 100:.1f}% of rows over {len(_cnt)} pairs")

# ------------- P. the structural claims about which distances work (main-text Results)
_I = pd.read_csv("manuscripts/final/tables/iz_risk_score_spearman.csv")
_DESC = ["mahalanobis_feature_nn", "mahalanobis_feature_knn_mean_5",
         "mahalanobis_feature_knn_mean_10", "mahalanobis_feature_centroid",
         "mahalanobis_feature_median", "euclidean_feature_nn", "euclidean_feature_median"]
_LIG = ["tanimoto_nn_distance", "tanimoto_knn_mean_5", "tanimoto_knn_mean_10",
        "tanimoto_median_distance"] + [s for s in _I.score if s.startswith("chemprop_")]
_r = _I.set_index("score").fold_mean
_sd = _I.set_index("score").fold_sd

chk("results_23_scores_and_exactly_10_negative",
    len(_I) == 23 and int((_I.fold_mean < 0).sum()) == 10,
    f"{len(_I)} scores, {(_I.fold_mean < 0).sum()} negative")
chk("results_the_10_negatives_are_the_embeddings_plus_fingerprint_NN_variants",
    set(_I[_I.fold_mean < 0].score)
    == set([s for s in _I.score if s.startswith("chemprop_")])
    | {"tanimoto_nn_distance", "tanimoto_knn_mean_5", "tanimoto_knn_mean_10"},
    f"{sorted(_I[_I.fold_mean < 0].score)}")
chk("results_all_seven_descriptor_distances_positive_0.092_to_0.250",
    len(_DESC) == 7 and (_r[_DESC] > 0).all()
    and near(_r[_DESC].min(), 0.092, 1.1e-3) and near(_r[_DESC].max(), 0.250, 1.1e-3),
    f"{len(_DESC)} descriptor scores, min {_r[_DESC].min():+.3f}, max {_r[_DESC].max():+.3f}")
chk("results_ten_of_eleven_ligand_only_scores_negative",
    len(_LIG) == 11 and int((_r[_LIG] < 0).sum()) == 10
    and list(_r[_LIG][_r[_LIG] > 0].index) == ["tanimoto_median_distance"],
    f"{len(_LIG)} ligand-only or learned-representation scores, "
    f"{(_r[_LIG] < 0).sum()} negative; the exception is "
    f"{list(_r[_LIG][_r[_LIG] > 0].index)}")
chk("results_fingerprint_median_0.102_spread_nearly_as_large_as_value",
    near(_r["tanimoto_median_distance"], 0.102, 1.1e-3)
    and near(_sd["tanimoto_median_distance"], 0.096, 1.1e-3)
    and 0.8 < _sd["tanimoto_median_distance"] / _r["tanimoto_median_distance"] < 1.0,
    f"{_r['tanimoto_median_distance']:+.3f} +- {_sd['tanimoto_median_distance']:.3f}, "
    f"ratio {_sd['tanimoto_median_distance'] / _r['tanimoto_median_distance']:.2f}")

# ------------- Q. the median is the best aggregation in every distance family
_FAMILIES = {
    "descriptor Euclidean": ["euclidean_feature_median", "euclidean_feature_nn"],
    "descriptor Mahalanobis": ["mahalanobis_feature_median",
                               "mahalanobis_feature_centroid",
                               "mahalanobis_feature_nn",
                               "mahalanobis_feature_knn_mean_5",
                               "mahalanobis_feature_knn_mean_10"],
    "fingerprint": ["tanimoto_median_distance", "tanimoto_nn_distance",
                    "tanimoto_knn_mean_5", "tanimoto_knn_mean_10"],
}
_rho = pd.read_csv("manuscripts/final/tables/iz_risk_score_spearman.csv") \
    .set_index("score").fold_mean
_ret = pd.read_csv(T + "pooled_retention.csv")
_ret = _ret[(_ret.population == "free") & (_ret.retention == 0.50)] \
    .set_index("score")

for fam, members in _FAMILIES.items():
    med = members[0]
    best_rho = max(members, key=lambda s: float(_rho[s]))
    best_ret = min(members, key=lambda s: float(_ret.loc[s, "fold_mean"]))
    chk(f"fig1_median_is_best_aggregation_{fam.replace(' ', '_')}",
        best_rho == med and best_ret == med,
        f"rho best {best_rho} ({_rho[best_rho]:+.3f}); retained-error best {best_ret} "
        f"({_ret.loc[best_ret, 'fold_mean']:.3f}); median is "
        f"{_rho[med]:+.3f} / {_ret.loc[med, 'fold_mean']:.3f}")

chk("fig1_nearest_neighbour_is_worst_in_euclidean_and_fingerprint",
    max(_FAMILIES["descriptor Euclidean"],
        key=lambda s: float(_ret.loc[s, "fold_mean"])) == "euclidean_feature_nn"
    and max(_FAMILIES["fingerprint"],
            key=lambda s: float(_ret.loc[s, "fold_mean"])) == "tanimoto_nn_distance",
    "; ".join(f"{f}: worst is "
              f"{max(m, key=lambda s: float(_ret.loc[s, 'fold_mean']))}"
              for f, m in _FAMILIES.items()))

chk("s5_2_mahalanobis_global_aggregations_beat_every_neighbour_based_one",
    max(float(_ret.loc[s, "fold_mean"]) for s in ["mahalanobis_feature_median",
                                                  "mahalanobis_feature_centroid"])
    < min(float(_ret.loc[s, "fold_mean"]) for s in ["mahalanobis_feature_nn",
                                                    "mahalanobis_feature_knn_mean_5",
                                                    "mahalanobis_feature_knn_mean_10"])
    and all(near(_ret.loc[s, "fold_mean"], v, 1.1e-3) for s, v in
            [("mahalanobis_feature_median", 1.580),
             ("mahalanobis_feature_centroid", 1.586),
             ("mahalanobis_feature_nn", 1.635),
             ("mahalanobis_feature_knn_mean_5", 1.637),
             ("mahalanobis_feature_knn_mean_10", 1.643)]),
    "median 1.580, centroid 1.586 < NN 1.635, kNN-5 1.637, kNN-10 1.643")

chk("s6_1_mahalanobis_median_1.580_on_548.4_pairs_auc_1.439",
    near(_ret.loc["mahalanobis_feature_median", "fold_mean"], 1.580, 1.1e-3)
    and near(_ret.loc["mahalanobis_feature_median", "pairs_retained"], 548.4, 0.05)
    and near(_ret.loc["mahalanobis_feature_median", "auc"], 1.439, 1.1e-3),
    f"{_ret.loc['mahalanobis_feature_median', 'fold_mean']:.3f} on "
    f"{_ret.loc['mahalanobis_feature_median', 'pairs_retained']:.1f} pairs, "
    f"AUC {_ret.loc['mahalanobis_feature_median', 'auc']:.3f}")

chk("fig1_six_scores_at_r50_in_the_quoted_order",
    [round(float(_ret.loc[s, "fold_mean"]), 3) for s in
     ["sigma_total", "pair_density_risk", "euclidean_feature_median",
      "mahalanobis_feature_median", "tanimoto_median_distance", "random"]]
    == [0.730, 0.955, 1.404, 1.580, 1.702, 1.823],
    "; ".join(f"{s} {_ret.loc[s, 'fold_mean']:.3f}" for s in
              ["sigma_total", "pair_density_risk", "euclidean_feature_median",
               "mahalanobis_feature_median", "tanimoto_median_distance", "random"]))

# ------------- R. the absolute coverage grid tabulated in S8.1
_ABSGRID = [[0.545, 0.821, 1.438, 2.974],
            [0.280, 0.650, 1.321, 2.524],
            [0.223, 0.509, 0.940, 2.150],
            [0.275, 0.545, 0.898, 1.817]]
_bad = [(i + 1, j + 1) for i in range(4) for j in range(4)
        if not near(_gs("grid", i + 1, j + 1), _ABSGRID[i][j], 1.1e-3)]
chk("s7_1_absolute_grid_all_sixteen_cells", not _bad,
    f"mismatched cells {_bad}" if _bad else "all sixteen match to three decimals")
chk("s7_1_absolute_grid_overall_1.152",
    near(_gs("overall"), 1.152, 1.1e-3), f"{_gs('overall'):.3f}")

# ------------- S. any-hit values quoted for Fig. 3b,e
_MODEL_RULES = ["expected_improvement", "lcb_1sigma", "lcb_2sigma", "lcb_3sigma",
                "y_pred"]
chk("fig3_any_hit_near_zero_all_model_rules_above_0.975",
    _NZ.loc[_MODEL_RULES, "any_hit_at_5"].min() > 0.975
    and near(_NZ.loc["random", "any_hit_at_5"], 0.913, 1.1e-3),
    f"min {_NZ.loc[_MODEL_RULES, 'any_hit_at_5'].min():.3f} vs random "
    f"{_NZ.loc['random', 'any_hit_at_5']:.3f}")
chk("fig3_any_hit_high_selectivity_heavy_penalties_at_or_below_random",
    near(_HS.loc["expected_improvement", "any_hit_at_5"], 0.962, 1.1e-3)
    and near(_HS.loc["lcb_2sigma", "any_hit_at_5"], 0.854, 1.1e-3)
    and near(_HS.loc["lcb_3sigma", "any_hit_at_5"], 0.813, 1.1e-3)
    and near(_HS.loc["random", "any_hit_at_5"], 0.852, 1.1e-3)
    and _HS.loc["lcb_3sigma", "any_hit_at_5"] < _HS.loc["random", "any_hit_at_5"],
    f"margin {_HS.loc['expected_improvement', 'any_hit_at_5']:.3f}, "
    f"lcb2 {_HS.loc['lcb_2sigma', 'any_hit_at_5']:.3f}, "
    f"lcb3 {_HS.loc['lcb_3sigma', 'any_hit_at_5']:.3f}, "
    f"random {_HS.loc['random', 'any_hit_at_5']:.3f}")

# ------------- T. a second model and its own learned embedding (S7.6, Fig. S8)
_E = pd.read_csv(T + "embedding_retention.csv")
_E50 = _E[_E.retention == 0.50].set_index("score")
_EU = float(_E[_E.score == "unfiltered"].fold_mean.iloc[0])

chk("s6_6_dmpnn_unfiltered_2.054",
    near(_EU, 2.054, 1.1e-3)
    and near(float(_E[_E.score == "unfiltered"].fold_sd.iloc[0]), 0.216, 1.1e-3),
    f"{_EU:.3f} +- {float(_E[_E.score == 'unfiltered'].fold_sd.iloc[0]):.3f}")

for _s, _v, _p, _a in [("sigma_total", 0.858, 359.6, 0.907),
                       ("euclidean_feature_median", 1.404, 530.6, 1.301),
                       ("tanimoto_median_distance", 1.676, 562.8, 1.471),
                       ("random", 1.798, 561.0, 1.592),
                       ("chemprop_std_euclidean_nn", 1.950, 569.4, 1.725),
                       ("chemprop_mahalanobis_nn", 1.961, 597.2, 1.756)]:
    chk(f"s6_6_dmpnn_{_s}_{_v}",
        near(_E50.loc[_s, "fold_mean"], _v, 1.1e-3)
        and near(_E50.loc[_s, "pairs_retained"], _p, 0.06)
        and near(_E50.loc[_s, "auc"], _a, 1.1e-3),
        f"{_E50.loc[_s, 'fold_mean']:.3f} on {_E50.loc[_s, 'pairs_retained']:.1f} "
        f"pairs, AUC {_E50.loc[_s, 'auc']:.3f}")

_emb = ["chemprop_std_euclidean_nn", "chemprop_mahalanobis_nn"]
_gap = {(s, r): float(_E[(_E.score == s) & (_E.retention == r)].fold_mean.iloc[0])
        - float(_E[(_E.score == "random") & (_E.retention == r)].fold_mean.iloc[0])
        for s in _emb
        for r in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]}
_upto80 = [v for (s, r), v in _gap.items() if r <= 0.80]
chk("s6_6_both_embedding_distances_above_random_from_10_to_80_percent",
    all(v > 0 for v in _upto80)
    and near(min(_upto80), 0.032, 1.1e-3) and near(max(_upto80), 0.654, 1.1e-3),
    f"gap over random ranges {min(_upto80):+.3f} to {max(_upto80):+.3f} across "
    f"both scores and r <= 0.80")
chk("s6_6_embedding_converges_on_random_at_90_percent",
    all(_gap[(s, 0.90)] < 0 for s in _emb)
    and all(abs(_gap[(s, 0.90)]) < 0.16 for s in _emb),
    "; ".join(f"{s} {_gap[(s, 0.90)]:+.3f}" for s in _emb)
    + " at r = 0.90, inside the fold s.d.")
chk("s6_6_dmpnn_ordering_reproduces_the_workhorse",
    list(_E50.loc[["sigma_total", "euclidean_feature_median",
                   "tanimoto_median_distance", "random",
                   "chemprop_std_euclidean_nn", "chemprop_mahalanobis_nn"],
                  "fold_mean"]) == sorted(_E50.loc[
        ["sigma_total", "euclidean_feature_median", "tanimoto_median_distance",
         "random", "chemprop_std_euclidean_nn", "chemprop_mahalanobis_nn"],
        "fold_mean"]),
    "sigma < descriptor < fingerprint < random < both embedding distances")

# ------------- U. normalized error by regime (S4.4, Fig. S3b)
_R = pd.read_csv(T + "regime_normalized_error.csv")
_rg = lambda reg, m: _R[(_R.regime == reg) & (_R.metric_name == m)].iloc[0]
chk("s3_4_normalized_sigma_is_indistinguishable_between_regimes",
    near(_rg("high_selectivity", "normalized_sigma_total").fold_mean, 1.010, 1.1e-3)
    and near(_rg("near_zero", "normalized_sigma_total").fold_mean, 1.009, 1.1e-3)
    and abs(_rg("high_selectivity", "normalized_sigma_total").fold_mean
            - _rg("near_zero", "normalized_sigma_total").fold_mean)
    < _rg("high_selectivity", "normalized_sigma_total").fold_sd,
    f"{_rg('high_selectivity', 'normalized_sigma_total').fold_mean:.3f} vs "
    f"{_rg('near_zero', 'normalized_sigma_total').fold_mean:.3f}, fold s.d. "
    f"{_rg('high_selectivity', 'normalized_sigma_total').fold_sd:.3f}")
chk("s3_4_normalized_error_still_separates_the_regimes",
    near(_rg("high_selectivity", "normalized_error").fold_mean, 1.120, 1.1e-3)
    and near(_rg("near_zero", "normalized_error").fold_mean, 0.737, 1.1e-3)
    and (_rg("high_selectivity", "normalized_error").fold_mean
         > _rg("near_zero", "normalized_error").fold_mean),
    f"{_rg('high_selectivity', 'normalized_error').fold_mean:.3f} vs "
    f"{_rg('near_zero', 'normalized_error').fold_mean:.3f}")

# ------------- V. the ligand axis quoted in S2.5 and the Introduction
_L = pd.concat([pd.read_csv(f, usecols=["ligand_smiles"]) for f in FOLDS],
               ignore_index=True).ligand_smiles.value_counts()
_cs = np.cumsum(_L.values) / _L.values.sum() * 100
chk("s1_5_ligand_axis_2187_top10_14.19_top100_51.87",
    len(_L) == 2187 and near(_cs[9], 14.19, 5e-3) and near(_cs[99], 51.87, 5e-3),
    f"{len(_L)} ligands; top 10 carry {_cs[9]:.2f}%, top 100 carry {_cs[99]:.2f}%")

# ------------- W. the seven retention rows (Supplementary S7.1) and the framing
#                  values of the opening Results section
_T1 = [("sigma_total", 0.730, -65, 328.2, 0.835),
       ("pair_density_risk", 0.955, -54, 109.4, 1.212),
       ("euclidean_feature_median", 1.404, -32, 530.6, 1.301),
       ("mahalanobis_feature_median", 1.580, -24, 548.4, 1.439),
       ("tanimoto_median_distance", 1.702, -18, 562.8, 1.499),
       ("random", 1.823, -12, 561.0, 1.612),
       ("tanimoto_nn_distance", 1.888, -9, 606.6, 1.717)]
_pr = pd.read_csv(T + "pooled_retention.csv")
_u = _pr[_pr.score == "unfiltered"].iloc[0]
chk("table1_unfiltered_2.078_on_801.6_pairs",
    near(_u.fold_mean, 2.078, 1.1e-3) and near(_u.fold_sd, 0.130, 1.1e-3)
    and near(_u.pairs_retained, 801.6, 0.06),
    f"{_u.fold_mean:.3f} +- {_u.fold_sd:.3f} on {_u.pairs_retained:.1f} pairs")
_bad = []
for s, v, pct, pairs, auc in _T1:
    r = _ret.loc[s]
    got_pct = round(100 * (float(r.fold_mean) - float(_u.fold_mean)) / float(_u.fold_mean))
    if not (near(r.fold_mean, v, 1.1e-3) and near(r.pairs_retained, pairs, 0.06)
            and near(r.auc, auc, 1.1e-3) and got_pct == pct):
        _bad.append((s, round(float(r.fold_mean), 3), got_pct,
                     round(float(r.pairs_retained), 1), round(float(r.auc), 3)))
chk("table1_every_row_reproduces", not _bad,
    f"mismatches {_bad}" if _bad else "all seven rows match value, percent change, pairs and area")

_dl = pd.read_csv(T + "decision_level_retention.csv")
_nrm = _dl[(_dl.unit != "absolute_dlogK") & (_dl.score == "random") & (_dl.retention == 1.0)].iloc[0]
chk("results_unfiltered_normalized_0.974_barely_beats_the_pair_mean",
    near(_nrm.fold_mean, 0.974, 1.1e-3) and near(_nrm.fold_sd, 0.256, 1.1e-3)
    and _nrm.fold_mean < 1.0,
    f"{_nrm.fold_mean:.3f} +- {_nrm.fold_sd:.3f}; predicting each cell's mean gives 1.000")

# ------------- X. what the published benchmark measures (Results opening, Fig. 1a, S3.3)
_B = pd.read_csv(T + "benchmark_reproduction.csv").set_index("regime")
_bt = _B.loc["published split, test partition"]
_be = _B.loc["published split, external holdout"]
_bd = _B.loc["ligand-disjoint cross-validation"]
chk("results_published_test_0.599_mae_0.934_r2",
    near(_bt.published_mae, 0.599, 1.1e-3) and near(_bt.published_r2, 0.934, 1.1e-3),
    f"published MAE {_bt.published_mae:.3f}, R2 {_bt.published_r2:.3f}")
chk("results_both_reproductions_recover_the_published_test_error",
    near(_bt.released_weights_mae, 0.592, 6e-4)
    and near(_bt.retrained_mae, 0.595, 6e-4)
    and abs(_bt.released_weights_mae - _bt.published_mae) < 0.02
    and abs(_bt.retrained_mae - _bt.published_mae) < 0.02,
    f"released weights {_bt.released_weights_mae:.3f}, retrained "
    f"{_bt.retrained_mae:.3f} +- {_bt.retrained_mae_fold_sd:.3f} vs published "
    f"{_bt.published_mae:.3f}")
_o = json.load(open("results/canonical/cmi_split_ligand_overlap.json"))
chk("results_published_split_reuses_1568_of_1670_test_ligands",
    _o["n_ligands_shared_train_test"] == 1568 and _o["n_test_ligands"] == 1670
    and near(100 * _o["frac_test_ligands_seen_in_train"], 93.9, 0.05)
    and near(100 * _o["frac_test_rows_with_train_ligand"], 97.5, 0.05),
    f"{_o['n_ligands_shared_train_test']} of {_o['n_test_ligands']} ligands "
    f"({100 * _o['frac_test_ligands_seen_in_train']:.1f}%); "
    f"{100 * _o['frac_test_rows_with_train_ligand']:.1f}% of test rows")
chk("results_external_holdout_is_forty_unseen_ligands_at_0.834",
    _o["n_ext_ligands"] == 40 and near(_be.published_mae, 0.834, 1.1e-3)
    and near(_be.released_weights_mae, 0.834, 1.1e-3),
    f"{_o['n_ext_ligands']} ligands; published {_be.published_mae:.3f}, "
    f"released weights {_be.released_weights_mae:.3f}")
chk("results_ligand_disjoint_1.484_is_two_and_a_half_times_published",
    near(_bd.retrained_mae, 1.484, 6e-4)
    and near(_bd.retrained_mae_fold_sd, 0.161, 6e-4)
    and 2.4 <= _bd.retrained_mae / _bt.published_mae <= 2.6,
    f"{_bd.retrained_mae:.3f} +- {_bd.retrained_mae_fold_sd:.3f}, "
    f"{_bd.retrained_mae / _bt.published_mae:.2f}x the published 0.599")

# ------------- Y. composed versus direct selectivity modelling (Fig. 1b, S4.5, Fig. S4)
_S = pd.read_csv(T + "selectivity_construction.csv")
_ah = _S[_S.construction == "ad hoc, difference of two absolute predictions"]
_pt = _S[_S.construction == "ad hoc, point models without an interval"]
_di = _S[_S.construction == "direct, modelled on the pairwise target"]
chk("s3_5_scope_is_all_pairs_twelve_composed_three_point_eleven_direct",
    (_S.scope == "all_pairs").all() and len(_ah) == 12 and len(_pt) == 3
    and len(_di) == 11 and _S.is_workhorse.sum() == 1
    and _di[_di.is_workhorse].source_model_id.iloc[0] == "DS2_RDKIT_MLPENS5_HETERO",
    f"{len(_ah)} composed, {len(_pt)} point-only, {len(_di)} direct; workhorse is "
    f"the direct heteroscedastic MLP ensemble")
chk("s3_5_every_model_is_named_and_the_point_models_carry_no_interval",
    _S.source_model.notna().all() and not _S.source_model.str.startswith("DS").any()
    and _pt[["mop_picp_90", "mop_ace"]].isna().all().all()
    and _ah[["mop_picp_90", "mop_ace"]].notna().all().all()
    and _di[["mop_picp_90", "mop_ace"]].notna().all().all(),
    "all 26 rows carry a readable model name; only the point-only rows have "
    "empty calibration columns")
_ptb = _pt.sort_values("mop_rmse").iloc[0]
chk("results_best_point_only_ad_hoc_model_reaches_2.00",
    near(_ptb.mop_rmse, 2.002, 6e-4) and near(_ptb.mop_rmse_sd, 0.182, 6e-4)
    and _ptb.source_model == "RDKit MLP"
    and near(_pt.mop_rmse.max(), 2.109, 6e-4)
    and abs(_ptb.mop_rmse - float(_di[_di.is_workhorse].mop_rmse.iloc[0]))
    < float(_di[_di.is_workhorse].mop_rmse_sd.iloc[0]),
    f"{_ptb.source_model} {_ptb.mop_rmse:.3f} +- {_ptb.mop_rmse_sd:.3f}; the three "
    f"point models span {_pt.mop_rmse.min():.3f}-{_pt.mop_rmse.max():.3f}, and the "
    f"selected model is within its own fold s.d. of the best of them")
chk("results_point_accuracy_is_comparable_1.94_2.37_vs_1.93_2.24",
    near(_ah.mop_rmse.min(), 1.94, 5e-3) and near(_ah.mop_rmse.max(), 2.37, 5e-3)
    and near(_di.mop_rmse.min(), 1.93, 5e-3) and near(_di.mop_rmse.max(), 2.24, 5e-3),
    f"composed {_ah.mop_rmse.min():.3f}-{_ah.mop_rmse.max():.3f}, "
    f"direct {_di.mop_rmse.min():.3f}-{_di.mop_rmse.max():.3f}")
_ev = _ah[_ah.source_model_id.str.contains("EVID")]
chk("results_composed_intervals_are_the_ones_that_fail",
    near(_ah.mop_ace.min(), 0.200, 6e-4) and len(_ev) == 3
    and 0.37 <= _ev.mop_ace.min() and _ev.mop_ace.max() <= 0.381
    and 0.959 <= _ev.mop_picp_90.min() and _ev.mop_picp_90.max() <= 0.969,
    f"best composed ACE {_ah.mop_ace.min():.3f}; three evidential composed "
    f"{_ev.mop_ace.min():.3f}-{_ev.mop_ace.max():.3f} at PICP "
    f"{_ev.mop_picp_90.min():.3f}-{_ev.mop_picp_90.max():.3f} against a nominal 0.90")
_best_ah = float(_ah.mop_ace.min())
_gaps = (_di.mop_ace - _best_ah).sort_values().values
# Strictly lower, then the one that exceeds it by less than a thousandth -- well
# inside the fold s.d. of either model, so the text calls it a match -- then the
# one genuinely worse.
_better = int((_gaps < 0).sum())
_tie = float(_gaps[_better])
chk("results_nine_direct_models_beat_the_best_composed_one_a_tenth_matches_it",
    _better == 9 and len(_gaps) == 11 and 0 < _tie < 1e-3
    and _gaps[-1] > 1e-2
    and near(float(_di[_di.is_workhorse].mop_ace.iloc[0]), 0.153, 6e-4),
    f"{_better} of 11 direct models below the best composed ACE {_best_ah:.3f}; "
    f"the tenth exceeds it by {_tie:.5f} and the eleventh by {_gaps[-1]:.3f}; "
    f"selected model {float(_di[_di.is_workhorse].mop_ace.iloc[0]):.3f}")

# ------------- Z. internal disagreement in the record (Results opening, S2.6)
_D = pd.read_csv("manuscripts/final/tables/iz_duplicate_measurement_summary.csv")
_dm = _D[_D.subset == "DS1 metal-complex"].iloc[0]
chk("results_3048_groups_1234_disagree_368_by_more_than_one_unit",
    int(_dm.n_multi_keys) == 3048 and int(_dm.n_experiment_keys) == 27409
    and int(_dm.n_soft_disagreement) + int(_dm.n_hard_disagreement) == 1234
    and int(_dm.n_hard_disagreement) == 368,
    f"{int(_dm.n_multi_keys)} multi-record groups of {int(_dm.n_experiment_keys)}; "
    f"{int(_dm.n_soft_disagreement) + int(_dm.n_hard_disagreement)} disagree by more "
    f"than 0.20 and {int(_dm.n_hard_disagreement)} by more than one unit")
chk("s1_6_within_group_range_median_0.11_and_q90_1.16",
    near(_dm.within_group_range_median, 0.11, 6e-4)
    and near(_dm.within_group_range_q90, 1.163, 6e-4)
    and _dm.within_group_range_q90 > 1.0,
    f"median range {_dm.within_group_range_median:.2f}, 90th percentile "
    f"{_dm.within_group_range_q90:.3f}, which exceeds the one-unit selectivity threshold")

# ------------- AA. the interpretive acquisition claims in the rewritten Results
chk("results_one_wasted_first_experiment_in_three_becomes_one_in_six",
    round(1 / (1 - float(_HS.loc["random", "event_precision_at_1"]))) == 3
    and round(1 / (1 - float(_HS.loc["expected_improvement", "event_precision_at_1"]))) == 6,
    f"random wastes {1 - float(_HS.loc['random', 'event_precision_at_1']):.3f} of first "
    f"experiments, the margin rule {1 - float(_HS.loc['expected_improvement', 'event_precision_at_1']):.3f}")
chk("results_even_one_sigma_of_penalty_ranks_below_the_value_only_rule",
    float(_HS.loc["lcb_1sigma", "event_precision_at_1"])
    < float(_HS.loc["y_pred", "event_precision_at_1"])
    and list(_HS.loc[["lcb_1sigma", "lcb_2sigma", "lcb_3sigma"], "event_precision_at_1"])
    == sorted(_HS.loc[["lcb_1sigma", "lcb_2sigma", "lcb_3sigma"],
                      "event_precision_at_1"], reverse=True),
    f"1-sigma {float(_HS.loc['lcb_1sigma', 'event_precision_at_1']):.3f} below value-only "
    f"{float(_HS.loc['y_pred', 'event_precision_at_1']):.3f}, and the penalty series "
    f"decreases monotonically")
chk("results_two_heaviest_penalties_are_below_random_on_any_hit_at_10",
    near(_HS.loc["lcb_3sigma", "any_hit_at_10"], 0.821, 6e-4)
    and near(_HS.loc["lcb_2sigma", "any_hit_at_10"], 0.861, 6e-4)
    and near(_HS.loc["random", "any_hit_at_10"], 0.893, 6e-4)
    and _HS.loc["lcb_2sigma", "any_hit_at_10"] < _HS.loc["random", "any_hit_at_10"],
    f"3-sigma {_HS.loc['lcb_3sigma', 'any_hit_at_10']:.3f} and 2-sigma "
    f"{_HS.loc['lcb_2sigma', 'any_hit_at_10']:.3f} against random "
    f"{_HS.loc['random', 'any_hit_at_10']:.3f}")
chk("results_top_ten_holds_the_pool_optimum_three_times_in_four",
    near(_RHS.loc["expected_improvement", "pool_optimum"], 7.535, 6e-4)
    and 0.73 <= _RHS.loc["expected_improvement", "selected_contains_optimum_at_10"] <= 0.75
    and 0.73 <= _RHS.loc["y_pred", "selected_contains_optimum_at_10"] <= 0.75
    and _RHS.loc["random", "selected_contains_optimum_at_10"] < 0.5,
    f"pool optimum {_RHS.loc['expected_improvement', 'pool_optimum']:.3f}; margin "
    f"{_RHS.loc['expected_improvement', 'selected_contains_optimum_at_10']:.3f}, value-only "
    f"{_RHS.loc['y_pred', 'selected_contains_optimum_at_10']:.3f}, random "
    f"{_RHS.loc['random', 'selected_contains_optimum_at_10']:.3f}")

# ------------- AB. the event-threshold sensitivity restored to S9.4 (Fig. S10)
_TA = pd.read_csv("manuscripts/final/tables/pair_acquisition_tau2_half_hit_and_precision.csv")
_TR = pd.read_csv("manuscripts/final/tables/pair_acquisition_tau2_half_regret.csv")
_TE = pd.read_csv("manuscripts/final/tables/pair_acquisition_tau2_half_eligibility_summary.csv")
_THS = _TA[(_TA.task == "high_selectivity") & (_TA.orientation == "both")].set_index("strategy")
_TRHS = _TR[(_TR.task == "high_selectivity") & (_TR.orientation == "both")].set_index("strategy")
_TNZ = _TA[_TA.task == "low_selectivity"].set_index("strategy")
chk("s8_4_thresholds_are_2_and_one_half_with_739_and_346_tasks",
    (_TE.high_threshold == 2.0).all() and (_TE.low_threshold == 0.5).all()
    and near(_THS.n_tasks_fold_mean.iloc[0], 739.0, 0.06)
    and near(float(_THS.n_tasks_fold_sd.iloc[0]), 69.8, 0.06)
    and near(_TNZ.n_tasks_fold_mean.iloc[0], 346.4, 0.06)
    and _THS.n_tasks_fold_mean.iloc[0] < _HS.n_tasks_fold_mean.iloc[0]
    and _TNZ.n_tasks_fold_mean.iloc[0] < _NZ.n_tasks_fold_mean.iloc[0],
    f"{_THS.n_tasks_fold_mean.iloc[0]:.0f} +- {float(_THS.n_tasks_fold_sd.iloc[0]):.0f} high "
    f"and {_TNZ.n_tasks_fold_mean.iloc[0]:.1f} near-zero tasks, both fewer than at the "
    f"primary threshold")
_T841 = [("expected_improvement", 0.833, 0.988, 2.534),
         ("y_pred", 0.830, 0.959, 1.813),
         ("lcb_1sigma", 0.790, 0.906, 2.255),
         ("lcb_2sigma", 0.718, 0.855, 3.260),
         ("lcb_3sigma", 0.633, 0.805, 4.176),
         ("random", 0.600, 0.877, 4.657)]
_tb = [(s, round(float(_THS.loc[s, "event_precision_at_1"]), 3),
        round(float(_THS.loc[s, "any_hit_at_10"]), 3),
        round(float(_TRHS.loc[s, "regret_at_1"]), 3))
       for s, e, a, r in _T841
       if not (near(_THS.loc[s, "event_precision_at_1"], e, 6e-4)
               and near(_THS.loc[s, "any_hit_at_10"], a, 6e-4)
               and near(_TRHS.loc[s, "regret_at_1"], r, 6e-4))]
chk("s8_4_every_tabulated_row_reproduces", not _tb,
    f"mismatches {_tb}" if _tb else "all six rows match event precision@1, any hit@10 and regret@1")
chk("s8_4_every_main_text_ordering_survives_the_stricter_threshold",
    _THS.event_precision_at_1.idxmax() == "expected_improvement"
    and _THS.event_precision_at_5.idxmax() == "expected_improvement"
    and _THS.event_precision_at_10.idxmax() == "expected_improvement"
    and list(_THS.loc[["lcb_1sigma", "lcb_2sigma", "lcb_3sigma"], "event_precision_at_1"])
    == sorted(_THS.loc[["lcb_1sigma", "lcb_2sigma", "lcb_3sigma"],
                       "event_precision_at_1"], reverse=True)
    and _THS.loc["lcb_2sigma", "any_hit_at_10"] < _THS.loc["random", "any_hit_at_10"]
    and _THS.loc["lcb_3sigma", "any_hit_at_10"] < _THS.loc["random", "any_hit_at_10"]
    and _TRHS.regret_at_1.idxmin() == "y_pred"
    and _TNZ.loc["lcb_1sigma", "event_precision_at_1"]
    > _TNZ.loc["y_pred", "event_precision_at_1"],
    "margin best on precision at every k; penalties monotone; the two heaviest below "
    "random on any hit@10; regret won by the value-only rule; lower bounds lead near zero")
chk("s8_4_the_margin_over_random_widens_as_the_objective_tightens",
    near(_TRHS.loc["random", "pool_optimum"], 8.770, 6e-4)
    and _TRHS.loc["random", "pool_optimum"] > _RHS.loc["random", "pool_optimum"]
    and near(_THS.loc["random", "event_precision_at_1"], 0.600, 6e-4)
    and (_THS.loc["expected_improvement", "event_precision_at_1"]
         - _THS.loc["random", "event_precision_at_1"])
    > (_HS.loc["expected_improvement", "event_precision_at_1"]
       - _HS.loc["random", "event_precision_at_1"]),
    f"pool optimum {_TRHS.loc['random', 'pool_optimum']:.3f} against "
    f"{_RHS.loc['random', 'pool_optimum']:.3f}; margin over random "
    f"{_THS.loc['expected_improvement', 'event_precision_at_1'] - _THS.loc['random', 'event_precision_at_1']:.3f} "
    f"against {_HS.loc['expected_improvement', 'event_precision_at_1'] - _HS.loc['random', 'event_precision_at_1']:.3f}")

# ------------- AC. the measurement floor (Results opening, S2.7, S1.5)
_NF = pd.read_csv(T + "noise_floor.csv").set_index("estimate")
_inc = _NF.loc["inclusive, every repeated group"]
_trm = _NF.loc["trimmed, group range <= 1"]
chk("s1_7_inclusive_floor_0.496_rmse_0.241_mae_0.990_r2_ceiling",
    _inc.n_groups == 3048 and _inc.n_records == 7668
    and near(_inc.logk_rmse_floor, 0.496, 6e-4)
    and near(_inc.logk_mae_floor, 0.241, 6e-4)
    and near(_inc.logk_r2_ceiling, 0.9897, 6e-4),
    f"{int(_inc.n_groups)} groups, {int(_inc.n_records)} records: RMSE floor "
    f"{_inc.logk_rmse_floor:.3f}, MAE floor {_inc.logk_mae_floor:.3f}, R2 ceiling "
    f"{_inc.logk_r2_ceiling:.4f}")
chk("s1_7_trimmed_floor_0.190_rmse_0.113_mae_on_88_percent_of_groups",
    near(_trm.logk_rmse_floor, 0.190, 6e-4)
    and near(_trm.logk_mae_floor, 0.113, 6e-4)
    and near(_trm.logk_r2_ceiling, 0.9985, 6e-4)
    and 0.875 <= _trm.share_of_groups <= 0.885,
    f"{100 * _trm.share_of_groups:.1f}% of groups: RMSE floor "
    f"{_trm.logk_rmse_floor:.3f}, MAE floor {_trm.logk_mae_floor:.3f}, R2 ceiling "
    f"{_trm.logk_r2_ceiling:.4f}")
chk("s1_7_selectivity_floor_scales_by_1.369_to_0.680_and_0.260",
    near(_inc.pair_noise_factor, 1.369, 6e-4)
    and near(_inc.selectivity_rmse_floor, 0.680, 6e-4)
    and near(_trm.selectivity_rmse_floor, 0.260, 6e-4),
    f"factor {_inc.pair_noise_factor:.3f}; dlogK floors "
    f"{_inc.selectivity_rmse_floor:.3f} inclusive, {_trm.selectivity_rmse_floor:.3f} trimmed")
chk("s1_7_published_0.599_is_two_to_five_times_the_mae_floor",
    2.0 <= 0.599 / _inc.logk_mae_floor <= 2.6
    and 5.0 <= 0.599 / _trm.logk_mae_floor <= 5.5
    and 0.934 < _inc.logk_r2_ceiling,
    f"0.599 is {0.599 / _inc.logk_mae_floor:.1f}x the inclusive and "
    f"{0.599 / _trm.logk_mae_floor:.1f}x the trimmed MAE floor; the published R2 of "
    f"0.934 is below the ceiling {_inc.logk_r2_ceiling:.4f}")
chk("s1_7_ligand_disjoint_1.484_is_six_to_thirteen_times_the_floor",
    6.0 <= _bd.retrained_mae / _inc.logk_mae_floor <= 6.5
    and 12.5 <= _bd.retrained_mae / _trm.logk_mae_floor <= 13.5,
    f"{_bd.retrained_mae / _inc.logk_mae_floor:.1f}x inclusive and "
    f"{_bd.retrained_mae / _trm.logk_mae_floor:.1f}x trimmed")
chk("s1_7_half_retention_0.730_reaches_the_inclusive_selectivity_floor",
    2.9 <= float(_u.fold_mean) / _inc.selectivity_rmse_floor <= 3.2
    and 1.0 <= float(_ret.loc["sigma_total"].fold_mean) / _inc.selectivity_rmse_floor <= 1.10
    and 2.7 <= float(_ret.loc["sigma_total"].fold_mean) / _trm.selectivity_rmse_floor <= 2.9,
    f"unfiltered {float(_u.fold_mean):.3f} is "
    f"{float(_u.fold_mean) / _inc.selectivity_rmse_floor:.1f}x the inclusive floor; "
    f"retained {float(_ret.loc['sigma_total'].fold_mean):.3f} is "
    f"{float(_ret.loc['sigma_total'].fold_mean) / _inc.selectivity_rmse_floor:.2f}x it and "
    f"{float(_ret.loc['sigma_total'].fold_mean) / _trm.selectivity_rmse_floor:.1f}x the trimmed one")

# ------------- AD. mean-of-pairs against row-weighted aggregation (Results section 2)
_FM = pd.read_csv("results/cv_metrics/fold_metric_summary.csv")
_fw = _FM[(_FM.model_id == "DS2_RDKIT_MLPENS5_HETERO") & (_FM.scope == "all_pairs")]
_fm = lambda m: _fw[_fw.metric_name == m].iloc[0]
chk("results_row_weighted_r2_0.725_but_mean_of_pairs_r2_is_negative",
    near(_fm("r2").fold_mean, 0.725, 6e-4) and near(_fm("r2").fold_sd, 0.031, 6e-4)
    and _fm("mop_r2").fold_mean < 0,
    f"row-weighted R2 {_fm('r2').fold_mean:.3f} +- {_fm('r2').fold_sd:.3f}; "
    f"mean-of-pairs R2 {_fm('mop_r2').fold_mean:.2f} +- {_fm('mop_r2').fold_sd:.2f}, "
    f"below zero, so the model loses to each pair's own mean")

# ------------- AE. the uncertainty is not the predicted magnitude restated (S8.3)
def _rank(x): return pd.Series(x).rank().values
def _partial(a, b, c):
    a, b, c = _rank(a), _rank(b), _rank(c)
    ra = a - np.polyval(np.polyfit(c, a, 1), c)
    rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(ra, rb)[0, 1])
_sig, _mag, _par = [], [], []
for _f in FOLDS:
    _d = pd.read_csv(_f, usecols=["y_true", "y_pred", "sigma_total"])
    _e = (_d.y_true - _d.y_pred).abs().values
    _s, _m = _d.sigma_total.values, np.abs(_d.y_pred.values)
    _sig.append(float(pd.Series(_s).corr(pd.Series(_e), method="spearman")))
    _mag.append(float(pd.Series(_m).corr(pd.Series(_e), method="spearman")))
    _par.append(_partial(_s, _e, _m))
chk("s7_3_sigma_survives_controlling_for_predicted_magnitude_at_0.428",
    near(np.mean(_sig), 0.648, 6e-4) and near(np.mean(_mag), 0.541, 6e-4)
    and near(np.mean(_par), 0.428, 6e-4) and near(np.std(_par, ddof=1), 0.026, 6e-4)
    and np.mean(_par) > 0.5 * np.mean(_sig),
    f"rho(sigma) {np.mean(_sig):+.3f}, rho(|yhat|) {np.mean(_mag):+.3f}, partial "
    f"{np.mean(_par):+.3f} +- {np.std(_par, ddof=1):.3f}, more than half the raw value")

# ------------- AF. the criterion the published external set was built from (Results, S3.3)
_SP = pd.read_csv("manuscripts/final/tables/iz_risk_score_spearman.csv").set_index("score")
_cen = _SP.loc["mahalanobis_feature_centroid"]
# Distances only: not the model's own uncertainty, not the training-frequency count.
_dist = _SP[~_SP.index.str.startswith(("sigma", "pair_density", "yhat"))]
chk("results_mahalanobis_centroid_the_external_set_criterion_reaches_only_0.125",
    near(_cen.fold_mean, 0.125, 6e-4) and near(_cen.fold_sd, 0.076, 6e-4)
    and _cen.fold_mean < 2.0 * _cen.fold_sd
    and near(_cen.fold_mean / _dist.fold_mean.max(), 0.5, 0.02),
    f"Mahalanobis centroid {_cen.fold_mean:+.3f} +- {_cen.fold_sd:.3f}, "
    f"{_cen.fold_mean / _cen.fold_sd:.2f} fold s.d. from zero and "
    f"{_cen.fold_mean / _dist.fold_mean.max():.2f} of the best distance "
    f"({_dist.fold_mean.max():+.3f})")

n = sum(c["passed"] for c in checks)
json.dump(dict(**{"pass": n == len(checks)}, n_checks=len(checks), n_pass=n, checks=checks),
          open(T + "validate_manuscript_values.json", "w"), indent=2)
for c in checks:
    print(("OK  " if c["passed"] else "FAIL"), c["check"], "|", c["detail"])
print(f"\n{n}/{len(checks)} passed")
sys.exit(0 if n == len(checks) else 1)
