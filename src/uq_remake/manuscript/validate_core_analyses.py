"""Validate the derived-table analyses (tasks 1-10).

Re-derives headline numbers independently from the frozen fold scores /
processed datasets and asserts they agree with what the analysis modules
wrote to `manuscripts/derived/tables/*.csv`. Also checks the structural claims
from the task specs (`pair_density_risk` constant within every (pair, fold)
cell; no table presents a pooled-row statistic as a headline; coverage-grid
cell counts partition each fold exactly; the two coverage-override bootstrap
intervals actually exclude the overall mean; the Task 8 cell overlap is
reported and non-trivial; the Task 9 full ligand-disjoint reference
reproduces the frozen workhorse MoP RMSE; no Task 9 external set overlaps
its fold's training ligands; the Task 9 gap decomposition identity holds;
Task 10's per-fold pair row counts match the fold scores and every
acquisition metric is bounded in [0, 1]) and, for Task 1, that the
within-pair Spearman values reproduce the prototype figures in
`manuscripts/derived/analysis/chemical_sanity_checks.json` to +/-0.01.

Exits non-zero and prints every failing check on failure.

    python -m uq_remake.manuscript.validate_core_analyses
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uq_remake.downstream.protocol import FOLDS, PAIR_COLUMN
from uq_remake.manuscript.coverage_override_grid import _assign_quartiles, build_composition, build_grid
try:  # optional: only needed when the Task 9 table is present
    from uq_remake.manuscript.external_set_emulation import (
        WORKHORSE_EXPECTED_MOP_RMSE, _load_fold_geometry, _pca_mahalanobis,
        _score_rows, _shell_window)
except ModuleNotFoundError:
    WORKHORSE_EXPECTED_MOP_RMSE = None
from uq_remake.manuscript.manuscript_common import (
    DS2_PATH,
    DERIVED_TABLES,
    PROTOTYPE_JSON,
    assert_pair_density_constant_within_cells,
    concat_folds,
    join_ds2_chemical_axes,
    load_all_folds,
    load_ds2_chemical_axes,
    with_abs_residual,
)
from uq_remake.manuscript.scale_free_error import add_pair_pooled_sdp
from uq_remake.manuscript.sigma_chemical_axes import RESPONSES, analysis_6_1, analysis_6_2, analysis_6_3
try:  # optional: only needed when the Task 10 table is present
    from uq_remake.manuscript.single_pair_campaign import PAIRS_IN_FREQUENCY_ORDER
except ModuleNotFoundError:
    PAIRS_IN_FREQUENCY_ORDER = ()
from uq_remake.manuscript.within_pair_ordering import _pair_pooled_rho

TOL_3DP = 5e-4
TOL_PROTOTYPE = 0.01


def add(rows, name, ok, detail=""):
    rows.append({"check": name, "pass": bool(ok), "detail": str(detail)})


def _close(a, b, tol) -> bool:
    if a is None or b is None:
        return False
    try:
        return bool(np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= tol)
    except (TypeError, ValueError):
        return False


def check_pair_density_constant(rows, fold_frames) -> None:
    try:
        result = assert_pair_density_constant_within_cells(fold_frames)
        add(rows, "pair_density_risk_constant_in_all_cells", result["n_violations"] == 0, result)
    except AssertionError as exc:
        add(rows, "pair_density_risk_constant_in_all_cells", False, str(exc))


def check_within_pair_ordering(rows, fold_frames) -> None:
    path = DERIVED_TABLES / "within_pair_score_residual.csv"
    if not path.exists():
        add(rows, "within_pair_score_residual_exists", False, str(path))
        return
    add(rows, "within_pair_score_residual_exists", True)
    table = pd.read_csv(path)
    all_df = with_abs_residual(concat_folds(fold_frames))
    prototype = json.loads(PROTOTYPE_JSON.read_text(encoding="utf-8")) if PROTOTYPE_JSON.exists() else {}
    wp = prototype.get("within_pair", {})

    targets = {
        "sigma_total": {"mean_rho": wp.get("sigma_mean_rho"), "median_rho": wp.get("sigma_median_rho"), "frac_positive": wp.get("sigma_frac_positive")},
        "mahalanobis_feature_centroid": {"mean_rho": wp.get("mahalanobis_mean_rho"), "frac_positive": wp.get("mahalanobis_frac_positive")},
        "tanimoto_nn_distance": {"mean_rho": wp.get("tanimoto_nn_mean_rho"), "frac_positive": wp.get("tanimoto_nn_frac_positive")},
    }
    for score, expected in targets.items():
        recomputed = _pair_pooled_rho(all_df, score)
        ok = recomputed.loc[recomputed.status == "ok", "rho"]
        computed = {
            "mean_rho": float(ok.mean()) if len(ok) else float("nan"),
            "median_rho": float(ok.median()) if len(ok) else float("nan"),
            "frac_positive": float((ok > 0).mean()) if len(ok) else float("nan"),
            "n_pairs_eligible": int(len(ok)),
        }
        row = table[(table.score_name == score) & (table.aggregation == "distribution_over_pairs")]
        if len(row) != 1:
            add(rows, f"within_pair_row_present:{score}", False, f"n_rows={len(row)}")
            continue
        r = row.iloc[0]
        add(rows, f"within_pair_n_pairs_matches:{score}", int(r["n_pairs_eligible"]) == computed["n_pairs_eligible"], f"csv={r['n_pairs_eligible']} recomputed={computed['n_pairs_eligible']}")
        add(rows, f"within_pair_mean_rho_matches_csv:{score}", _close(r["mean_rho"], computed["mean_rho"], TOL_3DP), f"csv={r['mean_rho']} recomputed={computed['mean_rho']}")
        for metric, expected_val in expected.items():
            if expected_val is None:
                continue
            add(
                rows,
                f"within_pair_{metric}_matches_prototype:{score}",
                _close(computed[metric], expected_val, TOL_PROTOTYPE),
                f"prototype={expected_val} recomputed={computed[metric]}",
            )

    pdr = table[(table.score_name == "pair_density_risk") & (table.aggregation == "distribution_over_pairs")]
    if len(pdr) == 1:
        add(rows, "pair_density_risk_reported_undefined_not_zero", int(pdr.iloc[0]["n_pairs_eligible"]) == 0, str(pdr.iloc[0]["n_pairs_eligible"]))
    else:
        add(rows, "pair_density_risk_row_present", False, f"n_rows={len(pdr)}")


def check_scale_free_error(rows, fold_frames) -> None:
    path = DERIVED_TABLES / "scale_free_error.csv"
    if not path.exists():
        add(rows, "scale_free_error_exists", False, str(path))
        return
    add(rows, "scale_free_error_exists", True)
    table = pd.read_csv(path)
    all_df = concat_folds(fold_frames)
    used, _diag = add_pair_pooled_sdp(all_df)
    ree = used[used["is_ree_pair"] == True]  # noqa: E712
    nonree = used[used["is_ree_pair"] == False]  # noqa: E712
    raw_ratio = float(nonree["_abs_residual"].mean() / ree["_abs_residual"].mean())
    sf_ratio = float(nonree["e_norm"].mean() / ree["e_norm"].mean())

    prototype = json.loads(PROTOTYPE_JSON.read_text(encoding="utf-8")) if PROTOTYPE_JSON.exists() else {}
    sf = prototype.get("scale_free", {})
    add(
        rows,
        "raw_abs_error_ratio_matches_prototype",
        _close(raw_ratio, sf.get("raw_abs_err_ratio_nonree_over_ree"), 0.01),
        f"prototype={sf.get('raw_abs_err_ratio_nonree_over_ree')} recomputed={raw_ratio}",
    )
    add(
        rows,
        "scale_free_error_ratio_matches_prototype",
        _close(sf_ratio, sf.get("scale_free_err_ratio_nonree_over_ree"), 0.01),
        f"prototype={sf.get('scale_free_err_ratio_nonree_over_ree')} recomputed={sf_ratio}",
    )

    csv_raw = table[(table.metric_name == "mean_abs_error_raw") & (table.group == "ratio_nonree_over_ree")]
    csv_sf = table[(table.metric_name == "mean_e_norm_scale_free") & (table.group == "ratio_nonree_over_ree")]
    if len(csv_raw) == 1:
        add(rows, "scale_free_csv_raw_ratio_matches_recompute", _close(float(csv_raw.iloc[0]["value"]), raw_ratio, TOL_3DP))
    if len(csv_sf) == 1:
        add(rows, "scale_free_csv_sf_ratio_matches_recompute", _close(float(csv_sf.iloc[0]["value"]), sf_ratio, TOL_3DP))

    sigma = pd.to_numeric(used["sigma_total"], errors="coerce")
    e_norm = used["e_norm"]
    rho1, _ = spearmanr(sigma, e_norm)
    rho2, _ = spearmanr(sigma / used["_sd_p"], e_norm)
    corr_rows = table[table.table_section == "pooled_correlations"]
    csv_rho1 = corr_rows[corr_rows.metric_name == "pooled_rho_sigma_total_vs_e_norm"]
    csv_rho2 = corr_rows[corr_rows.metric_name == "pooled_rho_sigma_over_sdp_vs_e_norm"]
    if len(csv_rho1) == 1:
        add(rows, "pooled_rho_sigma_vs_enorm_matches_prototype", _close(float(csv_rho1.iloc[0]["value"]), 0.271, TOL_PROTOTYPE), f"csv={csv_rho1.iloc[0]['value']}")
    if len(csv_rho2) == 1:
        add(rows, "pooled_rho_sigma_over_sd_vs_enorm_matches_prototype", _close(float(csv_rho2.iloc[0]["value"]), 0.410, TOL_PROTOTYPE), f"csv={csv_rho2.iloc[0]['value']}")
    add(
        rows,
        "pooled_correlation_rows_tagged_si_diagnostic",
        bool((corr_rows["role"] == "si_diagnostic_only").all()) if "role" in corr_rows.columns and len(corr_rows) else False,
    )


def check_logk_censoring(rows) -> None:
    path = DERIVED_TABLES / "logk_censoring.csv"
    if not path.exists():
        add(rows, "logk_censoring_exists", False, str(path))
        return
    add(rows, "logk_censoring_exists", True)
    table = pd.read_csv(path)
    ds2 = pd.read_csv(DS2_PATH, usecols=["logK_x", "logK_y"])
    pooled = pd.concat([ds2["logK_x"], ds2["logK_y"]], ignore_index=True).dropna()
    computed = {
        "frac_le_0": float((pooled <= 0).mean()),
        "frac_le_1": float((pooled <= 1).mean()),
        "frac_le_2": float((pooled <= 2).mean()),
        "median": float(pooled.median()),
        "minimum": float(pooled.min()),
    }
    row = table[(table.dataset == "DS2_endpoints_pooled") & (table.metal_block == "all")]
    if len(row) != 1:
        add(rows, "logk_censoring_ds2_row_present", False, f"n_rows={len(row)}")
        return
    r = row.iloc[0]
    for metric, val in computed.items():
        add(rows, f"logk_censoring_ds2_{metric}_matches_recompute", _close(float(r[metric]), val, TOL_3DP), f"csv={r[metric]} recomputed={val}")
    prototype = json.loads(PROTOTYPE_JSON.read_text(encoding="utf-8")) if PROTOTYPE_JSON.exists() else {}
    cs = prototype.get("censoring", {})
    add(rows, "logk_censoring_frac_le0_matches_prototype", _close(computed["frac_le_0"], cs.get("frac_endpoint_logK_le0"), TOL_PROTOTYPE))
    add(rows, "logk_censoring_frac_le1_matches_prototype", _close(computed["frac_le_1"], cs.get("frac_le1"), TOL_PROTOTYPE))
    add(rows, "logk_censoring_frac_le2_matches_prototype", _close(computed["frac_le_2"], cs.get("frac_le2"), TOL_PROTOTYPE))
    add(rows, "logk_censoring_median_matches_prototype", _close(computed["median"], cs.get("median_logK"), 0.1))


def check_no_pooled_headline(rows) -> None:
    ret_path = DERIVED_TABLES / "within_pair_retention.csv"
    reg_path = DERIVED_TABLES / "uncertainty_by_regime.csv"
    for path in (ret_path, reg_path):
        if not path.exists():
            add(rows, f"fold_aggregated_present:{path.name}", False, str(path))
            continue
        table = pd.read_csv(path)
        has_fold_cols = {"fold_sd", "n_valid_folds"}.issubset(table.columns)
        add(rows, f"fold_aggregated_columns_present:{path.name}", has_fold_cols)
        if has_fold_cols:
            n5 = int((pd.to_numeric(table.n_valid_folds, errors="coerce") == 5).sum())
            add(rows, f"fold_aggregated_mostly_5_folds:{path.name}", n5 >= 0.9 * len(table), f"n_valid_folds==5 for {n5}/{len(table)} rows")

    sf_path = DERIVED_TABLES / "scale_free_error.csv"
    if sf_path.exists():
        table = pd.read_csv(sf_path)
        pooled_rows = table[table.table_section.isin(["ree_comparison", "pooled_correlations"])]
        headline_rows = table[~table.table_section.isin(["ree_comparison", "pooled_correlations"])]
        add(
            rows,
            "scale_free_pooled_rows_confined_to_labeled_sections",
            bool(pooled_rows["table_section"].isin(["ree_comparison", "pooled_correlations"]).all()),
        )
        if "n_valid_folds" in headline_rows.columns:
            add(rows, "scale_free_headline_retention_is_fold_aggregated", bool(headline_rows["n_valid_folds"].notna().all()))


def _joined_fold_frames(fold_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ds2_derived, _diag = load_ds2_chemical_axes()
    return {fold: join_ds2_chemical_axes(with_abs_residual(df), ds2_derived) for fold, df in fold_frames.items()}


def check_sigma_chemical_axes(rows, fold_frames) -> None:
    path = DERIVED_TABLES / "sigma_chemical_axes.csv"
    if not path.exists():
        add(rows, "sigma_chemical_axes_exists", False, str(path))
        return
    add(rows, "sigma_chemical_axes_exists", True)
    table = pd.read_csv(path)
    joined = _joined_fold_frames(fold_frames)

    recomputed_rows: list[dict] = []
    for i, response in enumerate(RESPONSES):
        recomputed_rows.extend(analysis_6_1(joined, response))
        recomputed_rows.extend(analysis_6_2(joined, response))
        recomputed_rows.extend(analysis_6_3(joined, response, with_vif=(i == 0)))
    recomputed = pd.DataFrame(recomputed_rows)

    n_checked = 0
    n_mismatched = 0
    for _, rrow in recomputed.iterrows():
        key_cols = {"analysis": rrow["analysis"], "response": rrow["response"], "metric_name": rrow["metric_name"]}
        if "unit" in rrow and pd.notna(rrow.get("unit")):
            key_cols["unit"] = rrow["unit"]
        mask = pd.Series(True, index=table.index)
        for k, v in key_cols.items():
            mask &= table[k] == v
        match = table[mask]
        if len(match) != 1:
            add(rows, f"sigma_chemical_axes_row_present:{key_cols}", False, f"n_rows={len(match)}")
            continue
        csv_row = match.iloc[0]
        value_col = "mean_rho" if "mean_rho" in rrow and pd.notna(rrow.get("mean_rho")) else "fold_mean"
        n_checked += 1
        if not _close(csv_row[value_col], rrow[value_col], TOL_3DP):
            n_mismatched += 1
    add(rows, "sigma_chemical_axes_all_coefficients_reproduce_to_3dp", n_mismatched == 0, f"checked={n_checked} mismatched={n_mismatched}")

    # The claim Task 6.1 needs: |rho(response, favoured_logK | support)| > |rho(response, support | favoured_logK)|.
    for response in RESPONSES:
        a = table[(table.analysis == "6.1_partial_corr_support_controlled") & (table.response == response) & (table.metric_name == "rho_response_favouredlogK_given_support")]
        b = table[(table.analysis == "6.1_partial_corr_support_controlled") & (table.response == response) & (table.metric_name == "rho_response_support_given_favouredlogK")]
        if len(a) == 1 and len(b) == 1:
            add(
                rows,
                f"favouredlogK_partial_corr_exceeds_support_partial_corr:{response}",
                abs(float(a.iloc[0]["fold_mean"])) > abs(float(b.iloc[0]["fold_mean"])),
                f"favouredlogK={a.iloc[0]['fold_mean']} support={b.iloc[0]['fold_mean']}",
            )

    # The over-weighting caveat: does sigma track favoured_logK more strongly than |error| does?
    sig = table[(table.analysis == "6.1_partial_corr_support_controlled") & (table.response == "sigma_total") & (table.metric_name == "rho_response_favouredlogK_given_support")]
    err = table[(table.analysis == "6.1_partial_corr_support_controlled") & (table.response == "abs_error") & (table.metric_name == "rho_response_favouredlogK_given_support")]
    if len(sig) == 1 and len(err) == 1:
        sigma_stronger = abs(float(sig.iloc[0]["fold_mean"])) > abs(float(err.iloc[0]["fold_mean"]))
        add(
            rows,
            "sigma_overweights_favouredlogK_relative_to_error_CAVEAT",
            True,  # informational: reports the finding, does not gate pass/fail either way
            f"sigma_rho={sig.iloc[0]['fold_mean']} error_rho={err.iloc[0]['fold_mean']} sigma_tracks_favouredlogK_more_strongly={sigma_stronger}",
        )


def check_coverage_override_grid(rows, fold_frames) -> None:
    grid_path = DERIVED_TABLES / "coverage_override_grid.csv"
    fold_level_path = DERIVED_TABLES / "coverage_override_grid_fold_level.csv"
    if not (grid_path.exists() and fold_level_path.exists()):
        add(rows, "coverage_override_grid_exists", False, f"{grid_path} / {fold_level_path}")
        return
    add(rows, "coverage_override_grid_exists", True)

    joined = _joined_fold_frames(fold_frames)
    quartiled = {fold: _assign_quartiles(df) for fold, df in joined.items()}

    fold_level = pd.read_csv(fold_level_path)
    n_sum_mismatches = []
    for fold in FOLDS:
        expected_n = int(len(quartiled[fold]))
        actual_n = int(pd.to_numeric(fold_level[fold_level.outer_fold_id == fold]["n_rows"], errors="coerce").sum())
        if actual_n != expected_n:
            n_sum_mismatches.append({"fold": fold, "expected": expected_n, "actual": actual_n})
    add(rows, "coverage_grid_cell_counts_sum_to_fold_row_count", len(n_sum_mismatches) == 0, n_sum_mismatches)

    grid_recomputed, _fold_level_recomputed = build_grid(quartiled)
    grid_csv = pd.read_csv(grid_path)
    n_checked, n_mismatched = 0, 0
    for _, rrow in grid_recomputed[grid_recomputed.table_section.isin(["grid", "support_marginal", "sigma_marginal", "overall"])].iterrows():
        mask = (
            (grid_csv.table_section == rrow["table_section"])
            & (grid_csv.aggregation == rrow["aggregation"])
            & (grid_csv.support_quartile.astype(str) == str(rrow["support_quartile"]))
            & (grid_csv.sigma_quartile.astype(str) == str(rrow["sigma_quartile"]))
        )
        match = grid_csv[mask]
        if len(match) != 1:
            continue
        n_checked += 1
        if not _close(float(match.iloc[0]["mean_abs_error"]), float(rrow["mean_abs_error"]), TOL_3DP):
            n_mismatched += 1
    add(rows, "coverage_grid_all_cells_reproduce_to_3dp", n_mismatched == 0, f"checked={n_checked} mismatched={n_mismatched}")

    boot_rows = grid_recomputed[grid_recomputed.table_section == "bootstrap_ci"]
    for _, brow in boot_rows.iterrows():
        name = brow["cell_name"]
        ci_low, ci_high = float(brow["ci_low_95"]), float(brow["ci_high_95"])
        overall = float(brow["overall_mean_abs_error_pooled"])
        excludes = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and not (ci_low <= overall <= ci_high))
        add(
            rows,
            f"bootstrap_ci_excludes_overall_mean:{name}",
            excludes,
            f"ci=[{ci_low:.4f},{ci_high:.4f}] overall_mean={overall:.4f} " + ("EXCLUDED (supported)" if excludes else "NOT EXCLUDED -- central interpretability claim is NOT statistically supported for this cell"),
        )
        csv_boot = grid_csv[(grid_csv.table_section == "bootstrap_ci") & (grid_csv.cell_name == name)]
        if len(csv_boot) == 1:
            add(
                rows,
                f"bootstrap_ci_reproduces_stored_csv:{name}",
                _close(float(csv_boot.iloc[0]["ci_low_95"]), ci_low, 1e-6) and _close(float(csv_boot.iloc[0]["ci_high_95"]), ci_high, 1e-6),
                f"csv=[{csv_boot.iloc[0]['ci_low_95']},{csv_boot.iloc[0]['ci_high_95']}] recomputed=[{ci_low},{ci_high}]",
            )


def check_flagged_region_composition(rows, fold_frames) -> None:
    path = DERIVED_TABLES / "flagged_region_composition.csv"
    if not path.exists():
        add(rows, "flagged_region_composition_exists", False, str(path))
        return
    add(rows, "flagged_region_composition_exists", True)
    table = pd.read_csv(path)

    joined = _joined_fold_frames(fold_frames)
    quartiled = {fold: _assign_quartiles(df) for fold, df in joined.items()}
    all_df = concat_folds(quartiled)
    recomputed = build_composition(all_df)

    overlap_csv = table[table.row_type == "overlap"]
    overlap_recomputed = recomputed[recomputed.row_type == "overlap"]
    n_overlap_row = overlap_csv[overlap_csv.key == "n_pairs_overlap"]
    add(rows, "task8_overlap_reported", len(n_overlap_row) == 1, f"n_rows={len(n_overlap_row)}")
    if len(n_overlap_row) == 1:
        n_overlap = int(n_overlap_row.iloc[0]["n_rows"])
        frac_rows = overlap_csv[overlap_csv.key == "fraction_of_cell_rows_from_overlap_pairs"]
        min_frac = float(pd.to_numeric(frac_rows["share"], errors="coerce").min()) if len(frac_rows) else float("nan")
        NON_TRIVIAL_PAIR_COUNT = 1
        NON_TRIVIAL_ROW_FRACTION = 0.05
        non_trivial = (n_overlap >= NON_TRIVIAL_PAIR_COUNT) and np.isfinite(min_frac) and (min_frac >= NON_TRIVIAL_ROW_FRACTION)
        add(
            rows,
            "task8_overlap_is_non_trivial",
            non_trivial,
            f"n_pairs_overlap={n_overlap} min_fraction_of_cell_rows_from_overlap_pairs={min_frac:.4f} "
            + ("(supports 'sigma partitions within pairs')" if non_trivial else "(claim NOT supported -- cells barely share pairs, must be retracted)"),
        )

    n_checked, n_mismatched = 0, 0
    for _, rrow in recomputed[recomputed.row_type == "overlap"].iterrows():
        mask = (overlap_csv.cell == rrow["cell"]) & (overlap_csv.key == rrow["key"])
        match = overlap_csv[mask]
        if len(match) != 1:
            continue
        n_checked += 1
        csv_n = match.iloc[0].get("n_rows")
        if pd.notna(csv_n) and pd.notna(rrow.get("n_rows")) and int(csv_n) != int(rrow["n_rows"]):
            n_mismatched += 1
    add(rows, "task8_overlap_counts_reproduce", n_mismatched == 0, f"checked={n_checked} mismatched={n_mismatched}")


def check_external_set_emulation(rows, fold_frames) -> None:
    path = DERIVED_TABLES / "external_set_emulation.csv"
    if not path.exists():
        # Task 9 was not carried into the submitted manuscript; its table is absent
        # by design. Skip rather than fail so the validator reflects the paper as
        # submitted. See Supplementary S8.3.
        return
        return
    add(rows, "external_set_emulation_exists", True)
    table = pd.read_csv(path)

    # Full ligand-disjoint reference reproduces the frozen workhorse MoP RMSE.
    from uq_remake.metrics.cv_protocol import fold_mean_sd
    from uq_remake.metrics.fold_metrics import mop_block

    joined = {fold: with_abs_residual(df) for fold, df in fold_frames.items()}
    full_vals = [mop_block(joined[f])["mop_rmse"] for f in FOLDS]
    full_stats = fold_mean_sd(full_vals)
    add(
        rows,
        "task9_full_ligand_disjoint_reproduces_frozen_2078_130",
        _close(full_stats["fold_mean"], WORKHORSE_EXPECTED_MOP_RMSE[0], TOL_3DP) and _close(full_stats["fold_sd"], WORKHORSE_EXPECTED_MOP_RMSE[1], TOL_3DP),
        f"recomputed={full_stats['fold_mean']:.4f}+/-{full_stats['fold_sd']:.4f} expected={WORKHORSE_EXPECTED_MOP_RMSE}",
    )
    csv_full = table[(table.table_section == "full_reference")]
    if len(csv_full) == 1:
        add(rows, "task9_full_reference_csv_matches_recompute", _close(float(csv_full.iloc[0]["fold_mean"]), full_stats["fold_mean"], TOL_3DP))

    # Re-derive geometry + one selection per (fold, N) fresh; compare row/pair counts
    # and re-check overlap with that fold's training ligands and the decomposition identity.
    fold_level = table[table.table_section == "external_set_fold_level"]
    n_row_mismatches = 0
    n_overlap_violations = 0
    n_checked = 0
    for i, fold in enumerate(FOLDS):
        geo = _load_fold_geometry(fold)
        pm = _pca_mahalanobis(geo["X_ligand"])
        order = np.argsort(pm["distances"], kind="mergesort")
        sorted_ligands = geo["ligand_ids"][order]
        sorted_dist = pm["distances"][order]
        df = joined[fold]
        for N in (20, 30, 40):
            if len(sorted_ligands) < N:
                continue
            start, end = _shell_window(sorted_dist, N)
            shell_ligands = set(sorted_ligands[start:end])
            tail_ligands = set(sorted_ligands[-N:])
            if shell_ligands & geo["train_ligands"] or tail_ligands & geo["train_ligands"]:
                n_overlap_violations += 1
            shell_score = _score_rows(df[df["ligand_smiles"].isin(shell_ligands)])
            csv_row = fold_level[(fold_level.outer_fold_id == fold) & (fold_level.N == N) & (fold_level.kind == "shell")]
            n_checked += 1
            if len(csv_row) == 1 and int(csv_row.iloc[0]["n_pairs_covered"]) != shell_score["n_pairs_covered"]:
                n_row_mismatches += 1
    add(rows, "task9_no_external_set_overlaps_training_ligands", n_overlap_violations == 0, f"violations={n_overlap_violations}")
    add(rows, "task9_shell_pair_counts_reproduce", n_row_mismatches == 0, f"checked={n_checked} mismatched={n_row_mismatches}")

    decomp = table[table.table_section == "decomposition"]
    if len(decomp):
        identity_col = decomp["identity_holds"].astype(str)
        add(rows, "task9_decomposition_identity_holds_all_rows", bool((identity_col == "True").all()), f"n_rows={len(decomp)} n_true={(identity_col == 'True').sum()}")
        recheck = (
            decomp["gap_total_vs_full"] - (decomp["gap_size_component_vs_random"] + decomp["gap_selection_component_vs_random"])
        ).abs()
        add(rows, "task9_decomposition_identity_recomputed", bool((recheck <= 1e-6).all()), f"max_abs_residual={float(recheck.max()) if len(recheck) else float('nan')}")

    headline = table[table.table_section.isin(["external_set_summary", "full_reference", "decomposition"])]
    if "n_valid_folds" in headline.columns:
        nvf = pd.to_numeric(headline["n_valid_folds"], errors="coerce").dropna()
        add(rows, "task9_headline_rows_are_fold_aggregated", bool((nvf == 5).all()) if len(nvf) else False, f"n_valid_folds values: {sorted(nvf.unique().tolist())}")


def check_single_pair_campaign(rows, fold_frames) -> None:
    path = DERIVED_TABLES / "single_pair_campaign.csv"
    if not path.exists():
        # Task 10 was not carried into the submitted manuscript (Supplementary S8.3).
        return
        return
    add(rows, "single_pair_campaign_exists", True)
    table = pd.read_csv(path)

    fold_level = table[table.table_section == "campaign_setup_fold_level"]
    n_checked, n_mismatched = 0, 0
    for pair in PAIRS_IN_FREQUENCY_ORDER:
        for fold, df in fold_frames.items():
            expected_n = int((df[PAIR_COLUMN] == pair).sum())
            row = fold_level[(fold_level.pair == pair) & (fold_level.outer_fold_id == fold)]
            n_checked += 1
            if len(row) != 1 or int(row.iloc[0]["n_rows"]) != expected_n:
                n_mismatched += 1
    add(rows, "task10_per_fold_pair_counts_match_fold_scores", n_mismatched == 0, f"checked={n_checked} mismatched={n_mismatched}")

    topk = table[table.table_section == "topk_metrics"]
    bounded_metrics = topk[topk.metric_name.astype(str).str.startswith(("event_precision_at_", "hit_at_"))]
    vals = pd.to_numeric(bounded_metrics["fold_mean"], errors="coerce").dropna()
    out_of_bounds = vals[(vals < -1e-9) | (vals > 1.0 + 1e-9)]
    add(rows, "task10_acquisition_metrics_bounded_0_1", len(out_of_bounds) == 0, f"n_checked={len(vals)} n_out_of_bounds={len(out_of_bounds)}")

    # Fold-aggregated, not a pooled-row headline. n_valid_folds < 5 is expected and
    # legitimate here: some (pair, task, orientation) combinations only clear the
    # >1-event eligibility bar (pair_acquisition.eligible_tasks) in a subset of
    # folds -- that is fold-respecting aggregation over fewer folds, not pooling.
    headline = table[table.table_section.isin(["campaign_setup", "topk_metrics", "retention_curve"])]
    if "n_valid_folds" in headline.columns:
        nvf = pd.to_numeric(headline["n_valid_folds"], errors="coerce").dropna()
        in_range = nvf.between(1, 5)
        add(rows, "task10_headline_rows_are_fold_aggregated", bool(in_range.all()) if len(nvf) else False, f"n_valid_folds range: {sorted(nvf.unique().tolist())}")

    comp = table[table.table_section == "cross_pair_comparison"]
    add(rows, "task10_cross_pair_comparison_reported", len(comp) == len(PAIRS_IN_FREQUENCY_ORDER), f"n_rows={len(comp)}")


def main() -> dict:
    rows: list[dict] = []
    fold_frames = load_all_folds()
    check_pair_density_constant(rows, fold_frames)
    check_within_pair_ordering(rows, fold_frames)
    check_scale_free_error(rows, fold_frames)
    check_logk_censoring(rows)
    check_no_pooled_headline(rows)
    check_sigma_chemical_axes(rows, fold_frames)
    check_coverage_override_grid(rows, fold_frames)
    check_flagged_region_composition(rows, fold_frames)
    check_external_set_emulation(rows, fold_frames)
    check_single_pair_campaign(rows, fold_frames)

    report = {
        "pass": all(r["pass"] for r in rows),
        "n_checks": len(rows),
        "n_pass": int(sum(r["pass"] for r in rows)),
        "checks": rows,
    }
    out_path = DERIVED_TABLES / "validate_core_analyses.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps({k: rep[k] for k in ("pass", "n_checks", "n_pass")}, indent=2))
    if not rep["pass"]:
        for row in rep["checks"]:
            if not row["pass"]:
                print(row)
        raise SystemExit(1)
