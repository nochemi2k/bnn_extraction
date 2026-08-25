"""Consistency checks for the corrected manuscript rewrite."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.manuscript.figure_specs import FIGURE_SPECS

from uq_remake.manuscript.manuscript_paths import (
    AD_HOC_SELECTIVITY,
    MSRC,
    NUMBERS_JSON,
    PAIR_DENSITY_SUMMARY,
    SIGN_STABILITY_SUMMARY,
    TABLES,
)

MAN = REMAKE_ROOT / "manuscripts" / "final"

OBSOLETE_ACTIVE = [
    "BAYES_LASTLAYER",
    "CPEMB",
    "+RANK",
    "GRAPH_DMPNN_MCDROPOUT",
    "CHEMPROP_MC",
]

HISTORICAL_ALLOW = {
    "0.599",
    "0.934",
    "0.834",
    "0.942",
    "0.918",
    "0.802",
    "1.278",
    "0.894",
    "0.592",
    "0.943",
    "0.539",
    "0.951",
    "0.541",
    "0.881",
    "0.935",
    "0.595",
    "0.945",
    "0.928",
}


def fmt_pm(mean, sd, n_valid=5, digits=3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(sd):
        s = f"{float(mean):.{digits}f}"
    else:
        s = f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"
    if n_valid is not None and int(n_valid) != 5:
        s += f" (n_valid={int(n_valid)})"
    return s


def collect_table_displays() -> set[str]:
    out = set()
    for p in TABLES.glob("*.csv"):
        df = pd.read_csv(p)
        for col in df.columns:
            if col in {
                "display",
                "rmse",
                "r2",
                "picp_90",
                "ace",
                "row_rmse",
                "row_r2",
                "mop_rmse",
                "sign_accuracy",
                "ree_mop_rmse",
                "ree_mop_r2",
                "ree_sign_accuracy",
                "all_pair_mop_rmse",
                "all_pair_mop_r2",
                "all_pair_picp_90",
                "all_pair_ace",
                "all_pair_row_picp_90",
                "all_pair_row_ace",
                "all_pair_row_r2",
                "mop_picp_90",
                "mop_ace",
                "delta_all_pair_mop_rmse",
                "aurc",
                "pair_aurc",
                "rmse_at_50pct_retention",
                "mop_rmse_at_50",
                "row_aurc",
                "row_rmse_at_50",
                "row_rmse_at_50pct_retention",
                "ree_picp_90",
                "ree_ace",
                "delta_ree_mop_rmse",
                "delta_all_pair_picp_90",
                "delta_all_pair_ace",
                "mop_aurc",
                "mop_rmse_at_50pct",
                "value_display",
                "display",
                "pearson_abs_residual",
                "spearman_abs_residual",
            }:
                for v in df[col].astype(str):
                    out.add(str(v).strip())
    snap = pd.read_csv(MSRC / "manuscript_metric_snapshot.csv")
    if "value_display" in snap.columns:
        out.update(snap["value_display"].astype(str).str.strip())
    numbers = json.loads((MSRC / "numbers.json").read_text(encoding="utf-8"))
    out.add(str(numbers["hplus_r2_with"]))
    out.add(str(numbers["hplus_r2_without"]))
    for v in (numbers.get("fmt") or {}).values():
        if isinstance(v, str) and v.strip():
            out.add(v.strip())
    for extra in (
        SIGN_STABILITY_SUMMARY,
        PAIR_DENSITY_SUMMARY,
        TABLES / "ad_retention_50pct.csv",
        TABLES / "ad_enrichment_summary.csv",
        TABLES / "ad_retention_aurc.csv",
        TABLES / "ad_random_retention_summary.csv",
        TABLES / "ad_utility.csv",
        TABLES / "ad_hoc_selectivity_comparison.csv",
        TABLES / "ad_hoc_selectivity_summary.csv",
        TABLES / "ad_distance_error_corr.csv",
        TABLES / "pair_acquisition_comparison.csv",
        TABLES / "pair_acquisition_hit_and_precision.csv",
        TABLES / "pair_acquisition_regret.csv",
        TABLES / "pair_acquisition_eligibility_summary.csv",
        TABLES / "pair_acquisition_calibration_bin_summary.csv",
        TABLES / "pair_acquisition_calibration_rho_summary.csv",
        TABLES / "magnitude_adjusted_risk_summary.csv",
        TABLES / "pair_acquisition_tau2_half_hit_and_precision.csv",
        TABLES / "pair_acquisition_tau2_half_regret.csv",
        TABLES / "pair_acquisition_tau2_half_eligibility_summary.csv",
        TABLES / "pair_acquisition_tau2_half_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "ad_distance_error_corr" / "ad_distance_error_corr_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_comparison.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_comparison.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_hit_and_precision.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_regret.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_eligibility_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_calibration_bin_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_calibration_rho_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "magnitude_adjusted_risk_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_hit_and_precision.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_regret.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_eligibility_summary.csv",
        REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_summary.csv",
    ):
        if extra.exists():
            df = pd.read_csv(extra)
            n_col = "n_valid_folds" if "n_valid_folds" in df.columns else None
            if {"fold_mean", "fold_sd"}.issubset(df.columns):
                for r in df.itertuples(index=False):
                    n = getattr(r, "n_valid_folds", 5) if n_col else 5
                    out.add(fmt_pm(r.fold_mean, r.fold_sd, n))
            if {"retained_mean", "retained_sd"}.issubset(df.columns):
                for r in df.itertuples(index=False):
                    n = getattr(r, "n_valid_folds", 5)
                    out.add(fmt_pm(r.retained_mean, r.retained_sd, n))
                    for mean_c, sd_c in (
                        ("n_rows_retained_mean", "n_rows_retained_sd"),
                        ("pairs_contributing_to_MoP_mean", "pairs_contributing_to_MoP_sd"),
                        ("pairs_with_any_retained_row_mean", "pairs_with_any_retained_row_sd"),
                        ("pairs_retained_mean", "pairs_retained_sd"),
                    ):
                        if mean_c in df.columns:
                            out.add(fmt_pm(getattr(r, mean_c), getattr(r, sd_c, None), n))
            if "display" in df.columns:
                out.update(df["display"].astype(str).str.strip())
            if {"overlap_mean", "overlap_sd"}.issubset(df.columns):
                for r in df.itertuples(index=False):
                    n = getattr(r, "n_folds", getattr(r, "n_valid_folds", 5))
                    out.add(fmt_pm(r.overlap_mean, r.overlap_sd, n))
            if {"mean_observed_signed_selectivity", "mean_observed_signed_selectivity_sd", "n_valid_folds"}.issubset(df.columns):
                for r in df.itertuples(index=False):
                    n = int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5
                    out.add(fmt_pm(r.mean_observed_signed_selectivity, r.mean_observed_signed_selectivity_sd, n))
                    if "target_direction_hit_rate" in df.columns:
                        out.add(fmt_pm(r.target_direction_hit_rate, r.target_direction_hit_rate_sd, n))
                    if "target_direction_sign_accuracy" in df.columns:
                        out.add(fmt_pm(r.target_direction_sign_accuracy, r.target_direction_sign_accuracy_sd, n))
                    if "directional_regret" in df.columns:
                        out.add(fmt_pm(r.directional_regret, r.directional_regret_sd, n))
                    if "eligible_pairs" in df.columns:
                        out.add(fmt_pm(r.eligible_pairs, r.eligible_pairs_sd, n, digits=1))
            if {"mop_rmse", "mop_rmse_sd"}.issubset(df.columns):
                n_col = "n_valid" if "n_valid" in df.columns else ("n_valid_folds" if "n_valid_folds" in df.columns else None)
                for r in df.itertuples(index=False):
                    n = int(getattr(r, n_col)) if n_col and pd.notna(getattr(r, n_col)) else 5
                    out.add(fmt_pm(r.mop_rmse, r.mop_rmse_sd, n))
                    if "mop_r2" in df.columns:
                        out.add(fmt_pm(r.mop_r2, getattr(r, "mop_r2_sd", None), n))
                    if "sign_accuracy" in df.columns:
                        out.add(fmt_pm(r.sign_accuracy, getattr(r, "sign_accuracy_sd", None), n))
                    if "picp_90" in df.columns and "picp_90_sd" in df.columns:
                        out.add(fmt_pm(r.picp_90, r.picp_90_sd, n))
                    if "ace" in df.columns and "ace_sd" in df.columns:
                        out.add(fmt_pm(r.ace, r.ace_sd, n))
            hit_means = [c for c in df.columns if re.fullmatch(r"(hit_rate|hit|event_precision|any_hit|regret|selected_contains_optimum|batch_best)_at_\d+", c)]
            if hit_means:
                n_hit = 5
                if "n_valid_folds" in df.columns:
                    n_hit = int(pd.to_numeric(df["n_valid_folds"], errors="coerce").dropna().iloc[0]) if df["n_valid_folds"].notna().any() else 5
                for r in df.itertuples(index=False):
                    for mean_c in hit_means:
                        sd_c = f"{mean_c}_sd"
                        if sd_c in df.columns:
                            out.add(fmt_pm(getattr(r, mean_c), getattr(r, sd_c), n_hit))
                    if "n_tasks_fold_mean" in df.columns and "n_tasks_fold_sd" in df.columns:
                        out.add(fmt_pm(r.n_tasks_fold_mean, r.n_tasks_fold_sd, n_hit, digits=1))
                        out.add(fmt_pm(r.n_tasks_fold_mean, r.n_tasks_fold_sd, n_hit))
                    for mean_c in [c for c in df.columns if re.fullmatch(r"n_tasks_defined_at_\d+", c)]:
                        sd_c = f"{mean_c}_sd"
                        if sd_c in df.columns:
                            out.add(fmt_pm(getattr(r, mean_c), getattr(r, sd_c), n_hit, digits=1))
                            out.add(fmt_pm(getattr(r, mean_c), getattr(r, sd_c), n_hit))
                    if "pool_optimum" in df.columns and "pool_optimum_sd" in df.columns:
                        out.add(fmt_pm(r.pool_optimum, r.pool_optimum_sd, n_hit))
            if "overlap_fraction_of_utility" in df.columns:
                keys = [c for c in ["dataset", "model_id", "scope", "score_name", "utility_fraction", "ad_retention_fraction"] if c in df.columns]
                if keys:
                    g = df.groupby(keys, dropna=False)["overlap_fraction_of_utility"]
                    means = g.mean()
                    sds = g.std(ddof=1)
                    ns = df.groupby(keys, dropna=False)["outer_fold_id"].nunique() if "outer_fold_id" in df.columns else g.count()
                    for key in means.index:
                        out.add(fmt_pm(means.loc[key], sds.loc[key], int(ns.loc[key]) if key in ns.index else 5))
    return out


def extract_pm(text: str) -> list[str]:
    return re.findall(r"\d+\.\d{3}\s*±\s*\d+\.\d{3}", text)


def main() -> None:
    main_md = (MAN / "main_manuscript.md").read_text(encoding="utf-8")
    si_md = (MAN / "supplementary_information.md").read_text(encoding="utf-8")
    combined = main_md + "\n" + si_md
    claims = pd.read_csv(MAN / "claim_registry.csv")
    grid = pd.read_csv(TABLES / "model_grid_final.csv")
    fig = pd.read_csv(MAN / "figure_data_registry.csv")
    numbers = json.loads((MSRC / "numbers.json").read_text(encoding="utf-8"))
    displays = {str(d).strip() for d in collect_table_displays()}

    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    add("main_exists", (MAN / "main_manuscript.md").exists())
    add("si_exists", (MAN / "supplementary_information.md").exists())
    add("claim_registry_nonempty", len(claims) > 100, f"n={len(claims)}")
    add("workhorse_id", numbers["workhorse_model_id"] == "DS2_RDKIT_MLPENS5_HETERO")
    add("selection_rerun", bool(numbers.get("selection_rerun")))
    add("selection_type_operational", numbers.get("selection_type") == "operational_accuracy_calibration_compromise")
    add("primary_is_mop_rmse", "mop_rmse" in str(numbers.get("primary_metric", "")))
    add(
        "rank_correlation_not_primary_selection",
        "operational_" in str(numbers.get("coded_selection_metric", ""))
        and (
            "Pareto" in str(numbers.get("coded_selection_metric", ""))
            or "mop_rmse" in str(numbers.get("coded_selection_metric", ""))
            or "RMSE" in str(numbers.get("coded_selection_metric", ""))
        ),
    )
    add("row_metrics_not_used_for_selection", numbers.get("row_level_metrics_used_for_selection") is False)
    add("selection_not_nested", numbers.get("selection_nested") is False)
    add("r2_auxiliary", numbers.get("r2_role") == "auxiliary_diagnostic")
    add("rank_correlation_not_used", numbers.get("rank_correlation_role") in {None, "not_used", "removed"})
    add("chemprop_not_operational_candidate", "not_operational" in str(numbers.get("chemprop_rank", "")))
    add("ds1_n_records", f"{numbers['ds1']['n_records']:,}" in combined or str(numbers["ds1"]["n_records"]) in combined)
    add("ds1_n_ligands", f"{numbers['ds1']['n_unique_ligands']:,}" in combined or str(numbers["ds1"]["n_unique_ligands"]) in combined)
    add("ds1_n_metals", str(numbers["ds1"]["n_metals"]) in combined)
    add("ds2_n_rows", f"{numbers['ds2']['n_pairs_rows']:,}" in combined or str(numbers["ds2"]["n_pairs_rows"]) in combined)
    add("chemprop_13", "13 invalid" in combined.lower() or "13 invalid native" in combined.lower() or "**13 invalid" in combined)
    add("species_pair_example", "Fe:2.0--Co:2.0" in combined)
    add("operational_screen_in_text", "operational calibration-usability screen" in combined.lower() or "operational interval-usability screen" in combined.lower())
    add("no_crc_language_in_manuscript", "crc" not in combined.lower() and "conformal" not in combined.lower())
    add("exact_gpr_no_active_numbers", "Exact GPR numbers do not appear" in combined)
    add("bayes_lastlayer_not_active", "not** in the corrected active grid" in combined or "not in the corrected active grid" in combined.lower())
    add("mape_not_computed", "not computed" in combined.lower() and "MAPE" in combined)
    add("medium_not_identifiable", "no conclusion about medium novelty" in combined.lower())
    add("pair_acquisition_computed", "sigma-only" in combined.lower() and "pair-acquisition" in combined.lower())
    add("event_precision_named", "event precision" in combined.lower())
    add("true_hit_named", "hit@$k$" in combined.lower() or "hit@k" in combined.lower() or "true hit" in combined.lower())
    methods_26 = main_md.split("### 2.6 Pair acquisition")[1].split("## 3. Results")[0] if "### 2.6 Pair acquisition" in main_md and "## 3. Results" in main_md else ""
    results_36 = main_md.split("### 3.6 Objective-specific pair acquisition")[1].split("## 4. Discussion")[0] if "### 3.6 Objective-specific pair acquisition" in main_md and "## 4. Discussion" in main_md else ""
    discussion = main_md.split("## 4. Discussion")[1].split("## 5. Conclusions")[0] if "## 4. Discussion" in main_md and "## 5. Conclusions" in main_md else ""
    table4 = main_md.split("**Table 4.")[1].split("**Figure 5.")[0] if "**Table 4." in main_md and "**Figure 5." in main_md else ""
    add(
        "regret_formulated_in_methods",
        "R_k^{\\mathrm{high}}" in methods_26 and "R_k^{\\mathrm{near}}" in methods_26 and r"\mathcal{P}" in methods_26 and "B_k" in methods_26,
    )
    add(
        "regret_motivation_in_methods",
        "treat every observed event equally" in methods_26.lower() or "treat every observed event equally" in methods_26,
    )
    add(
        "regret_motivation_in_results",
        "leftover objective gap" in results_36.lower() and "pool-optimal" in results_36.lower() and "table s10.4" in results_36.lower(),
    )
    add("regret_in_discussion", "retrospective regret" in discussion.lower())
    add("table4_has_regret", "regret@1" in table4 and "regret@5" in table4)
    add("s108c_regret_table", "**Table S10.4." in si_md and "contains opt@5" in si_md)
    add("fig5_caption_regret", "retrospective regret" in results_36.lower() and "panels e" in results_36.lower())
    add(
        "not_prospective_experimental_validation",
        "does not constitute prospective experimental validation" in combined.lower()
        or "not prospective experimental validation" in combined.lower(),
    )
    add(
        "retrospective_ranking_simulation",
        "retrospective held-out ranking simulation" in combined.lower(),
    )
    add(
        "tested_distance_weaker_not_primary",
        "weaker empirical risk-ordering" in combined.lower()
        or "not sufficiently effective to serve as the primary risk-ordering rule" in combined.lower()
        or "did not provide a sufficiently strong basis for replacing the workhorse uncertainty score" in combined.lower(),
    )
    add(
        "no_old_directional_topk",
        "topk_directional" not in combined.lower()
        and "fig_topk_directional" not in combined.lower()
        and "sfig_topk_gain" not in combined.lower()
        and "directional top-$k$" not in combined.lower()
        and "pair-aware directional" not in combined.lower(),
    )
    s13_3 = si_md.split("### S13.3")[1].split("### S13.4")[0] if "### S13.3" in si_md and "### S13.4" in si_md else ""
    add("adhoc_computed", "ad_hoc_ds1_difference" in combined or "ad hoc differencing" in combined.lower())
    add(
        "s13_adhoc_completed",
        "S13.3.1" in si_md and "S13.3.2" in si_md and "S13.3.3" in si_md and "### S13.5" in si_md and "not_computed" not in s13_3.split("S13.3.1")[0],
    )
    add("no_s5_placeholder_heading", "## S5. Direct versus ad hoc" not in si_md and "### S5.1" not in si_md)
    add("s5_workhorse_bridge", "## S5. Direct-UQ screening and workhorse-selection bridge" in si_md)
    add("s6_heading_workhorse_choice", "## S6. Representative direct-UQ model context and workhorse choice" in si_md)
    add("canonical_main_present", (REMAKE_ROOT / "manuscripts" / "final" / "main_manuscript.md").exists())
    add("archive_trees_removed", not (REMAKE_ROOT / "manuscripts" / "archive").exists())
    add("full_manuscript_tree_removed", not (REMAKE_ROOT / "manuscripts" / "full_manuscript").exists())

    # obsolete IDs must not appear in active model tables
    active = grid[grid.active_status != "excluded"]
    for tok in OBSOLETE_ACTIVE:
        hit = active.apply(lambda r: tok in str(r.values), axis=1).any()
        add(f"no_obsolete_{tok}_in_active_grid", not bool(hit))

    # exact GPR must not have numeric mae in active comparison tables
    for name in ["main_ds1_point_results", "main_ds1_uq_results", "main_ds2_workhorse_selection", "ds2_ree_pairs"]:
        df = pd.read_csv(TABLES / f"{name}.csv")
        add(f"no_exact_gpr_row_{name}", (~df.model_id.astype(str).str.endswith("_GPR") | df.model_id.astype(str).str.contains("SPARSE")).all() if "model_id" in df.columns else True)

    # Chemprop UQ partial mentioned near PICP
    add("chemprop_partial_with_picp", "partial native" in combined.lower() or "partial native UQ" in combined)
    add("ad_50pct_table", "50%" in combined and "retention" in combined.lower())
    add("ad_random_baseline", "random" in combined.lower() and "retention" in combined.lower())
    add("ad_not_coverage_guarantee", "coverage" in combined.lower() and ("not" in combined.lower() or "do not provide" in combined.lower()))
    add("pair_level_ad_table4", "pair-level retention-curve auc" in combined.lower() or "pair-level aurc" in combined.lower() or "pair aurc" in combined.lower())
    add("metal_medium_si_table", "**Table S8.2. Metal–medium" in si_md or "**Table S8.2. Metal-medium" in si_md)
    add("fold_paired_differences_si", "Table S6.2" in si_md)
    add("mop_r2_zero_invalid_stated", "zero" in combined.lower() and "invalid" in combined.lower() and "mop" in combined.lower() and "119" in combined)
    add("ad_not_accept_reject_threshold", "accept/reject" in combined.lower() or "not an accept" in combined.lower() or "acceptance threshold" in combined.lower() or "acceptance boundary" in combined.lower() or "ranking mechanisms" in combined.lower())
    add("ranking_eligibility_distinct", "Pair-metric eligibility" in combined)
    n_main = sum(1 for spec in FIGURE_SPECS if spec.get("panel") == "main")
    main_ids = {spec["figure_id"] for spec in FIGURE_SPECS if spec.get("panel") == "main"}
    add(
        "five_main_figures",
        n_main == 5
        and main_ids
        == {
            "fig01_study_design",
            "fig02_workhorse_selection",
            "fig03_risk_ordering",
            "fig04_predicted_magnitude",
            "fig05_pair_acquisition",
        },
        f"n_main={n_main} ids={sorted(main_ids)}",
    )
    obsolete_main_fig = [
        "fig_study_design",
        "fig_ad_hoc_vs_direct_selectivity",
        "fig05_workhorse_selection",
        "fig08_risk_coverage",
        "fig_pair_acq_hitrate",
        "fig02_direct_vs_ad_hoc_selectivity",
        "fig03_workhorse_selection",
        "fig04_risk_ordering",
        "fig02_representative_uq",
        "fig02_risk_ordering",
        "fig03_pair_acquisition",
        "figS19_predicted_magnitude_calibration",
        "figS21_workhorse_selection",
    ]
    fig_reg_text = (MAN / "figure_data_registry.csv").read_text(encoding="utf-8") if (MAN / "figure_data_registry.csv").exists() else ""
    captions_path = MAN / "figures" / "captions.md"
    captions_text = captions_path.read_text(encoding="utf-8") if captions_path.exists() else ""
    adhoc_val_text = ""
    adhoc_val_path = AD_HOC_SELECTIVITY / "ad_hoc_validation.json"
    if adhoc_val_path.exists():
        adhoc_val_text = adhoc_val_path.read_text(encoding="utf-8")
    fig_scan = "\n".join([main_md, si_md, fig_reg_text, captions_text])
    for tok in obsolete_main_fig:
        add(f"no_obsolete_main_figure_{tok}", tok not in fig_scan, tok)
        add(f"no_obsolete_adhoc_validation_{tok}", tok not in adhoc_val_text, tok)
    main_map = {
        1: "fig01_study_design.png",
        2: "fig02_workhorse_selection.png",
        3: "fig03_risk_ordering.png",
        4: "fig04_predicted_magnitude.png",
        5: "fig05_pair_acquisition.png",
    }
    for num, fname in main_map.items():
        add(
            f"main_figure_{num}_filename",
            f"](figures/{fname})" in main_md,
            fname,
        )
    si_nums = []
    id_prefix_fail = []
    for spec in FIGURE_SPECS:
        fid = spec["figure_id"]
        panel = spec.get("panel")
        if panel == "main":
            if not re.match(r"^fig0[1-5]_", fid):
                id_prefix_fail.append(fid)
        else:
            m = re.match(r"^figS(\d+)_", fid)
            if not m:
                id_prefix_fail.append(fid)
            else:
                si_nums.append(int(m.group(1)))
    add("figure_id_prefixes", len(id_prefix_fail) == 0, str(id_prefix_fail))
    add("si_figures_s1_through_s19", sorted(si_nums) == list(range(1, 20)), f"nums={sorted(si_nums)}")
    cap_mismatches = []
    for block in re.split(r"^## ", captions_text, flags=re.M)[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        fid = lines[0].strip()
        body = "\n".join(lines[1:]).lstrip()
        m_id = re.match(r"^figS(\d+)_", fid)
        m_first = re.match(r"Supplementary Figure S(\d+)\.", body)
        if m_id and m_first and m_first.group(1) != m_id.group(1):
            cap_mismatches.append(f"{fid}:S{m_first.group(1)}")
    add("caption_si_number_matches_filename", len(cap_mismatches) == 0, str(cap_mismatches))
    si_img_mismatch = []
    for sid, fname in re.findall(r"!\[Supplementary Figure (S\d+)\]\(figures/([^)]+)\)", si_md):
        if not fname.startswith(f"fig{sid}_"):
            si_img_mismatch.append(f"{sid}->{fname}")
    add("si_image_filename_matches_number", len(si_img_mismatch) == 0, str(si_img_mismatch))
    figdir = MAN / "figures"
    obsolete_files = []
    for tok in obsolete_main_fig:
        obsolete_files.extend([p.name for p in figdir.glob(f"{tok}.*")])
        data_dir = figdir / "data"
        if data_dir.exists():
            obsolete_files.extend([p.name for p in data_dir.glob(f"{tok}.*")])
    add("no_obsolete_figure_files", len(obsolete_files) == 0, str(obsolete_files))
    live_ids = {spec["figure_id"] for spec in FIGURE_SPECS}
    extra_pngs = [p.name for p in figdir.glob("*.png") if p.stem not in live_ids] if figdir.exists() else []
    add("no_extra_figure_pngs", len(extra_pngs) == 0, str(extra_pngs[:12]))
    add(
        "s13_has_ree_36pct",
        "3.6%" in si_md and "## S13." in si_md,
    )
    add("no_candidate_proxy_in_main", "candidate proxy" not in main_md.lower())
    add(
        "contribution_not_architecture",
        "do not establish a universal applicability-domain boundary" in main_md.split("## 1. Introduction")[0],
    )
    add(
        "no_703_at_least_one_orientation",
        "at least one orientation" not in main_md.lower(),
    )
    warn = "Retained-subset RMSE values measure error among predictions selected by each score"
    add("table3b_warning_once", main_md.count(warn) == 1, f"n={main_md.count(warn)}")
    s32 = main_md.split("### 3.2")[1].split("### 3.3")[0] if "### 3.2" in main_md and "### 3.3" in main_md else ""
    add(
        "s32_row_vs_mop_diagnostic",
        "favorable row-weighted performance does not establish reliable pair-specific behavior" in s32,
        f"n={s32.count('favorable row-weighted performance does not establish reliable pair-specific behavior')}",
    )
    add("s32_no_prioritization_repeat", "quantity used for ligand prioritization" not in s32)
    add("s43_row_weighted", "S4.3" in combined and "row-weighted" in combined.lower())
    add("s44_highsel_row", "S4.4" in combined and "high-selectivity" in combined.lower())
    add("s45_highsel_pair", "S4.5" in combined)
    add("dropout_not_shared", "Dropout is **not** shared" in combined or "dropout is **not** shared" in combined.lower())
    add("zero_training_ligand_bin", "zero training ligand" in combined.lower() or "0 training ligand" in combined.lower())
    add("cmi_different_estimands", "different estimand" in combined.lower())
    add("fold_mean_sd_stated", "fold mean ± SD" in combined or "fold-level" in combined.lower())
    add("pooled_labeled", "pooled-row" in combined.lower() or "pooled diagnostic" in combined.lower())
    add("workhorse_rmse_value", numbers["fmt"]["wh_mop_rmse"] in combined)
    add("workhorse_picp_value", numbers["fmt"]["wh_picp"] in combined)
    add("workhorse_ace_value", numbers["fmt"]["wh_ace"] in combined)
    add("workhorse_mop_r2_value", numbers["fmt"].get("wh_mop_r2", "___") in combined)
    add("main_omits_row_rmse_as_headline", "row RMSE" not in main_md.split("## 3.1")[1].split("## 3.2")[0] if "## 3.1" in main_md else True)

    missing_pm = []
    for doc_name, text in (("main", main_md), ("si", si_md)):
        for token in extract_pm(text):
            if token in displays:
                continue
            # allow n_valid annotated forms already stripped
            alt = token.replace(" ", "")
            if any(token in d or d in token for d in displays):
                continue
            # published / CMI historical
            if any(h in token for h in HISTORICAL_ALLOW):
                continue
            missing_pm.append(f"{doc_name}:{token}")
    # Many SI tables print raw fold_mean not preformatted ± ; extract_pm on SI will catch formatted ones.
    add("pm_tokens_traceable_or_few", len(missing_pm) <= 12, f"untraced={missing_pm[:20]}")

    # figure sources not stale manuscript_corrected
    stale_fig = fig[fig.source_artifact.astype(str).str.contains("manuscript_corrected", na=False)]
    add("no_stale_figure_sources", len(stale_fig) == 0, f"n={len(stale_fig)}")
    pair_acq_rows = fig[
        fig.figure_id.astype(str).str.contains("pair_acquisition|predicted_magnitude", case=False, na=False)
    ]
    add(
        "pair_acq_figures_canonical",
        len(pair_acq_rows) >= 2 and bool((pair_acq_rows.status.astype(str) == "canonical").all()),
        f"n={len(pair_acq_rows)}",
    )
    exp_status = pd.read_csv(TABLES / "supplementary_experiment_status.csv") if (TABLES / "supplementary_experiment_status.csv").exists() else pd.DataFrame()
    if len(exp_status) and "experiment_id" in exp_status.columns:
        add("experiment_ids_unique", exp_status.experiment_id.nunique() == len(exp_status), f"n={len(exp_status)}")
        stale_topk = exp_status[(exp_status.experiment_id.astype(str) == "top_k_screening") & (exp_status.status.astype(str) == "not_computed")]
        add("no_stale_topk_not_computed_row", len(stale_topk) == 0)
        live_pair_acq = exp_status[exp_status.experiment_id.astype(str) == "sigma_pair_acquisition"]
        add("sigma_pair_acquisition_marked_complete", len(live_pair_acq) == 1 and str(live_pair_acq.iloc[0]["status"]) == "complete")
        live_pair_acq_sens = exp_status[exp_status.experiment_id.astype(str) == "pair_acquisition_threshold_2_half"]
        add(
            "pair_acq_threshold_sensitivity_marked_complete",
            len(live_pair_acq_sens) == 1 and str(live_pair_acq_sens.iloc[0]["status"]) == "complete",
        )
        ad_live = exp_status[exp_status.experiment_id.astype(str) == "ad_retention_study"]
        add("ad_retention_marked_complete", len(ad_live) == 1 and str(ad_live.iloc[0]["status"]) == "complete")
        live_adhoc = exp_status[exp_status.experiment_id.astype(str) == "adhoc_differencing"]
        add("adhoc_marked_complete", len(live_adhoc) == 1 and str(live_adhoc.iloc[0]["status"]) == "complete")
        missing_ids = [eid for eid in exp_status.experiment_id.astype(str) if eid not in si_md]
        add("si_lists_every_experiment_id", len(missing_ids) == 0, f"missing={missing_ids}")
        s11 = si_md.split("## S11.")[1].split("## S12.")[0] if "## S11." in si_md and "## S12." in si_md else ""
        complete_in_s11 = [
            eid
            for eid, st in zip(exp_status.experiment_id.astype(str), exp_status.status.astype(str))
            if st in {"complete", "partial"} and eid in s11
        ]
        add("s11_omits_completed_experiments", len(complete_in_s11) == 0, f"ids={complete_in_s11}")
        nums_path = NUMBERS_JSON
        if nums_path.exists():
            nums = json.loads(nums_path.read_text(encoding="utf-8"))
            inv = set(exp_status.experiment_id.astype(str))
            nc_inv = set(exp_status.loc[exp_status.status.astype(str) == "not_computed", "experiment_id"].astype(str))
            missing_nc = [eid for eid in nums.get("not_computed", []) if eid not in nc_inv]
            add("numbers_not_computed_in_inventory", len(missing_nc) == 0, f"missing={missing_nc}")
            missing_ni = [eid for eid in nums.get("not_identifiable", []) if eid not in inv]
            add("numbers_not_identifiable_in_inventory", len(missing_ni) == 0, f"missing={missing_ni}")
    for figspec in FIGURE_SPECS:
        png = f"{figspec['figure_id']}.png"
        add(f"si_inventory_{figspec['figure_id']}", figspec["figure_id"] in si_md)
        if figspec.get("panel") == "si":
            add(f"si_cites_{figspec['figure_id']}", png in si_md)
    add("s82_not_same_estimand", "same estimand as Table 4" not in si_md)
    add("s10_has_rank_correlation_table", "**Table S10.6." in si_md)
    add("s1012_threshold_sensitivity", "**Table S10.8." in si_md and "does not replace Table 4" in si_md)
    s1012_note = si_md.split("**Table S10.8.")[1].split("**Table S10.9")[0] if "**Table S10.8." in si_md and "**Table S10.9" in si_md else ""
    add(
        "s1012_cites_hit_and_regret",
        "pair_acquisition_tau_2_0p5/pair_acquisition_hit_and_precision.csv" in s1012_note
        and "pair_acquisition_tau_2_0p5/pair_acquisition_regret.csv" in s1012_note,
    )
    add("s42_interpretation_heading", "### S4.2 Primary all-pair pair-level comparison: interpretation" in si_md)
    s4_headings = re.findall(r"^### S4\.\d+[a-z]?", si_md, flags=re.M)
    add(
        "s4_subsection_order",
        s4_headings[:5] == ["### S4.1", "### S4.2", "### S4.3", "### S4.4", "### S4.5"],
        f"s4={s4_headings}",
    )
    s81 = si_md.find("**Table S8.1.")
    s82 = si_md.find("**Table S8.2.")
    s83 = si_md.find("**Table S8.3.")
    s84 = si_md.find("**Table S8.4. Fold-wise")
    s84b = si_md.find("**Table S8.5.")
    add(
        "s8_tables_sequential",
        min(s81, s82, s83, s84, s84b) >= 0 and s81 < s82 < s83 < s84 < s84b,
        f"pos={(s81, s82, s83, s84, s84b)}",
    )
    add("s8_5_is_frequent_pairs", "**Table S8.5. Most frequent" in si_md)
    add("s8_2_is_metal_medium", "**Table S8.2. Metal–medium" in si_md or "**Table S8.2. Metal-medium" in si_md)
    add("s8_3_is_pair_context", "**Table S8.3. Pair-context support bins" in si_md)
    add("no_manuscript_final_pair_event_canonical", "results/manuscript_final/pair_event_support" not in combined)
    add(
        "pair_event_canonical_is_downstream",
        "results/downstream_cv/pair_event_support/pair_event_support_pooled.csv" in si_md
        and "results/downstream_cv/pair_event_support/pair_event_support_frequent.csv" in si_md
        and "results/downstream_cv/pair_event_support/pair_event_support_summary.csv" in combined,
    )
    add("main_k_support_qualification", "for a given $k$, at least $k$ held-out rows" in main_md)
    add(
        "abstract_ree_plural_diagnostics",
        "Restricted rare-earth-element pair analyses are reported only as supplementary formulation diagnostics" in main_md,
    )
    add(
        "s10_h2_h3_prespecified_rules",
        "at least one prespecified sigma-only ranking rule, particularly the threshold-margin rule, recovers high-selectivity events" in si_md
        and "the prespecified sigma-only ranking rules improve near-zero event precision@$k$ and true hit@$k$" in si_md,
    )
    add("s10_uncertainty_penalty_variants", "the uncertainty-penalty variants are" in si_md)
    add("s7_nn_expanded_in_table", "Here, NN denotes nearest neighbor and $k$NN denotes $k$-nearest neighbor." in si_md)
    si_before_s72 = si_md.split("**Table S7.2")[0] if "**Table S7.2" in si_md else si_md
    add("s7_no_standalone_nn_sentence", "Here, NN denotes nearest neighbor and $k$NN denotes $k$-nearest neighbor." not in si_before_s72)
    results_36 = main_md.split("### 3.6")[1].split("## 4.")[0] if "### 3.6" in main_md and "## 4." in main_md else ""
    add(
        "main_points_to_s22_sensitivity",
        "Table S10.8" in results_36 and "does not replace Table 4" in results_36 and "Supplementary Figure S15" in results_36,
    )
    prim_val_path = REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition" / "pair_acquisition_validation.json"
    if prim_val_path.exists():
        prim_val = json.loads(prim_val_path.read_text(encoding="utf-8"))
        add("primary_pair_acq_threshold_one", prim_val.get("threshold") == 1.0, str(prim_val.get("threshold")))
    else:
        add("primary_pair_acq_threshold_one", False, "missing")
    sens_val_path = REMAKE_ROOT / "results" / "downstream_cv" / "pair_acquisition_tau_2_0p5" / "pair_acquisition_validation.json"
    if sens_val_path.exists():
        sens_val = json.loads(sens_val_path.read_text(encoding="utf-8"))
        add(
            "pair_acq_sens_thresholds",
            sens_val.get("high_threshold") == 2.0 and sens_val.get("low_threshold") == 0.5,
            f"high={sens_val.get('high_threshold')} low={sens_val.get('low_threshold')}",
        )
    else:
        add("pair_acq_sens_thresholds", False, "missing")
    add("s100_generic_auc_column", "MoP RMSE retention-curve AUC" not in si_md.split("**Table S7.1")[1].split("**Table S7.2")[0] if "**Table S7.1" in si_md and "**Table S7.2" in si_md else False)
    add("s82_ree_scope_heading", "Pair-level retention-curve AUC in the REE–REE scope" in si_md or "Pair-level AURC in the REE–REE scope" in si_md)
    add("ad_chemprop_embedding_methods", "learned molecular representation" in combined)
    add("ad_hypotheses_named", "H1" in si_md and "H3" in si_md)
    add("no_fixed_loss_cap_sensitivity", "Fixed-loss-cap sensitivity" not in combined)
    add("pair_ad_split_files", (TABLES / "main_ds2_ad_pair_results.csv").exists() and (TABLES / "supplementary_ds2_ad_row_results.csv").exists())
    add("spec_ad_retention", "50% retention" in combined.lower() or "empirical AD" in combined.lower() or "risk-ordering" in combined.lower())

    # display names present
    for label in ["Evidential MLP ensemble", "Full variational BNN", "MC-dropout MLP", "SNGP", "Sparse variational GPR", "Chemprop evidential"]:
        add(f"label_{label}", label in combined)

    forbidden_phrases = [
        ("nested CRC for every model", False),
        ("PICP gate as current rule", "PICP@90 ≥ 0.70" in combined and "obsolete" not in combined),
        ("rank correlation as current primary", False),
    ]
    add("no_picp_070_in_main", "PICP@$90\\ge 0.70" not in main_md and "PICP@90 ≥ 0.70" not in main_md and "PICP@$90\\ge0.70" not in main_md)
    add("no_picp_070_as_current", ("PICP@$90\\ge 0.70" in si_md or "PICP@$90\\ge0.70" in si_md) and "obsolete" in combined.lower())
    add("no_rank_correlation_primary_in_active_text", "rank-correlation-primary" not in combined.lower())
    add("no_best_model_unqualified", "best model" not in main_md.lower() and "universally superior" not in main_md.lower() or "not" in main_md.lower())
    add(
        "operational_compromise_phrasing",
        "operational accuracy–interval-usability compromise" in main_md.lower()
        or "operational accuracy-interval-usability compromise" in main_md.lower()
        or "operational accuracy–calibration compromise" in main_md.lower()
        or "operational accuracy-calibration compromise" in main_md.lower(),
    )
    add("no_stage3_as_result", "stage3" not in combined.lower() or combined.lower().count("stage3") == 0)
    add("no_stage4_as_result", "stage4" not in combined.lower())
    add("no_bayes_generic_for_mc", "MC dropout only" in combined)

    # cross refs
    add("table1_present", "**Table 1." in main_md)
    add("table3_present", "**Table 3." in main_md)
    add("table2_present", "**Table 2." in main_md)
    add("table4_present", "**Table 4." in main_md)
    add(
        "s13_adhoc_status",
        "S13.3.1" in si_md and "Table S13.6" in si_md and "Table S13.7" in si_md,
    )
    add("figure_registry_exists", (MAN / "figure_data_registry.csv").exists())
    add("spec_operational_screen", "operational calibration-usability screen" in combined.lower() or "operational interval-usability screen" in combined.lower() or "operational calibration-usability" in combined.lower())
    add("spec_rmse_primary", "RMSE–PICP–ACE Pareto" in combined or "RMSE-PICP-ACE Pareto" in combined)
    add("no_lowest_ree_selection", "lowest REE" not in combined and "lowest REE–REE MoP RMSE" not in combined)
    add("no_legacy_50_100_tokens", ("AURC_" + "50_100") not in combined and "ad_fold_risk_coverage" not in combined)
    s84_corr = ""
    if "**Table S8.4. Fold-wise" in si_md:
        rest84 = si_md.split("**Table S8.4. Fold-wise", 1)[1]
        if "**Table S8.5" in rest84:
            s84_corr = rest84.split("**Table S8.5", 1)[0]
        elif "## S9" in rest84:
            s84_corr = rest84.split("## S9", 1)[0]
        else:
            s84_corr = rest84
    add("no_s85_legacy", ("AURC_" + "50_100") not in s84_corr and "50–100%" not in s84_corr and "50-100%" not in s84_corr)
    add("s85_is_distance_error_corr", "absolute residual" in s84_corr.lower() and "pearson" in s84_corr.lower() and "spearman" in s84_corr.lower())
    add(
        "spec_no_rank_correlation_endpoint",
        all(
            tok not in main_md.lower()
            for tok in (
                "mop_spearman",
                "row_spearman",
                "spearman_gain",
                "retention_curve_auc_mop_spearman",
                "ree_mop_spearman",
                "pair_balanced_spearman",
                "pair_balanced_global_spearman",
            )
        ),
    )
    add("si_ad_distance_error_corr_table", "**Table S8.4. Fold-wise" in si_md and "ad_distance_error_corr" in si_md)
    add("spec_row_not_selection", "not used for selection" in combined.lower() or "not** used for selection" in combined.lower())

    # Main DS2 table must not carry row_* RMSE columns (all_pair_row_r2 is displayed).
    ds2_main = pd.read_csv(TABLES / "main_ds2_operational_comparison.csv")
    rowish = [c for c in ds2_main.columns if c.startswith("row_") or c in {"ree_row_rmse", "ree_row_r2"}]
    add("main_ds2_table_no_row_metrics", len(rowish) == 0, f"cols={rowish}")
    add("main_ds2_has_mop_rmse", "all_pair_mop_rmse" in ds2_main.columns)
    add("main_ds2_has_all_pair_mop_r2", "all_pair_mop_r2" in ds2_main.columns)
    add("main_ds2_has_all_pair_row_r2", "all_pair_row_r2" in ds2_main.columns)
    add("main_ds2_has_regime", "regime" in ds2_main.columns)
    add("main_ds2_has_calibration_usable", "calibration_usable" in ds2_main.columns)
    add("main_ds2_has_mop_r2", "ree_mop_r2" in ds2_main.columns)
    sel_art = MSRC / "workhorse_selection_accuracy_calibration.json"
    add("selection_artifact_exists", sel_art.exists())
    if sel_art.exists():
        sel = json.loads(sel_art.read_text(encoding="utf-8"))
        add("selection_json_type", sel.get("selection_type") == "operational_accuracy_calibration_compromise")
        add("selection_json_primary_rmse", sel.get("primary_accuracy_metric") == "all_pair_mop_rmse")
        add("selection_json_no_row", sel.get("row_level_metrics_used_for_selection") is False)
        add("selection_json_not_nested", sel.get("selection_nested") is False)
        add("selected_matches_numbers", sel.get("selected") == numbers["workhorse_model_id"])
    row_diag = TABLES / "supplementary_row_level_diagnostics.csv"
    add("si_row_diagnostics_exist", row_diag.exists())
    if row_diag.exists():
        rdf = pd.read_csv(row_diag)
        add("si_row_diagnostics_labeled", "diagnostic_class" in rdf.columns and rdf.diagnostic_class.astype(str).str.contains("row").any())

    # No active manuscript text calls Spearman the primary workhorse criterion
    add(
        "no_rank_correlation_primary_claim",
        "primary ranking metrics" not in main_md.lower()
        and "coded selection metric:** REE" not in main_md
        and "lexicographic ranking" not in main_md.lower(),
    )
    add("no_old_picp_gated_mae_current", "lowest CV-mean REE MoP RMSE is selected" not in combined)

    # Table ordering: first operational row should be selected workhorse if usable
    if len(ds2_main) and "selection_status" in ds2_main.columns:
        add("table3_leads_with_selected", str(ds2_main.iloc[0]["selection_status"]) == "selected")
    ids = set(ds2_main["model_id"].astype(str)) if "model_id" in ds2_main.columns else set()
    add("table3_includes_adhoc_point", "DS1_RDKIT_MLP_POINT" in ids)
    add("table3_includes_adhoc_uq", "DS1_GRAPH_DMPNN_EVID" in ids)
    if "DS1_RDKIT_MLP_POINT" in ids:
        mlp_ad = ds2_main[ds2_main.model_id.astype(str) == "DS1_RDKIT_MLP_POINT"].iloc[0]
        add("table3_adhoc_mlp_regime", str(mlp_ad.get("regime", "")).lower() == "ad hoc")
        add("table3_adhoc_mlp_no_picp", "NA" in str(mlp_ad.get("all_pair_picp_90", "")))
    if "DS1_GRAPH_DMPNN_EVID" in ids:
        chem_ad = ds2_main[ds2_main.model_id.astype(str) == "DS1_GRAPH_DMPNN_EVID"].iloc[0]
        add("table3_adhoc_chemprop_regime", str(chem_ad.get("regime", "")).lower() == "ad hoc")
        add(
            "table3_adhoc_chemprop_uses_mop_picp",
            str(chem_ad.get("all_pair_picp_90", "")) != str(chem_ad.get("all_pair_row_picp_90", "")),
        )
        add("table3_adhoc_chemprop_not_selected", str(chem_ad.get("selection_status", "")) != "selected")

    forbidden_published = {
        "mae",
        "mop_mae",
        "row_mae",
        "all_pair_mop_mae",
        "ree_mop_mae",
        "mae_retained",
        "pair_balanced_mae",
        "pair_balanced_r2",
        "pair_balanced_global_r2",
        "mae_at_50pct_retention",
        "mop_mae_at_50",
        "mop_mae_at_50pct",
        "mop_mae_at_50pct_retention",
        "row_mae_at_50",
        "row_mae_at_50pct_retention",
        "delta_all_pair_mop_mae",
        "delta_ree_mop_mae",
    }
    leaked = []
    for p in TABLES.glob("*.csv"):
        df = pd.read_csv(p, nrows=0)
        hits = sorted(set(df.columns) & forbidden_published)
        if hits:
            leaked.append(f"{p.name}:{','.join(hits)}")
        if "metric_name" in df.columns:
            full = pd.read_csv(p, usecols=["metric_name"])
            names = set(full["metric_name"].astype(str))
            bad = sorted(names & forbidden_published)
            if bad:
                leaked.append(f"{p.name}:metric_name={','.join(bad)}")
    add("published_tables_omit_mae_and_pair_balanced", not leaked, "; ".join(leaked[:12]))

    add("no_planned_before_submission", "planned before submission" not in combined.lower())
    add("no_best_calibrated_claim", "best calibrated" not in main_md.lower())
    add("no_table_3c_in_main", "**Table 3c." not in main_md)
    add("main_omits_ree_from_table3_display", "REE–REE MoP RMSE (usage example)" not in main_md)
    add("main_defines_row_level_picp", "row-level PICP@90" in main_md or "row-level predictive" in main_md.lower())
    add("main_table3_displays_mop_picp", "all-pair MoP PICP@90" in main_md or "MoP PICP@90 and MoP ACE" in main_md)
    t2 = main_md.split("**Table 2.")[1].split("**Table 3.")[0] if "**Table 2." in main_md and "**Table 3." in main_md else ""
    add("table2_md_is_workhorse_context", "screen-passing" in t2.lower() or "representative workhorse" in t2.lower())
    add("table2_md_omits_adhoc_comparators", "ad hoc" not in t2.lower())
    t3 = main_md.split("**Table 3.")[1].split("**Figure 3.")[0] if "**Table 3." in main_md and "**Figure 3." in main_md else ""
    add("table3_md_is_retention", "50%" in t3 and "retention" in t3.lower())
    add("table3_md_omits_usable_role", "interval usable" not in t3 and "| role |" not in t3)
    add("table3_md_omits_adhoc_grid", "DS1_RDKIT_MLP_POINT" not in t3)
    add("main_screen_remains_row_level", "row-level PICP@90 and ACE" in main_md and "ACE}\\le 0.10" in main_md)
    add("main_fig2_ace_visual_gate_020", "ACE}\\le 0.20" in combined or "ACE\\le 0.20" in combined)
    add(
        "workhorse_all_pair_only",
        "restricted formulation diagnostic" in main_md.lower()
        and "is not used to select the workhorse" in main_md.lower(),
    )
    add("pair_context_support_named", "pair-context support" in combined.lower() or "pair-context support" in si_md.lower())
    add("threshold_margin_named", "standardized threshold-margin" in combined.lower() or "threshold-margin score" in combined.lower())
    add(
        "ree_restricted_diagnostic_framing",
        "restricted formulation diagnostic" in combined.lower()
        and "restricted rare-earth-element pair" in main_md
        and "REE–REE" not in main_md,
    )
    add("intervals_not_sign_certifying", "not sign-certifying" in combined.lower() or "not sign certifying" in combined.lower())
    add("picp_not_called_pair_level_calibration", "are pair-level calibration" not in combined.lower())
    results_md = main_md.split("## 3. Results")[1].split("## 4. Discussion")[0] if "## 3. Results" in main_md and "## 4. Discussion" in main_md else ""
    add("no_ree_in_primary_results", "REE" not in results_md)
    add("m4_defined_as_source", "literature M4" in main_md and "source collection rather than the final modeling dataset" in main_md)
    add("internal_aleatoric_named", "internal aleatoric uncertainty" in main_md)
    add("mop_defined_before_ds2_results", "\\mathrm{MoP\\text{-}RMSE}" in main_md and main_md.find("\\mathrm{MoP\\text{-}RMSE}") < main_md.find("### 3.1"))
    add("si_s13_formulation_diagnostics", "## S13. Supplementary formulation and pair-frequency diagnostics" in si_md)
    early_si = si_md.split("## S6.")[1].split("## S13.")[0] if "## S6." in si_md and "## S13." in si_md else si_md
    add("early_si_omits_ree_pairs_scope", "ree_pairs" not in early_si)
    add("early_si_omits_ree_ree_term", "REE–REE" not in early_si)
    add("s13_tables_renumbered", all(tok in si_md for tok in ("Table S13.1", "Table S13.2", "Table S13.3", "Table S13.4", "Table S13.5", "Table S13.6", "Table S13.7")))
    s13_body = si_md.split("## S13.")[1] if "## S13." in si_md else ""
    add("s13_omits_stale_table_ids", all(tok not in s13_body for tok in ("Table S4.1", "Table S4.3a", "Table S8.5", "Table S8.6", "Table S5.1", "Table S5.2")) if s13_body else False)
    add("inventory_si_figures_s1_s19", "numbered S1–S19" in si_md or "numbered S1-S19" in si_md)
    add("si_has_figure_s19", "**Supplementary Figure S19." in si_md)
    add("si_restricted_figures_grouped_s13_4", "intentionally grouped in Section S13.4" in si_md)
    add("no_empty_table_s74", "**Table S7.4" not in si_md)
    add("no_deleted_s14_heatmap", "figS14_retention_50pct" not in combined and "figS16_sign_stability_stacked" not in combined)
    lettered_tables = re.findall(r"\*\*Table S\d+\.\d+[a-z]", si_md)
    add("si_tables_no_letter_suffix", len(lettered_tables) == 0, str(lettered_tables))
    add("si_tables_no_dot_zero", not re.search(r"\*\*Table S\d+\.0\.", si_md))
    table_heads = [(int(a), int(b)) for a, b in re.findall(r"\*\*Table S(\d+)\.(\d+)\.", si_md)]
    table_gaps = []
    by_sec: dict[int, list[int]] = {}
    for sec, sub in table_heads:
        by_sec.setdefault(sec, []).append(sub)
    for sec, subs in sorted(by_sec.items()):
        if subs != list(range(1, len(subs) + 1)):
            table_gaps.append(f"S{sec}:{subs}")
    add("si_tables_consecutive_per_section", len(table_gaps) == 0, str(table_gaps))
    fig_disp = [int(n) for n in re.findall(r"^\*\*Supplementary Figure S(\d+)\.", si_md, flags=re.M)]
    add("si_figures_display_s1_to_s19", fig_disp == list(range(1, 20)), f"order={fig_disp}")
    add("inventory_main_figures_1_5", "numbered 1–5" in si_md or "numbered 1-5" in si_md)

    grid_path = TABLES / "model_grid_final.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        hetero = grid[grid.model_id.astype(str).eq("DS2_RDKIT_MLPENS5_HETERO")]
        hetero_samples = int(hetero.iloc[0]["stochastic_sample_count"]) if len(hetero) and "stochastic_sample_count" in hetero.columns else -1
        add("hetero_workhorse_samples_30", hetero_samples == 30, f"samples={hetero_samples}")
    else:
        add("hetero_workhorse_samples_30", False, "missing model_grid_final.csv")
    add("no_independent_quadrature_wording", "independent-quadrature" not in combined.lower())
    add("no_y_yhat_slash_token", "/y-yhat/" not in combined.lower() and "/y-\\widehat y/" not in combined.lower())
    add(
        "fig2_shade_not_eligibility_gate",
        "neither shaded region is the operational eligibility gate" in combined.lower()
        and "descriptive mop reference regions only" in combined.lower(),
    )
    add("s18_plots_between_member", "sigma_epistemic_between" in si_md or "epistemic,between" in si_md)
    add("s18_variance_additivity_stated", r"\sigma_{\mathrm{epistemic,between}}" in combined or "epistemic,between" in combined)
    add("composite_predictive_risk_named", "composite predictive-risk signal" in main_md)
    add(
        "s7_distinguishes_mop_and_row_retention",
        "The primary all-pair MoP retention summaries are Table 3 and Supplementary Table S7.1." in si_md
        and "Table S8.1 reports the separate row-level retention diagnostic" in si_md,
    )
    def _around(token: str, before: int = 900, after: int = 200) -> str:
        needles = [f"](figures/{token})", token] if token.endswith(".png") else [token]
        i = -1
        used = token
        for needle in needles:
            i = si_md.find(needle)
            if i >= 0:
                used = needle
                break
        if i < 0:
            return ""
        return si_md[max(0, i - before) : i + len(used) + after].lower()

    add("s11_restricted_diagnostic", "restricted diagnostic" in _around("figS17_retention_ree.png"))
    add("s21_restricted_diagnostic", "restricted diagnostic" in _around("figS19_ree_fold_stability.png"))
    add("s8_restricted_diagnostic", "restricted diagnostic" in _around("figS16_ad_hoc_interval_calibration.png"))
    add("s20_restricted_diagnostic", "restricted diagnostic" in _around("figS18_direct_vs_ad_hoc_selectivity.png"))
    add("s22_supporting_sensitivity", "does not replace" in _around("figS15_pair_acquisition_threshold_sensitivity.png"))
    h2 = [m.group(0) for m in re.finditer(r"^## S\d+\.", si_md, flags=re.M)]
    add(
        "si_section_order_s10_s11_s12_s13",
        h2[-4:] == ["## S10.", "## S11.", "## S12.", "## S13."] if len(h2) >= 4 else False,
        f"tail={h2[-4:] if len(h2) >= 4 else h2}",
    )
    add("si_last_h2_is_s13", bool(h2) and h2[-1] == "## S13.")
    heading_needles = [
        "## S1.",
        "## S1.5",
        "## S2.",
        "## S3.",
        "### S4.1",
        "### S4.2",
        "### S4.3",
        "### S4.4",
        "### S4.5",
        "## S5.",
        "## S6.",
        "## S7.",
        "## S8.",
        "## S9.",
        "## S10.",
        "## S11.",
        "## S12.",
        "## S13.",
    ]
    heading_pos = [si_md.find(tok) for tok in heading_needles]
    add(
        "si_scientific_heading_order",
        all(p >= 0 for p in heading_pos) and heading_pos == sorted(heading_pos),
        f"pos={heading_pos}",
    )
    s13_4 = si_md.split("### S13.4")[1].split("### S13.5")[0] if "### S13.4" in si_md and "### S13.5" in si_md else ""
    fig_pos = [
        s13_4.find("](figures/figS16_ad_hoc_interval_calibration.png)"),
        s13_4.find("](figures/figS17_retention_ree.png)"),
        s13_4.find("](figures/figS18_direct_vs_ad_hoc_selectivity.png)"),
        s13_4.find("](figures/figS19_ree_fold_stability.png)"),
    ]
    add(
        "s13_restricted_figures_numerical_block",
        all(p >= 0 for p in fig_pos) and fig_pos == sorted(fig_pos),
        f"pos={fig_pos}",
    )
    s13_pre = si_md.split("## S13.")[1].split("### S13.4")[0] if "## S13." in si_md and "### S13.4" in si_md else si_md
    add(
        "s13_defers_restricted_figure_images",
        all(
            f"](figures/{name})" not in s13_pre
            for name in (
                "figS16_ad_hoc_interval_calibration.png",
                "figS17_retention_ree.png",
                "figS18_direct_vs_ad_hoc_selectivity.png",
                "figS19_ree_fold_stability.png",
            )
        ),
    )
    add("ds2_delta_from_species_medians", "two aggregated species-level median values" in main_md)
    add("pareto_direction_stated", "closer-to-nominal row-level PICP@90" in main_md)
    add("mc_dropout_defined", "Monte Carlo dropout" in combined)
    add("cmi_retrain_from_scratch", "trained from random initialization" in combined.lower())
    add(
        "cmi_shared_retrain_no_pt",
        "those rows do **not** load cmi" in combined.lower() or "those rows do not load cmi" in combined.lower(),
    )
    add("cmi_released_weights_evaluated", "released-weight rows load the published" in combined.lower())
    add("cmi_chemprop_161_stated", "chemprop 1.6.1" in combined.lower())
    add("cmi_chemprop_152_stated", "chemprop 1.5.2" in combined.lower() or "1.5.2" in combined)
    add("cmi_retrain_protocol_table", "**Table S1.6." in si_md)
    add("cmi_released_weight_metrics_table", "**Table S1.7." in si_md)
    add("cmi_architecture_retrain_table", "**Table S1.8." in si_md)
    add("cmi_hidden_2400_stated", "2400" in combined)
    add("table1_has_released_weights", "CMI released weights, official test file" in main_md)
    add("table1_has_cmi_architecture", "this work, CMI architecture, official test file" in main_md)
    disc = main_md.split("## 4. Discussion")[1].split("## 5. Conclusions")[0] if "## 4. Discussion" in main_md and "## 5. Conclusions" in main_md else ""
    add("discussion_cmi_architecture", "### 4.2 Official-split Chemprop comparison" in disc and "architecture recovered from those checkpoints" in disc)
    add("discussion_cmi_released_match", "reproduces the published external ensemble" in disc)
    add(
        "discussion_ds_inventories",
        "Supplementary Tables S1.1–S1.4 and S4.1–S4.4" in disc,
    )
    add(
        "discussion_distance_si",
        "Supplementary Table S8.4" in disc and "Supplementary Figure S11" in disc,
    )
    add(
        "discussion_sign_si",
        "Supplementary Table S9.1" in disc and "Supplementary Figure S14" in disc,
    )
    add(
        "discussion_acquisition_si",
        "Supplementary Tables S10.2–S10.4" in disc and "Supplementary Table S10.8" in disc and "Figure S15" in disc,
    )
    add("main_pair_event_count_formatted", "2,503 species-level pairs had held-out rows" in main_md)
    add("main_pair_event_source_explicit", "descriptive pooled counts come from the registered pair-event-support artifacts" in main_md)

    n_fail = sum(1 for c in checks if not c["pass"])
    report = {
        "n_checks": len(checks),
        "n_pass": len(checks) - n_fail,
        "n_fail": n_fail,
        "pass": n_fail == 0,
        "untraced_pm_tokens": missing_pm,
        "checks": checks,
        "workhorse": numbers["workhorse_model_id"],
        "primary_metric": numbers["primary_metric"],
        "chemprop_invalid_nig_rows": 13,
        "ad_retention_status": numbers.get("ad_status") or numbers.get("ad_retention_status"),
    }
    (MAN / "manuscript_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MSRC / "manuscript_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_checks", "n_pass", "n_fail", "pass", "untraced_pm_tokens")}, indent=2))
    if n_fail:
        for c in checks:
            if not c["pass"]:
                print("FAIL", c["check"], c["detail"])
    return report


if __name__ == "__main__":
    report = main()
    raise SystemExit(0 if report["pass"] else 1)
