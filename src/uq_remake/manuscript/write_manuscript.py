"""Write main manuscript and SI from registered tables. No hardcoded stale values.

SUPERSEDED FOR PROSE. The submitted text is hand-maintained in
manuscripts/final/main_manuscript_iz.md and manuscripts/final/supplementary_information.md.
This module still reproduces a generated draft from the registered tables, but it now
writes it to manuscripts/final/_generated/ so the curated text is never overwritten;
set UQ_REMAKE_OVERWRITE_PROSE=1 only if you intend to discard the curated version.
The claim registry it emits is still canonical and is written to its usual paths.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from uq_remake import REMAKE_ROOT
from uq_remake.downstream.protocol import PAIR_ACQ_EVENT_MIN_EXCLUSIVE, family_display_name, pair_acq_event_elig_text, score_display_name
from uq_remake.manuscript.figure_specs import FIGURE_SPECS
from uq_remake.manuscript.markdown import fmt_msd as _fmt_msd, md_table, note_block, pair_acq_compact_md
from uq_remake.manuscript.write_main_short import build_main_markdown

from uq_remake.manuscript.manuscript_paths import (
    AD_HOC_SELECTIVITY,
    CHEMPROP_DIAGNOSTICS,
    CHEMPROP_STATUS,
    CMI_OFFICIAL_SPLIT,
    CMI_OVERLAP,
    CMI_REPRODUCTION,
    HPLUS_DECISION,
    MAN,
    MSRC,
    NUMBERS_JSON,
    PAIR_ACQUISITION,
    PAIR_ACQ_SENS,
    PAIR_DENSITY_SUMMARY,
    PAIR_EVENT,
    PREPARE_DATA_FINDINGS,
    SIGN_STABILITY_SUMMARY,
    TABLES,
)


def source_note(canonical: str, rendering: str | None = None) -> str:
    if rendering:
        return f"Source: `{canonical}`; manuscript rendering copy: `{rendering}`."
    return f"Source: `{canonical}`."


def s10_score_label(family: str, score_name: str) -> str:
    fam = str(family)
    sc = str(score_name)
    if fam == "chemprop" and sc == "sigma_total":
        return r"Chemprop native UQ ($\sigma_{\mathrm{total}}$)"
    return score_display_name(sc)


def si_figure(sid: str, title: str, filename: str, note: str = "") -> list[str]:
    lines = [f"**Supplementary Figure {sid}. {title}**", ""]
    if note:
        lines.extend([note_block(note), ""])
    lines.extend([f"![Supplementary Figure {sid}](figures/{filename})", ""])
    return lines


def write_si_s11(s, exp) -> None:
    s("## S11. Additional diagnostics and limitations")
    s("")
    s("This section consolidates analyses that are not computed, excluded, or not identifiable. It answers which requested diagnostics cannot support a scientific claim. Completed experiments are listed in S12 rather than repeated here.")
    s("")
    s("The following experiments have **no canonical completed result** on the corrected grid. Older manuscript numbers are not reused. Composition counts such as rare-earth-element ∩ high-selectivity rows are not a combined CV estimand. Replicate-spread versus uncertainty, and electrolyte-seen / ligand–medium-seen strata, are `not_identifiable` under ligand-disjoint splits (Table S12.2); the metal–medium slice in Table S8.2 is a small-slice diagnostic, not a medium-extrapolation result.")
    s("")
    s("The following analyses were not performed and have no canonical completed result. They are reported as `not_computed`.")
    s("")
    nc = exp[exp.status.astype(str).isin(["not_computed", "excluded", "not_identifiable"])].copy() if "status" in exp.columns else exp
    nc = nc.fillna("")
    if "experiment_id" in nc.columns:
        nc = nc.sort_values(["status", "experiment_id"])
    nc_cols = [c for c in ["experiment_id", "scientific_question", "status", "known_limitations"] if c in nc.columns]
    s("**Table S11.1. Analyses not reported as completed results.**")
    s(md_table(nc, nc_cols))
    s("")


def write_si_s12(s, exp, status) -> None:
    s("## S12. Reproducibility and artifact inventory")
    s("")
    s("This section lists canonical artifacts and figure files. Path prose is reduced to this inventory; earlier sections use short source notes. The remaining section is a restricted diagnostic (S13). It does not reopen the primary all-pair evaluation.")
    s("")
    s("Canonical pipeline entrypoint: `src/uq_remake/run_full_pipeline.py`. This manuscript rebuild does not retrain; full retraining is supported via `run_full_pipeline.py` (or `scripts/run_pipeline.py`). Environment: Python 3.11, pinned RDKit 2023.09.6 and Chemprop 1.6.1. Grid: `shared_ds1_ds2_uq_v3`.")
    s("")
    s("**Table S12.1. Canonical manuscript artifact roles.** Paths below are the files used for the displayed figures and tables. Companion files are listed only when they have a distinct role.")
    s(
        md_table(
            pd.DataFrame(
                [
                    {"role": "Representative direct-UQ model context", "canonical artifact": "results/manuscript_final/workhorse_selection_accuracy_calibration.csv", "used for": "Figure 2"},
                    {"role": "Ad hoc versus direct selectivity summary", "canonical artifact": "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv", "used for": "Supplementary Figure S18; Table S13.6; Section S13"},
                    {"role": "Ad hoc versus direct fold-level metrics", "canonical artifact": "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_fold_metrics.csv", "used for": "Table S13.7"},
                    {"role": "All-pair retention", "canonical artifact": "results/downstream_cv/ad_retention/fold_retention.csv", "used for": "Figure 3; Table 3; Supplementary Figure S11"},
                    {"role": "Fold-mean retention curves (plotted)", "canonical artifact": "results/downstream_cv/ad_retention/retention_summary.csv", "used for": "Figure 3 / S11 plotted means"},
                    {"role": "Derived retention-curve AUC", "canonical artifact": "results/downstream_cv/ad_retention/retention_auc_summary.csv", "used for": "Tables S7.1, S8.1, S13.3"},
                    {"role": "Molecular-distance score versus residual", "canonical artifact": "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv", "used for": "Table S8.4"},
                    {"role": "50% retention summary", "canonical artifact": "results/downstream_cv/ad_retention/retention_50pct.csv", "used for": "Table 3; Table S13.4"},
                    {"role": "Held-out pair-event support (pooled pairs)", "canonical artifact": "results/downstream_cv/pair_event_support/pair_event_support_pooled.csv", "used for": "Supplementary Figure S13"},
                    {"role": "Held-out pair-event support (summary counts)", "canonical artifact": "results/downstream_cv/pair_event_support/pair_event_support_summary.csv", "used for": "Section 3.6"},
                    {"role": "Most frequent held-out pairs", "canonical artifact": "results/downstream_cv/pair_event_support/pair_event_support_frequent.csv", "used for": "Table S8.5"},
                    {"role": "Sigma-only pair acquisition", "canonical artifact": "results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv", "used for": "Figure 5; Table 4; Tables S10.1–S10.4"},
                    {"role": "Magnitude/error calibration", "canonical artifact": "results/downstream_cv/pair_acquisition/pair_acquisition_calibration_bin_summary.csv", "used for": "Figure 4; Tables S10.5–S10.7"},
                    {"role": "True hit@k and event precision@k", "canonical artifact": "results/downstream_cv/pair_acquisition/pair_acquisition_hit_and_precision.csv", "used for": "Table 4; Table S10.3; Figure 5"},
                    {"role": "Retrospective regret@k", "canonical artifact": "results/downstream_cv/pair_acquisition/pair_acquisition_regret.csv", "used for": "Figure 5; Table 4; Table S10.4"},
                    {"role": "Pair-acquisition metric migration note", "canonical artifact": "results/downstream_cv/pair_acquisition/pair_acquisition_metric_migration.json", "used for": "legacy hit_rate is event precision; true hit is any_hit"},
                    {"role": "Magnitude-adjusted risk ordering", "canonical artifact": "results/downstream_cv/pair_acquisition/magnitude_adjusted_risk_summary.csv", "used for": "Section 3.5; Table S10.7; Figure 4"},
                    {"role": "Pair-acquisition event-threshold sensitivity (summary)", "canonical artifact": "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv", "used for": "Supplementary Figure S15"},
                    {"role": "Pair-acquisition event-threshold sensitivity (hit/precision)", "canonical artifact": "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_hit_and_precision.csv", "used for": "Table S10.8"},
                    {"role": "Pair-acquisition event-threshold sensitivity (regret)", "canonical artifact": "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_regret.csv", "used for": "Table S10.8"},
                    {"role": "All-pair retention-analysis validation", "canonical artifact": "results/manuscript_final/ad_retention_validation.json", "used for": "retention-analysis validation"},
                ]
            ),
            ["role", "canonical artifact", "used for"],
        )
    )
    s("")
    s("**Table S12.2. Experiment completeness.** `computation_status` records whether the artifact exists; `interpretation_status` records scientific interpretability (`canonical` or `supporting` only when a completed or partial result is reported). The retention study is fold-local empirical error-ordering (`ad_retention_study`) on the canonical 10–100% grid, not a coverage guarantee.")
    exp_disp = exp.fillna("").copy()
    if "experiment_id" in exp_disp.columns:
        exp_disp = exp_disp[~exp_disp.experiment_id.astype(str).isin({"fold_local_retention_error", "fold_local_risk_coverage", "topk_directional_screening"})].copy()
        exp_disp = exp_disp.sort_values("experiment_id")
        artifact_override = {
            "ad_retention_study": "results/downstream_cv/ad_retention/fold_retention.csv",
            "adhoc_differencing": "results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_summary.csv",
            "sigma_pair_acquisition": "results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv",
            "pair_acquisition_threshold_2_half": "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_summary.csv",
        }
        if "canonical_artifact" in exp_disp.columns:
            exp_disp["canonical_artifact"] = [
                artifact_override.get(str(eid), art)
                for eid, art in zip(exp_disp["experiment_id"], exp_disp["canonical_artifact"])
            ]
        lim_override = {
            "ad_retention_study": "canonical 10–100% fold-local study; plotted means in retention_summary.csv; AUC in retention_auc_summary.csv; 50% table in retention_50pct.csv",
            "adhoc_differencing": "aligned DS2-fold comparison; independent-error propagation of two absolute-binding predictive standard deviations; restricted formulation diagnostic; not a pair-acquisition study",
            "sigma_pair_acquisition": "n_event>" + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)) + "; sigma-only ranking; event precision@k and true hit@k; distance scores not used as ranking inputs; not prospective experimental validation",
            "pair_acquisition_threshold_2_half": "supporting sensitivity; high-selectivity event y>2; near-zero event $|y|<0.5$; does not replace Figure 5 or Table 4",
            "medium_ligand_medium_strata": "electrolyte_seen and ligand_medium_seen empty under ligand-disjoint split; metal-medium seen/unseen is computed in S8.2 as a small-slice diagnostic",
        }
        if "known_limitations" in exp_disp.columns:
            exp_disp["known_limitations"] = [
                lim_override.get(str(eid), lim)
                for eid, lim in zip(exp_disp["experiment_id"], exp_disp["known_limitations"])
            ]
    exp_cols = [c for c in ["experiment_id", "computation_status", "interpretation_status", "status", "canonical_artifact", "aggregation_method", "known_limitations"] if c in exp_disp.columns]
    s(md_table(exp_disp, exp_cols))
    s("")
    s("**Table S12.3. Model status excerpt (DS2).**")
    status_cols = [c for c in ["display_name", "model_id", "overall_status", "uq_status", "interval_status", "pair_metric_status", "native_uq_status", "invalid_native_nig_rows"] if c in status.columns]
    s(md_table(status[status.dataset == "DS2"], status_cols))
    s("")
    s("Exact GPR exclusion artifacts: `checkpoints/active/DS1_RDKIT_GPR/` and `DS2_RDKIT_GPR/` `infeasible.json`. Chemprop partial UQ: `results/cv_metrics/chemprop_evidential_status_v2.csv`. Claim registry: `manuscripts/final/claim_registry.csv`. Figures: `manuscripts/final/figures/` generated from `manuscripts/final/figure_data_registry.csv`. All-pair retention-analysis validation: `results/manuscript_final/ad_retention_validation.json`.")
    s("")
    def _fig_inv_key(spec: dict) -> tuple[int, int]:
        fid = spec["figure_id"]
        digits = "".join(ch for ch in fid.split("_", 1)[0][3:] if ch.isdigit())
        return (0 if spec.get("panel") == "main" else 1, int(digits or "99"))

    fig_inv = pd.DataFrame(
        [
            {
                "figure_id": spec["figure_id"],
                "panel": spec.get("panel", ""),
                "purpose": spec["purpose"],
                "png": f"manuscripts/final/figures/{spec['figure_id']}.png",
            }
            for spec in sorted(FIGURE_SPECS, key=_fig_inv_key)
        ]
    )
    s("**Table S12.4. Figure inventory.** Every generated figure is listed. Main-text figures are numbered 1–5 and use filenames `fig01_`–`fig05_`. Supplementary figures are numbered S1–S19 in document order; filenames match those numbers. Figures S16–S19 are intentionally grouped in Section S13.4 because they belong to the restricted rare-earth-element pair diagnostic rather than to the primary all-pair analysis.")
    s(md_table(fig_inv, ["figure_id", "panel", "purpose", "png"]))
    s("")


def exclude_ree_scope(df: pd.DataFrame, col: str = "scope") -> pd.DataFrame:
    """Keep early-SI tables all-pair unless a restricted diagnostic is being written."""
    if df is None or getattr(df, "empty", True) or col not in getattr(df, "columns", []):
        return df
    return df[df[col].astype(str) != "ree_pairs"].copy()


def ds1_chemprop_point_mae(scope: str = "metal_complex") -> dict:
    """Fold-mean metal-complex MAE for ligand-disjoint DS1 Chemprop point.

    MAE is not a published DS1 headline (those tables use RMSE) but is required
    for Table 1, which compares against CMI-reported MAE.
    """
    root = REMAKE_ROOT / "results" / "cv_predictions" / "fold_test_predictions" / "DS1_GRAPH_DMPNN_POINT"
    vals = []
    n_rows = []
    for fold in range(5):
        path = root / f"fold{fold}" / "fold_test_predictions.csv"
        if not path.exists():
            return {"display": "NA", "fold_mean": None, "fold_sd": None, "n_valid_folds": 0, "fold_values": []}
        df = pd.read_csv(path, usecols=["y_true", "y_pred", "family"])
        if scope != "all":
            df = df[df.family.astype(str) == scope]
        y = pd.to_numeric(df["y_true"], errors="coerce")
        pred = pd.to_numeric(df["y_pred"], errors="coerce")
        mask = y.notna() & pred.notna()
        vals.append(float((y[mask] - pred[mask]).abs().mean()))
        n_rows.append(int(mask.sum()))
    mean = float(pd.Series(vals).mean())
    sd = float(pd.Series(vals).std(ddof=1))
    return {
        "display": _fmt_msd(mean, sd, 5),
        "fold_mean": mean,
        "fold_sd": sd,
        "n_valid_folds": 5,
        "fold_values": vals,
        "n_rows_per_fold": n_rows,
        "scope": scope,
        "model_id": "DS1_GRAPH_DMPNN_POINT",
        "aggregation": "fold_mean_plus_sd_ddof1",
        "source": "results/cv_predictions/fold_test_predictions/DS1_GRAPH_DMPNN_POINT/",
    }


def ad50_val(df: pd.DataFrame, dataset: str, model_id: str, score: str, metric: str, field: str = "retained", scope: str | None = None) -> str:
    if df is None or df.empty:
        return "NA"
    hit = df[
        (df.dataset.astype(str) == dataset)
        & (df.model_id.astype(str) == model_id)
        & (df.score_name.astype(str) == score)
        & (df.metric_name.astype(str) == metric)
    ]
    if "scope" in df.columns:
        want = scope
        if want is None:
            if dataset == "DS1":
                want = "metal_complex"
            elif dataset == "DS2":
                want = "all_pairs"
        if want:
            scoped = hit[hit.scope.astype(str) == want]
            hit = scoped if len(scoped) else hit
    if hit.empty:
        return "NA"
    r = hit.iloc[0]
    n = int(r.n_valid_folds) if "n_valid_folds" in hit.columns and pd.notna(r.n_valid_folds) else 5
    if field == "full":
        return _fmt_msd(r.full_data_mean, None, n)
    if field == "rel":
        if pd.isna(r.relative_change):
            return "NA"
        return f"{100.0 * float(r.relative_change):+.1f}%"
    if field == "pairs":
        mean = getattr(r, "pairs_contributing_to_MoP_mean", getattr(r, "pairs_retained_mean", None))
        sd = getattr(r, "pairs_contributing_to_MoP_sd", getattr(r, "pairs_retained_sd", None))
        return _fmt_msd(mean, sd, n)
    if field == "n_rows":
        return _fmt_msd(getattr(r, "n_rows_retained_mean", None), getattr(r, "n_rows_retained_sd", None), n)
    return _fmt_msd(r.retained_mean, r.retained_sd, n)


# --- Curated-prose guard -------------------------------------------------------
# The submitted text (manuscripts/final/main_manuscript_iz.md and
# supplementary_information.md) is hand-maintained and has diverged from this
# generator: it carries the four-figure numbering, the reference list, and the
# Methods/SI corrections made after this module was last updated. Regenerating
# over those files would silently discard that work, so prose output goes to
# manuscripts/final/_generated/ unless UQ_REMAKE_OVERWRITE_PROSE=1 is set.
# Use the generated copies for diffing, not as the submission source.
PROSE_OVERWRITE_ENV = "UQ_REMAKE_OVERWRITE_PROSE"


def _prose_out_dir():
    if os.environ.get(PROSE_OVERWRITE_ENV, "") == "1":
        return MAN
    out = MAN / "_generated"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    numbers = json.loads((MSRC / "numbers.json").read_text(encoding="utf-8"))
    ds1p = pd.read_csv(TABLES / "main_ds1_point_results.csv")
    ds1u = pd.read_csv(TABLES / "main_ds1_uq_results.csv")
    wh = pd.read_csv(TABLES / "main_ds2_operational_comparison.csv")
    ree = pd.read_csv(TABLES / "ds2_ree_pairs.csv")
    allp = pd.read_csv(TABLES / "ds2_all_pairs.csv")
    hsrow = pd.read_csv(TABLES / "ds2_highsel_row.csv")
    hspair = pd.read_csv(TABLES / "ds2_highsel_pair.csv")
    pair_si = pd.read_csv(TABLES / "supplementary_pair_level_results.csv")
    row_si = pd.read_csv(TABLES / "supplementary_row_level_diagnostics.csv")
    sel_si = pd.read_csv(TABLES / "supplementary_workhorse_selection.csv")
    ad = pd.read_csv(TABLES / "main_ds2_ad_results.csv")
    pair_ad_path = TABLES / "main_ds2_ad_pair_results.csv"
    row_ad_path = TABLES / "supplementary_ds2_ad_row_results.csv"
    pair_ad_tbl = pd.read_csv(pair_ad_path) if pair_ad_path.exists() else ad
    row_ad_tbl = pd.read_csv(row_ad_path) if row_ad_path.exists() else ad
    ad50 = pd.read_csv(TABLES / "ad_retention_50pct.csv") if (TABLES / "ad_retention_50pct.csv").exists() else pd.DataFrame()
    ad_enrich = pd.read_csv(TABLES / "ad_enrichment_summary.csv") if (TABLES / "ad_enrichment_summary.csv").exists() else pd.DataFrame()
    ad_defs = pd.read_csv(TABLES / "ad_score_definitions.csv") if (TABLES / "ad_score_definitions.csv").exists() else pd.DataFrame()
    ad_aurc = pd.read_csv(TABLES / "ad_retention_aurc.csv") if (TABLES / "ad_retention_aurc.csv").exists() else pd.DataFrame()
    adhoc_cmp_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_comparison.csv"
    if not adhoc_cmp_path.exists():
        adhoc_cmp_path = TABLES / "ad_hoc_selectivity_comparison.csv"
    adhoc_cmp = pd.read_csv(adhoc_cmp_path) if adhoc_cmp_path.exists() else pd.DataFrame()
    adhoc_fold_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_fold_metrics.csv"
    adhoc_fold = pd.read_csv(adhoc_fold_path) if adhoc_fold_path.exists() else pd.DataFrame()
    adhoc_sum_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_selectivity_summary.csv"
    adhoc_sum = pd.read_csv(adhoc_sum_path) if adhoc_sum_path.exists() else pd.DataFrame()
    adhoc_snap_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_hoc_selectivity" / "ad_hoc_spec_snapshot.json"
    adhoc_snap = json.loads(adhoc_snap_path.read_text(encoding="utf-8")) if adhoc_snap_path.exists() else {}
    for name, src in (
        ("ad_hoc_selectivity_comparison.csv", adhoc_cmp_path if adhoc_cmp_path.exists() else None),
        ("ad_hoc_selectivity_summary.csv", adhoc_sum_path if adhoc_sum_path.exists() else None),
        ("ad_hoc_selectivity_fold_metrics.csv", adhoc_fold_path if adhoc_fold_path.exists() else None),
    ):
        if src is not None and src.exists():
            TABLES.joinpath(name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    mm_path = TABLES / "supplementary_metal_medium_strata.csv"
    mm = pd.read_csv(mm_path) if mm_path.exists() else pd.DataFrame()
    dens_bins_path = TABLES / "supplementary_pair_density_bins.csv"
    dens_bins = pd.read_csv(dens_bins_path) if dens_bins_path.exists() else pd.DataFrame()
    corr_path = TABLES / "ad_distance_error_corr.csv"
    if not corr_path.exists():
        corr_path = REMAKE_ROOT / "results" / "downstream_cv" / "ad_distance_error_corr" / "ad_distance_error_corr_summary.csv"
    corr = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()
    paired_path = TABLES / "supplementary_fold_paired_differences.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    mop_valid_path = TABLES / "supplementary_mop_pair_validity.csv"
    mop_valid = pd.read_csv(mop_valid_path) if mop_valid_path.exists() else pd.DataFrame()
    grid_risk_path = TABLES / "supplementary_grid_sigma_total_risk_ordering.csv"
    grid_risk = pd.read_csv(grid_risk_path) if grid_risk_path.exists() else pd.DataFrame()
    folds = pd.read_csv(TABLES / "fold_sizes.csv")
    grid = pd.read_csv(TABLES / "model_grid_final.csv")
    status = pd.read_csv(TABLES / "model_status_final.csv")
    sign = pd.read_csv(SIGN_STABILITY_SUMMARY)
    dens = pd.read_csv(PAIR_DENSITY_SUMMARY)
    if SIGN_STABILITY_SUMMARY.exists():
        TABLES.joinpath("sign_stability_summary.csv").write_text(
            SIGN_STABILITY_SUMMARY.read_text(encoding="utf-8"), encoding="utf-8"
        )
    pair_acq_root = PAIR_ACQUISITION
    for name in (
        "pair_acquisition_summary.csv",
        "pair_acquisition_comparison.csv",
        "pair_acquisition_hit_and_precision.csv",
        "pair_acquisition_metric_migration.json",
        "pair_acquisition_eligibility_summary.csv",
        "pair_acquisition_calibration_bin_summary.csv",
        "pair_acquisition_calibration_rho_summary.csv",
        "pair_acquisition_regret.csv",
        "pair_acquisition_regret_fold.csv",
        "magnitude_adjusted_risk_summary.csv",
        "magnitude_adjusted_risk_fold.csv",
    ):
        src = pair_acq_root / name
        if src.exists():
            TABLES.joinpath(name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for src_name, dest_name in (
        ("pair_acquisition_summary.csv", "pair_acquisition_tau2_half_summary.csv"),
        ("pair_acquisition_comparison.csv", "pair_acquisition_tau2_half_comparison.csv"),
        ("pair_acquisition_hit_and_precision.csv", "pair_acquisition_tau2_half_hit_and_precision.csv"),
        ("pair_acquisition_regret.csv", "pair_acquisition_tau2_half_regret.csv"),
        ("pair_acquisition_eligibility_summary.csv", "pair_acquisition_tau2_half_eligibility_summary.csv"),
    ):
        src = PAIR_ACQ_SENS / src_name
        if src.exists():
            TABLES.joinpath(dest_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    pair_acq_sum = pd.read_csv(TABLES / "pair_acquisition_summary.csv") if (TABLES / "pair_acquisition_summary.csv").exists() else pd.DataFrame()
    pair_acq_cmp = pd.read_csv(TABLES / "pair_acquisition_comparison.csv") if (TABLES / "pair_acquisition_comparison.csv").exists() else pd.DataFrame()
    pair_acq_hit = pd.read_csv(TABLES / "pair_acquisition_hit_and_precision.csv") if (TABLES / "pair_acquisition_hit_and_precision.csv").exists() else pd.DataFrame()
    pair_acq_regret = pd.read_csv(TABLES / "pair_acquisition_regret.csv") if (TABLES / "pair_acquisition_regret.csv").exists() else pd.DataFrame()
    pair_acq_elig = pd.read_csv(TABLES / "pair_acquisition_eligibility_summary.csv") if (TABLES / "pair_acquisition_eligibility_summary.csv").exists() else pd.DataFrame()
    pair_acq_bins = pd.read_csv(TABLES / "pair_acquisition_calibration_bin_summary.csv") if (TABLES / "pair_acquisition_calibration_bin_summary.csv").exists() else pd.DataFrame()
    pair_acq_rho = pd.read_csv(TABLES / "pair_acquisition_calibration_rho_summary.csv") if (TABLES / "pair_acquisition_calibration_rho_summary.csv").exists() else pd.DataFrame()
    mag_adj = pd.read_csv(TABLES / "magnitude_adjusted_risk_summary.csv") if (TABLES / "magnitude_adjusted_risk_summary.csv").exists() else pd.DataFrame()
    pair_acq_sens_hit = pd.read_csv(TABLES / "pair_acquisition_tau2_half_hit_and_precision.csv") if (TABLES / "pair_acquisition_tau2_half_hit_and_precision.csv").exists() else pd.DataFrame()
    pair_acq_sens_regret = pd.read_csv(TABLES / "pair_acquisition_tau2_half_regret.csv") if (TABLES / "pair_acquisition_tau2_half_regret.csv").exists() else pd.DataFrame()
    pair_acq_sens_elig = pd.read_csv(TABLES / "pair_acquisition_tau2_half_eligibility_summary.csv") if (TABLES / "pair_acquisition_tau2_half_eligibility_summary.csv").exists() else pd.DataFrame()
    for name in (
        "pair_event_support_pooled.csv",
        "pair_event_support_frequent.csv",
        "pair_event_support_summary.csv",
        "pair_event_support_fold.csv",
    ):
        src = PAIR_EVENT / name
        if not src.exists():
            src = MSRC / name
        if src.exists():
            TABLES.joinpath(name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    pair_event_pooled = pd.read_csv(TABLES / "pair_event_support_pooled.csv") if (TABLES / "pair_event_support_pooled.csv").exists() else pd.DataFrame()
    pair_event_frequent = pd.read_csv(TABLES / "pair_event_support_frequent.csv") if (TABLES / "pair_event_support_frequent.csv").exists() else pd.DataFrame()
    pair_event_summary = pd.read_csv(TABLES / "pair_event_support_summary.csv") if (TABLES / "pair_event_support_summary.csv").exists() else pd.DataFrame()
    chem_st = pd.read_csv(CHEMPROP_STATUS)
    if "pair_metric_status" not in chem_st.columns and "ranking_status" in chem_st.columns:
        chem_st["pair_metric_status"] = chem_st["ranking_status"]
    chem_d = pd.read_csv(CHEMPROP_DIAGNOSTICS)
    cmi = json.loads(CMI_OFFICIAL_SPLIT.read_text(encoding="utf-8"))
    cmi_repro = json.loads(CMI_REPRODUCTION.read_text(encoding="utf-8")) if CMI_REPRODUCTION.exists() else {}
    overlap_path = CMI_OVERLAP
    cmi_overlap = json.loads(overlap_path.read_text(encoding="utf-8")) if overlap_path.exists() else {}
    hplus = json.loads(HPLUS_DECISION.read_text(encoding="utf-8"))
    findings = json.loads(PREPARE_DATA_FINDINGS.read_text(encoding="utf-8"))
    exp = pd.read_csv(TABLES / "supplementary_experiment_status.csv")

    fmt = numbers["fmt"]
    ds1 = numbers["ds1"]
    ds2 = numbers["ds2"]
    sc = numbers["ds2_scope_counts"]
    ranked = numbers["ranked_detail"]

    def dens_val(scope, bin_name, metric):
        hit = dens[(dens.scope == scope) & (dens.bin == bin_name) & (dens.metric_name == metric)]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        n = int(r.n_valid_folds)
        s = f"{float(r.fold_mean):.3f} ± {float(r.fold_sd):.3f}" if pd.notna(r.fold_sd) else f"{float(r.fold_mean):.3f}"
        if n != 5:
            s += f" (n_valid_folds={n})"
        return s

    def sign_val(scope, cat, metric):
        hit = sign[(sign.scope == scope) & (sign.category == cat) & (sign.metric_name == metric)]
        r = hit.iloc[0]
        return f"{float(r.fold_mean):.3f} ± {float(r.fold_sd):.3f}"

    def ad_val(scope, score, col="pair_aurc"):
        aliases = {
            "mahalanobis_joint": "mahalanobis_feature_centroid",
            "pair_aurc": "aurc",
            "mop_rmse_at_50": "rmse_at_50pct_retention",
        }
        score = aliases.get(score, score)
        hit = ad[(ad.scope == scope) & (ad.score_name == score)]
        if hit.empty and score == "mahalanobis_feature_centroid":
            hit = ad[(ad.scope == scope) & (ad.score_name == "mahalanobis_joint")]
        if col not in hit.columns and col in aliases and aliases[col] in hit.columns:
            col = aliases[col]
        return str(hit.iloc[0][col])

    def ret_aurc(scope, score, metric):
        if ad_aurc is None or ad_aurc.empty:
            return "NA"
        aliases = {"mahalanobis_joint": "mahalanobis_feature_centroid"}
        score = aliases.get(score, score)
        hit = ad_aurc[
            (ad_aurc.dataset.astype(str) == "DS2")
            & (ad_aurc.model_id.astype(str) == numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO"))
            & (ad_aurc.scope.astype(str) == scope)
            & (ad_aurc.score_name.astype(str) == score)
            & (ad_aurc.metric_name.astype(str) == metric)
        ]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        return _fmt_msd(r.fold_mean, r.fold_sd, int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5)

    def mm_val(scope, label, metric):
        hit = mm[(mm.scope == scope) & (mm.stratum_label == label) & (mm.metric_name == metric)]
        if hit.empty:
            return "NA"
        return str(hit.iloc[0].display)

    def dens_disp(metric, bin_name):
        if dens_bins is None or getattr(dens_bins, "empty", True) or "metric_name" not in dens_bins.columns:
            return dens_val("all_pairs", bin_name, metric)
        hit = dens_bins[(dens_bins.metric_name == metric) & (dens_bins.bin == bin_name)]
        if hit.empty:
            return dens_val("all_pairs", bin_name, metric)
        return str(hit.iloc[0].display)

    def adhoc_val(model_id, method, col="mop_rmse"):
        if adhoc_cmp is None or adhoc_cmp.empty:
            return "NA"
        if col not in adhoc_cmp.columns and col == "mop_rmse" and "mop_mae" in adhoc_cmp.columns:
            col = "mop_mae"
        hit = adhoc_cmp[
            (adhoc_cmp.source_model_id.astype(str) == model_id)
            & (adhoc_cmp.selectivity_method.astype(str) == method)
        ]
        if "scope" in hit.columns:
            scoped = hit[hit.scope.astype(str) == "ree_pairs"]
            hit = scoped if len(scoped) else hit
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        sd_col = f"{col}_sd"
        n = int(r.n_valid) if "n_valid" in hit.columns and pd.notna(r.n_valid) else 5
        return _fmt_msd(r[col] if col in hit.columns else None, r[sd_col] if sd_col in hit.columns else None, n)

    def paired_val(name, col):
        hit = paired[paired.display_name == name]
        if hit.empty:
            return "NA"
        return str(hit.iloc[0][col])

    def row(df, name):
        hit = df[df.display_name == name]
        if hit.empty:
            raise KeyError(name)
        if "regime" in hit.columns:
            direct = hit[hit["regime"].astype(str).str.lower().eq("direct")]
            if len(direct):
                return direct.iloc[0]
        return hit.iloc[0]

    wh_name = numbers.get("workhorse_display", "Heteroscedastic MLP ensemble")
    wh_row = row(wh, wh_name)
    chem_row = row(wh, "Chemprop evidential")
    hetero_ens = row(wh, "Heteroscedastic MLP ensemble")
    evid = row(wh, "Evidential MLP")

    ds1_test = f"{int(folds[folds.dataset=='DS1'].n_test.min())}–{int(folds[folds.dataset=='DS1'].n_test.max())}"
    ds2_test = int(folds[folds.dataset == "DS2"].n_test.iloc[0])
    ds1_train = f"{int(folds[folds.dataset=='DS1'].n_train.min())}–{int(folds[folds.dataset=='DS1'].n_train.max())}"
    ds2_train = f"{int(folds[folds.dataset=='DS2'].n_train.min())}–{int(folds[folds.dataset=='DS2'].n_train.max())}"

    cmi_test = cmi["evaluations"]["cmi_test"]["all_records"]
    cmi_ext = cmi["evaluations"]["cmi_ext"]["all_records"]
    pub = cmi["cmi_published"]

    invalid_total = int(chem_d.n_uq_invalid.sum())
    chem_note = chem_st.iloc[0]

    grid_active = grid[grid.active_status != "excluded"]
    grid_excl = grid[grid.active_status == "excluded"]

    usable_ids = set()
    if "calibration_usable" in wh.columns:
        usable_ids = set(wh.loc[wh.calibration_usable.astype(str).str.lower().isin({"true", "1"}), "model_id"].astype(str))
    usable_ids.add(str(numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO")))
    adhoc_table = ""
    adhoc_table_full = ""
    adhoc_table_brief = ""

    def _adhoc_md_rows(frame):
        rows = []
        rmse_col = "mop_rmse" if "mop_rmse" in frame.columns else "mop_mae"
        rmse_sd_col = "mop_rmse_sd" if "mop_rmse_sd" in frame.columns else "mop_mae_sd"
        for _, r in frame.iterrows():
            n = int(r.n_valid) if "n_valid" in frame.columns and pd.notna(r.n_valid) else 5
            rows.append(
                {
                    "method class": str(r.method_class).replace("direct_ds2", "direct DS2").replace("ad_hoc_ds1_uncertainty", "ad hoc DS1 UQ").replace("ad_hoc_ds1_point", "ad hoc DS1 point"),
                    "model": r.source_model,
                    "MoP RMSE": _fmt_msd(r[rmse_col], r[rmse_sd_col] if rmse_sd_col in frame.columns else None, n),
                    "MoP $R^2$": _fmt_msd(r.mop_r2, r.mop_r2_sd if "mop_r2_sd" in frame.columns else None, n),
                    "sign accuracy": _fmt_msd(r.sign_accuracy, r.sign_accuracy_sd if "sign_accuracy_sd" in frame.columns else None, n),
                }
            )
        return rows

    if len(adhoc_cmp):
        show = adhoc_cmp.copy()
        if "scope" in show.columns:
            show = show[show.scope.astype(str) == "ree_pairs"]
        keep = []
        for _, r in show.iterrows():
            cls = str(r.get("method_class", ""))
            mid = str(r.source_model_id)
            if cls == "direct_ds2" and mid in usable_ids:
                keep.append(True)
            elif cls in {"ad_hoc_ds1_uncertainty", "ad_hoc_ds1_point"}:
                keep.append(True)
            else:
                keep.append(False)
        show = show.loc[keep].copy()
        rows = _adhoc_md_rows(show)
        if rows:
            adhoc_table_full = md_table(pd.DataFrame(rows), list(rows[0].keys()))
        main_ids = {
            str(numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO")),
            "DS2_RDKIT_MLPENS5_EVID",
            "DS1_GRAPH_DMPNN_EVID",
            "DS1_RDKIT_MLPENS5_EVID",
            "DS1_RDKIT_XGB_POINT",
        }
        main_show = show[show.source_model_id.astype(str).isin(main_ids)].copy()
        class_ord = {"direct_ds2": 0, "ad_hoc_ds1_uncertainty": 1, "ad_hoc_ds1_point": 2}
        if len(main_show):
            main_show["_ord"] = main_show.method_class.astype(str).map(class_ord).fillna(9)
            metric = "mop_rmse" if "mop_rmse" in main_show.columns else "mop_mae"
            main_show = main_show.sort_values(["_ord", metric]).drop(columns=["_ord"])
            main_rows = _adhoc_md_rows(main_show)
            if main_rows:
                adhoc_table = md_table(pd.DataFrame(main_rows), list(main_rows[0].keys()))
        if not adhoc_table:
            adhoc_table = adhoc_table_full
        brief_ids = {
            str(numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO")),
            "DS1_GRAPH_DMPNN_EVID",
        }
        brief_show = show[show.source_model_id.astype(str).isin(brief_ids)].copy()
        if len(brief_show):
            brief_show["_ord"] = brief_show.method_class.astype(str).map(class_ord).fillna(9)
            brief_show = brief_show.sort_values(["_ord"]).drop(columns=["_ord"])
            brief_rows = _adhoc_md_rows(brief_show)
            if brief_rows:
                adhoc_table_brief = md_table(pd.DataFrame(brief_rows), list(brief_rows[0].keys()))
        if not adhoc_table_brief:
            adhoc_table_brief = adhoc_table
    from uq_remake.downstream.ad_hoc_selectivity import ad_hoc_comparison_phrase

    adhoc_phrase = ad_hoc_comparison_phrase(adhoc_cmp) if len(adhoc_cmp) else (adhoc_snap.get("comparison_phrase") or "was compared with")
    adhoc_direct_wh = adhoc_val(str(numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO")), "direct_ds2_selectivity", "mop_rmse")
    adhoc_best_uq = adhoc_val("DS1_GRAPH_DMPNN_EVID", "ad_hoc_ds1_difference", "mop_rmse")
    adhoc_best_uq_sign = adhoc_val("DS1_GRAPH_DMPNN_EVID", "ad_hoc_ds1_difference", "sign_accuracy")
    adhoc_evid_ens = adhoc_val("DS1_RDKIT_MLPENS5_EVID", "ad_hoc_ds1_difference", "mop_rmse")
    chem_ds1_mae_info = ds1_chemprop_point_mae("metal_complex")
    chem_ds1_mae = chem_ds1_mae_info["display"]
    numbers.setdefault("fmt", {})["ds1_chemprop_mae"] = chem_ds1_mae
    numbers["ds1_chemprop_point_metal_complex_mae"] = {
        "fold_mean": chem_ds1_mae_info["fold_mean"],
        "fold_sd": chem_ds1_mae_info["fold_sd"],
        "n_valid_folds": chem_ds1_mae_info["n_valid_folds"],
        "scope": "metal_complex",
    }
    (REMAKE_ROOT / "results" / "canonical" / "ds1_chemprop_ligand_disjoint_mae.json").write_text(
        json.dumps(chem_ds1_mae_info, indent=2, default=str),
        encoding="utf-8",
    )
    NUMBERS_JSON.write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")

    ctx = dict(locals())
    ctx["ad50_val"] = ad50_val
    main = build_main_markdown(ctx)
    _prose_out_dir().joinpath("main_manuscript.md").write_text("\n".join(main) + "\n", encoding="utf-8")

    si = []
    s = si.append
    s("# Supplementary Information")
    s("")
    s("## Supplementary Information for")
    s("")
    s("# From molecular similarity to prediction risk: uncertainty-aware ligand prioritization for metal–metal selectivity")
    s("")
    s("All numerical tables are generated from registered canonical artifacts. Headline entries of the form mean ± standard deviation (SD) are fold-level summaries (`ddof=1`) unless a row is explicitly marked partial, pooled, historical, excluded, or not computed. Aggregation conventions are defined once in Section S2 and are not restated before every table. The source collection is the Critical Materials Institute (CMI) literature M4 collection; DS1 and DS2 denote the absolute-binding and pairwise-selectivity representations, respectively. The main text uses mean-of-pairs (MoP) root mean squared error (RMSE) as the primary pair-level accuracy metric. This supplement provides inventories, fold-level values, and analyses marked not computed or not identifiable. Displayed supplementary figures are numbered S1–S19 in document order. Figures S16–S19 are intentionally grouped in Section S13.4 because they belong to the restricted rare-earth-element pair diagnostic rather than to the primary all-pair analysis.")
    s("")
    s("## S1. Data, targets, and split design")
    s("")
    s("This section records literature M4, the derived DS1 and DS2 representations, feature schemas, ligand-disjoint split sizes, and the historical CMI assignment. It answers what was modeled and how generalization was defined. It supports the main-text claim that official-split and ligand-disjoint protocols are different estimands. Restricted chemical-subset counts are deferred to Section S13.")
    s("")
    s("Literature M4 is the heterogeneous source collection of reported binding measurements. DS1 is the absolute-binding modeling representation derived from normalized M4 records: under `train_with_H`, protonation records are retained, each record keeps its reported $\\log K$, and CMI external rows are excluded. DS2 is the pairwise selectivity representation: metal-complex records are aggregated by median $\\log K$ on the species key, then paired when two metals share ligand, temperature, ionic strength, and electrolyte. Internal aleatoric variability among nominally matching records is described in main-text Methods 2.1; it is not treated as a completed replicate-versus-uncertainty experiment (Table S12.2).")
    s("")
    s("**Table S1.1. DS1 composition.** Source: `results/canonical/ds1_composition.json`.")
    s("")
    s("| Quantity | Value |")
    s("| --- | ---: |")
    s(f"| Total records | {ds1['n_records']:,} |")
    s(f"| Unique ligands | {ds1['n_unique_ligands']:,} |")
    s(f"| Metals | {ds1['n_metals']} |")
    s(f"| Species keys | {ds1['n_species_keys']:,} |")
    s(f"| Metal-complex records | {ds1['family_breakdown']['metal_complex']:,} |")
    s(f"| Protonation records | {ds1['family_breakdown']['protonation']:,} |")
    s(f"| CMI M4 external excluded (records / ligands) | {ds1['n_external_records']} / {ds1['n_external_ligands']} |")
    s(f"| H$^+$ policy | `{ds1['hplus_decision']}` |")
    s(f"| RDKit dimension | {ds1['rdkit_dimension']} |")
    s("")
    s("**Table S1.2. DS2 composition.** Source: `results/canonical/ds2_composition.json` and `data/processed/ds2.csv`.")
    s("")
    s("| Quantity | Value |")
    s("| --- | ---: |")
    s(f"| Rows | {ds2['n_pairs_rows']:,} |")
    s(f"| Unique ligands | {ds2['n_unique_ligands']:,} |")
    s(f"| Species-level pairs | {ds2['n_metal_species_pairs']:,} |")
    s(f"| Element pairs | {ds2['n_metal_element_pairs']:,} |")
    s(f"| Metal–H pairs | {ds2['n_metal_H_pairs']} |")
    s(f"| High-selectivity rows ($|y|>1$) | {sc['n_highsel_rows']:,} |")
    s(f"| Pair identifier | `metal_species_pair` (e.g. `{sc['pair_id_example']}`) |")
    s("")
    s("**Table S1.3. Canonical feature schemas.**")
    s("")
    s("| Family | Construction | Dimension | Imputation |")
    s("| --- | --- | ---: | --- |")
    s(f"| RDKit DS1 | {numbers['rdkit_n_selected']} 2D descriptors + 9 metal props + charges + temperature/ionic-strength features + solvent MW/density | {numbers['rdkit_ds1_dim']} | none |")
    s(f"| RDKit DS2 | same ligand block + paired metal/charge columns + temperature/ionic-strength features + solvent MW/density | {numbers['rdkit_ds2_dim']} | none |")
    s(f"| Chemprop DS1 | complex graph + {numbers['chemprop_ds1_extra']} additional continuous features | graph + {numbers['chemprop_ds1_extra']} | none in production CV |")
    s(f"| Chemprop DS2 | ligand graph + {numbers['chemprop_ds2_extra']} additional continuous features | graph + {numbers['chemprop_ds2_extra']} | none in production CV |")
    s("| CPEMB | excluded from active grid | — | — |")
    s("")
    s("Electrolyte identity is used for matching and grouping only. RDKit 23 descriptors were dropped (names in `data/processed/audits/prepare_data_findings.json`).")
    s("")
    s("**Table S1.4. Outer-fold sizes from RDKit manifests.**")
    s(note_block("These are partition sizes, not metrics. Source = `manuscripts/final/tables/fold_sizes.csv`."))
    s(md_table(folds, ["dataset", "fold_id", "n_train", "n_val", "n_test", "feature_dimension"]))
    s("")
    s("Held-out DS1 metal-complex row counts on the RDKit MLP (from `fold_metrics.csv`) are 6,250 / 6,372 / 6,396 / 6,473 / 6,538. DS2 test folds are uniformly 17,705 rows. Workhorse all-pair MoP retained-pair counts are 870 / 729 / 812 / 760 / 837.")
    s("")
    s("## S1.5 Preprocessing and validation protocol")
    s("")
    s("This subsection records the ligand-disjoint grouped $K$-fold cross-validation (GroupKFold) protocol and the historical CMI split. It is supporting documentation for Methods 2.1–2.2.")
    s("")
    s("### S1.5.1 Active preprocessing and validation protocol")
    s("")
    s("Splits are ligand-disjoint five-fold GroupKFold (`data/splits/project_folds.json`, `ds2_project_folds.json`). Inner validation is a ligand-disjoint subset of the outer-training ligands. Fold-local StandardScaler is fit on outer-train features only. No median imputation is used in the production CV pipelines. GPR_SPARSE inverts an internal target scaler to native units; other families use `none_native_units`. Headline metrics are five-fold means ± SD of held-out-fold summaries, not pooled-row estimates.")
    s("")
    s("### S1.5.2 Historical CMI-split verification")
    s("")
    s("Karunaratne et al. (JCIM 2025; https://doi.org/10.1021/acs.jcim.5c01546) remove a small ligand-held-out external set and then randomly divide the remaining rows. Published Table 1 **test** metrics are five-fold CV averages on that assignment, a different estimand from a single open test-file evaluation. That random assignment is not ligand-disjoint and should not be read as an applicability-domain CV. A fingerprint-similarity partition is a split-design choice rather than an applicability-domain validation. We did not perform a direct Tanimoto-stratified versus ligand-disjoint split comparison; the present conclusions about Tanimoto concern its tested residual-ordering behavior (Table S8.4).")
    s("")
    n_shared = cmi_overlap.get("n_ligands_shared_train_test", "NA")
    n_test_lig = cmi_overlap.get("n_test_ligands", "NA")
    frac_lig = cmi_overlap.get("frac_test_ligands_seen_in_train")
    frac_row = cmi_overlap.get("frac_test_rows_with_train_ligand")
    frac_lig_s = f"{100.0 * float(frac_lig):.1f}%" if frac_lig is not None else "NA"
    frac_row_s = f"{100.0 * float(frac_row):.1f}%" if frac_row is not None else "NA"
    s("**Table S1.5. Ligand leakage on the official CMI assignment versus project folds.**")
    s(
        note_block(
            "Shared ligand = the same `ligand_smiles` appears in both partitions, typically with a different metal or medium. Project folds are ligand-disjoint by construction. Source: `results/canonical/cmi_split_ligand_overlap.json`."
        )
    )
    s(
        md_table(
            pd.DataFrame(
                [
                    {"partition pair": "official train vs official test", "shared ligands": str(n_shared), "test ligands": str(n_test_lig), "fraction of test ligands shared": frac_lig_s, "fraction of test rows with a train ligand": frac_row_s},
                    {"partition pair": "official train vs official external", "shared ligands": str(cmi_overlap.get("n_ligands_shared_train_ext", 0)), "test ligands": str(cmi_overlap.get("n_ext_ligands", 40)), "fraction of test ligands shared": "0%", "fraction of test rows with a train ligand": "0%"},
                    {"partition pair": "project fold train vs held-out (all five folds)", "shared ligands": "0", "test ligands": "869–870 / fold", "fraction of test ligands shared": "0%", "fraction of test rows with a train ligand": "0%"},
                ]
            ),
            ["partition pair", "shared ligands", "test ligands", "fraction of test ligands shared", "fraction of test rows with a train ligand"],
        )
    )
    s("")
    s("**What retrain means, and what the released-weight experiment is.** Table 1 now reports three official-split Chemprop protocols. (1) The shared-architecture rows train a blank Chemprop 1.6.1 D-MPNN **point** model (`M4_GRAPH_DMPNN_POINT_cmisplit`) from random initialization on the published CMI train indices, with a ligand-disjoint 15% early-stopping holdout carved from train (`configs/shared_dmpnn.json`: hidden size 300, depth 3, two feed-forward layers of size 300, dropout 0, 50 epochs, batch size 512). Those rows do **not** load CMI `.pt` files. (2) The released-weight rows load the published five-fold `best_model` checkpoints from https://github.com/Critical-Materials-Institute/stabilityconstant-ml-models and score the official CMI test and external CSVs with Chemprop 1.6.1. (3) The CMI-architecture rows retrain from random initialization using the hidden size, depth, feed-forward width, batch size, epoch count, and seed recovered from those checkpoints (Table S1.6), without loading the published weights.")
    s("")
    s("**Software and input relative to the CMI GitHub repository.** Shared-architecture and architecture-matched retrains used Chemprop 1.6.1. Karunaratne et al. trained with Chemprop 1.5.2. The released-weight evaluation uses the same 1.6.1 runtime to load the 1.5.2 checkpoints. The graph input is the published complex SMILES (`cmplx_smiles_neutral_std` on the CMI train/test CSVs; `SMILES` on the external file) together with the same **16** additional continuous columns shipped as CMI `train_features.csv` / `test_features.csv` (Table S1.3 Chemprop DS1). That input matches the CMI additional-feature D-MPNN (`best_model`), not the README SMILES-only example. Recovered checkpoint arguments give hidden size 2400, depth 6, three feed-forward layers of size 2400, dropout 0, 30 epochs, batch size 50, and fold seeds 0–4. Predictions from released weights are reported both as the **mean of five fold-wise metrics** on the open test file (the closer analogue of the published test CV average) and as the **five-fold ensemble** used for the published external protocol.")
    s("")
    s("**Table S1.6. Official-split Chemprop protocol versus the CMI GitHub M4 release.**")
    s(
        note_block(
            "Architecture values for CMI are read from the released `best_model` checkpoints, not from Table S1 of the paper. Shared-architecture retrains use `configs/shared_dmpnn.json`. Source: `results/stage1/cmi_released_architecture.json`, `src/uq_remake/data/run_cmisplit.py`, and https://github.com/Critical-Materials-Institute/stabilityconstant-ml-models."
        )
    )
    s(
        md_table(
            pd.DataFrame(
                [
                    {"item": "Chemprop version", "this work (shared retrain)": "1.6.1", "CMI paper / released weights": "trained with 1.5.2; scored here with 1.6.1"},
                    {"item": "Weights", "this work (shared retrain)": "trained from random initialization", "CMI paper / released weights": "five-fold `best_model` `.pt` (evaluated in Table 1)"},
                    {"item": "Graph SMILES", "this work (shared retrain)": "published complex SMILES", "CMI paper / released weights": "same (`cmplx_smiles_neutral_std` / external `SMILES`)"},
                    {"item": "Additional features", "this work (shared retrain)": "16 CMI continuous columns", "CMI paper / released weights": "same for `best_model`; none for `SMILES_only_model`"},
                    {"item": "Hidden size / depth / FFN", "this work (shared retrain)": "300 / 3 / two layers of 300", "CMI paper / released weights": "2400 / 6 / three layers of 2400"},
                    {"item": "Batch size / epochs / dropout", "this work (shared retrain)": "512 / 50 / 0", "CMI paper / released weights": "50 / 30 / 0"},
                    {"item": "Training protocol", "this work (shared retrain)": "one model; 15% ligand-disjoint val from official train", "CMI paper / released weights": "five-fold CV on train (fold seeds 0–4)"},
                    {"item": "Official test metric in Table 1", "this work (shared retrain)": "one open test-file evaluation", "CMI paper / released weights": "paper: five-fold CV average; this work: mean of 5 fold-wise metrics"},
                    {"item": "External metric in Table 1", "this work (shared retrain)": "same single model", "CMI paper / released weights": "five-fold ensemble"},
                ]
            ),
            ["item", "this work (shared retrain)", "CMI paper / released weights"],
        )
    )
    s("")
    s("**Table S1.7. Official-file Chemprop MAE and $R^2$ under released weights.**")
    wt_block = (cmi_repro.get("released_weights") or {}) if isinstance(cmi_repro, dict) else {}
    wt_eval = wt_block.get("evaluations") or {}
    wt_test_b = wt_eval.get("cmi_test") or {}
    wt_ext_b = wt_eval.get("cmi_ext") or {}
    wt_test_ens_b = wt_test_b.get("ensemble") or {}
    wt_ext_ens_b = wt_ext_b.get("ensemble") or {}
    s(
        note_block(
            "Ensemble = mean of the five fold predictions. Fold-mean = unweighted mean of the five per-fold MAE or $R^2$ values on the same open file. Published external MAE 0.834 / $R^2$ 0.942 is an ensemble estimand. Published test MAE 0.599 / $R^2$ 0.934 is a CV-average estimand. Source: `results/stage1/cmi_reproduction_results.json`."
        )
    )
    if wt_test_ens_b and wt_ext_ens_b:
        s(
            md_table(
                pd.DataFrame(
                    [
                        {"file": "official test", "aggregation": "5-fold ensemble", "n": str(int(wt_test_b.get("n", 4338))), "MAE": f"{float(wt_test_ens_b['mae']):.3f}", "$R^2$": f"{float(wt_test_ens_b['r2']):.3f}"},
                        {"file": "official test", "aggregation": "mean of 5 fold-wise metrics", "n": str(int(wt_test_b.get("n", 4338))), "MAE": f"{float(wt_test_b['fold_mean_mae']):.3f}", "$R^2$": f"{float(wt_test_b['fold_mean_r2']):.3f}"},
                        {"file": "official external", "aggregation": "5-fold ensemble", "n": str(int(wt_ext_b.get("n", 398))), "MAE": f"{float(wt_ext_ens_b['mae']):.3f}", "$R^2$": f"{float(wt_ext_ens_b['r2']):.3f}"},
                        {"file": "official external", "aggregation": "mean of 5 fold-wise metrics", "n": str(int(wt_ext_b.get("n", 398))), "MAE": f"{float(wt_ext_b['fold_mean_mae']):.3f}", "$R^2$": f"{float(wt_ext_b['fold_mean_r2']):.3f}"},
                    ]
                ),
                ["file", "aggregation", "n", "MAE", "$R^2$"],
            )
        )
    else:
        s("Released-weight evaluation artifact not found.")
    s("")
    arch_rt = (cmi_repro.get("architecture_retrain") or {}) if isinstance(cmi_repro, dict) else {}
    if str(arch_rt.get("status", "")).lower() == "complete":
        ae = arch_rt.get("evaluations") or {}
        at = ae.get("cmi_test") or {}
        ax = ae.get("cmi_ext") or {}
        at_ens = at.get("ensemble") or {}
        ax_ens = ax.get("ensemble") or {}
        s("**Table S1.8. From-scratch CMI-architecture retrain (hidden 2400, depth 6, five folds).**")
        s(note_block("Published CMI weights were not loaded. Chemprop 1.6.1 was trained on the official train file with architecture recovered from the released checkpoints. Source: `results/stage1/cmi_reproduction_results.json`."))
        if at_ens and ax_ens:
            s(
                md_table(
                    pd.DataFrame(
                        [
                            {"file": "official test", "aggregation": "5-fold ensemble", "MAE": f"{float(at_ens['mae']):.3f}", "$R^2$": f"{float(at_ens['r2']):.3f}"},
                            {"file": "official test", "aggregation": "mean of 5 fold-wise metrics", "MAE": f"{float(at['fold_mean_mae']):.3f}", "$R^2$": f"{float(at['fold_mean_r2']):.3f}"},
                            {"file": "official external", "aggregation": "5-fold ensemble", "MAE": f"{float(ax_ens['mae']):.3f}", "$R^2$": f"{float(ax_ens['r2']):.3f}"},
                            {"file": "official external", "aggregation": "mean of 5 fold-wise metrics", "MAE": f"{float(ax['fold_mean_mae']):.3f}", "$R^2$": f"{float(ax['fold_mean_r2']):.3f}"},
                        ]
                    ),
                    ["file", "aggregation", "MAE", "$R^2$"],
                )
            )
        s("")
    s("**Historical CMI-split verification.** The shared-architecture from-scratch Chemprop point retrain produced a **single test-file** estimand: MAE "
      + f"{cmi_test['mae']:.3f}"
      + " and $R^2$ "
      + f"{cmi_test['r2']:.3f}"
      + f" ($n={int(cmi_test['n']):,}$ open test-file rows). The published Table 1 **test** MAE {pub['cmi_published_m4_test_mae']} and $R^2$ {pub['cmi_published_m4_test_r2']} are **five-fold CV averages**, a different estimand. On the **external file**, MAE was "
      + f"{cmi_ext['mae']:.3f}"
      + " and $R^2$ "
      + f"{cmi_ext['r2']:.3f}"
      + f" ($n={int(cmi_ext['n'])}$) versus published external MAE {pub['cmi_published_m4_ext_mae']} and $R^2$ {pub['cmi_published_m4_ext_r2']}. External additional features contained NaNs that were filled with **train-split column medians** before prediction; that fill is a split-verification detail, not the ligand-disjoint production pipeline. These numbers are a historical protocol comparison, not a ligand-disjoint generalization estimate. Ligand-disjoint DS1 Chemprop point MAE is reported in main-text Table 1 for comparison with the CMI MAE columns; the corresponding $R^2$ and RMSE are in S3.")
    s("")
    s("### S1.5.3 Protonation-policy audit")
    s("")
    s(f"The H$^+$ decision remains `train_with_H` on the basis of Chemprop point metal-complex fold-mean $R^2$ {numbers['hplus_r2_with']} versus {numbers['hplus_r2_without']}.")
    s("")
    s("## S2. Metrics and uncertainty definitions")
    s("")
    s("This section defines the aggregation conventions used throughout the supplement and lists the model inventory. It answers how row-level versus pair-level (MoP) accuracy and interval summaries are computed, and how predictive uncertainty quantification (UQ) intervals are formed. Metric definitions here are primary; later sections refer back to S2 rather than redefining the same terms. Model-architecture detail is supporting.")
    s("")
    s("Row-level RMSE: each held-out row contributes equally. Per-pair RMSE for pair $p$ is $\\mathrm{RMSE}_p=\\sqrt{n_p^{-1}\\sum_{i\\in p}(y_i-\\widehat y_i)^2}$. MoP RMSE is the unweighted mean of per-pair RMSE over pairs with at least three held-out ligands. Row-level $R^2$ is computed on held-out rows in each fold. MoP $R^2$ is the mean of within-pair $R^2$ values and is auxiliary. PICP@90 is empirical coverage of nominal 90% intervals; ACE is the absolute deviation from 0.90. Row-level PICP/ACE are row-weighted; MoP PICP/ACE are pair-equal. Retention-curve AUC is the area under the retention curve (AUC) of retained-subset error versus retention fraction on the canonical 10–100% grid. Event precision@$k$ is the fraction of the top $k$ ranked ligands that are events. True hit@$k$ is whether the shortlist contains at least one event. Fold aggregation is the five-fold mean ± SD (`ddof=1`) of fold-level summaries.")
    s("")
    s("### S2.1 Active model inventory")
    s("")
    s("**Table S2.1. Active and excluded models (both datasets).** " + source_note("manuscripts/final/tables/model_grid_final.csv"))
    s("")
    s(md_table(grid, ["display_name", "model_id", "dataset", "feature_representation", "uncertainty_type", "ensemble_size", "stochastic_sample_count", "active_status", "uq_status", "manuscript_role"], {"display_name": "label", "stochastic_sample_count": "samples", "active_status": "status"}))
    s("")
    s("Exact GPR numbers do not appear in active comparison tables. Exact GPR rows are `excluded` / `computationally_infeasible` and carry no performance numbers. Bayesian last-layer, Chemprop MC-dropout, ranking-auxiliary, and CPEMB models are absent from this table because they are not in the corrected active grid. The `samples` column is the number of stochastic forward passes at prediction; it is not ensemble size. Heteroscedastic models use 30 Monte Carlo dropout passes with dropout kept active at prediction, and the resulting within-member variance enters the law-of-total-variance total in Section S2.4.")
    s("")
    s("**Pair-metric eligibility and accuracy completeness.** Uncertainty-bearing DS2 models with complete five-fold all-pair MoP accuracy summaries (`n_valid_folds=5` on MoP RMSE, MoP $R^2$, and sign accuracy) are eligible for operational comparison. Point models are not operationally eligible. Exact GPR is excluded.")
    s("")
    s("### S2.2 Preprocessing and training settings")
    s("")
    s("Shared tabular MLP width is $[512,512,256]$. Learning rate $10^{-3}$, weight decay $10^{-5}$, maximum 200 epochs, batch size 2,048, and patience 25 are the shared AdamW loop settings. Dropout is **not** shared across MLP families. Family-specific architecture, uncertainty type, stochastic samples, and key hyperparameters are summarized below. Tabular evidential models select $\\lambda_{\\mathrm{reg}}$ from $\\{0.01,0.2\\}$ on inner validation (the audited workhorse checkpoints selected $0.2$).")
    s("")
    s("**Table S2.2. Family-specific training settings.**")
    s(
        md_table(
            pd.DataFrame(
                [
                    {"family": "Point MLP", "architecture": "tabular MLP $[512,512,256]$", "uncertainty type": "none (point)", "stochastic samples": "0", "key hyperparameters": "dropout $0.1$; ensemble size 1"},
                    {"family": "Evidential MLP / ensemble", "architecture": "tabular MLP $[512,512,256]$", "uncertainty type": "NIG / Student-$t$", "stochastic samples": "0", "key hyperparameters": "dropout $0.1$; ensemble size 1 or 5; $\\lambda_{\\mathrm{reg}}\\in\\{0.01,0.2\\}$"},
                    {"family": "Heteroscedastic MLP / ensemble", "architecture": "tabular MLP $[512,512,256]$", "uncertainty type": "input-dependent Gaussian variance", "stochastic samples": "30", "key hyperparameters": "dropout $0.2$; ensemble size 1 or 5; log-variance clamped to $[-8,8]$ then exponentiated; 30 Monte Carlo dropout passes at prediction"},
                    {"family": "MC-dropout MLP / ensemble", "architecture": "tabular MLP $[512,512,256]$", "uncertainty type": "Monte Carlo (MC) dropout", "stochastic samples": "30", "key hyperparameters": "dropout $0.2$; ensemble size 1 or 5"},
                    {"family": "SNGP / ensemble", "architecture": "spectrally normalized MLP + random Fourier feature (RFF) head", "uncertainty type": "Gaussian predictive moment", "stochastic samples": "0", "key hyperparameters": "dropout $0$; 512 random Fourier features; ensemble size 1 or 5"},
                    {"family": "Full variational Bayesian neural network (BNN) / ensemble", "architecture": "variational MLP over linear-layer weights", "uncertainty type": "empirical posterior predictive", "stochastic samples": "30", "key hyperparameters": "dropout $0$; ensemble size 1 or 5"},
                    {"family": "Sparse variational Gaussian process regression", "architecture": "sparse variational Gaussian process (SVGP), no neural backbone", "uncertainty type": "Gaussian predictive moment", "stochastic samples": "0", "key hyperparameters": "512 $k$-means inducing points; Matérn-5/2 ARD; 80 variational epochs; batch 1,024; patience 15"},
                    {"family": "Chemprop evidential", "architecture": "directed message-passing neural network (D-MPNN) hidden size 300, depth 3", "uncertainty type": "native NIG / Student-$t$", "stochastic samples": "0", "key hyperparameters": "dropout $0$; native evidential regularization $0.2$"},
                ]
            ),
            ["family", "architecture", "uncertainty type", "stochastic samples", "key hyperparameters"],
        )
    )
    s("")
    s("### S2.3 Predictive interval construction")
    s("")
    s("Not every UQ family uses the same interval construction. For normal-moment intervals, the nominal 90% interval is")
    s("")
    s("$$")
    s("\\bar{\\mu}(x)\\pm 1.645\\,\\sigma_{\\mathrm{total}}(x).")
    s("$$")
    s("")
    s("The factor $1.645$ is $\\Phi^{-1}(0.95)$ as implemented by `scipy.stats.norm.ppf`. Evidential intervals use the Student-$t$ predictive distribution implied by the Normal–Inverse-Gamma (NIG) parameters. The corresponding quantile is used for the nominal 90% interval. The exact interval implementation is `student_t_nig_predictive`; normal-moment intervals use `normal_moment`. Remaining families: full variational Bayesian neural network (BNN) = `empirical_posterior_predictive`; SNGP and sparse Gaussian process regression (GPR) = `gaussian_predictive_moment`.")
    s("")
    s("Two DS2 interval summaries are reported. Row-level prediction interval coverage probability (PICP) at 90% is the fraction of held-out ligand-level $\\Delta\\log K$ observations inside the corresponding nominal 90% intervals; it is row-weighted, so heavily sampled pairs contribute more observations. Row-level average coverage error (ACE) is the mean absolute deviation of empirical coverage from the nominal level on the same held-out rows. MoP PICP@90 and MoP ACE are unweighted means of within-pair PICP@90 and ACE on pairs with at least three held-out ligands and a complete set of valid intervals, matching the supported-pair set used for MoP RMSE. Main-text Table 2 displays MoP summaries for screen-passing models together with the row-level PICP@90 and ACE used by the operational interval-usability screen; the full row-weighted inventory is Supplementary Table S4.2. DS1 PICP@90 is the fraction of held-out metal-complex rows whose observed absolute $\\log K$ lies inside the nominal 90% predictive interval and is not a DS1 MoP pair metric. Fold-level summaries are then averaged across the five outer folds. For Chemprop evidential DS2, PICP and ACE are conditional on valid native NIG intervals (13 invalid native-UQ rows excluded). For the other DS2 models, all five fold summaries are complete.")
    s("")
    s("DS1 uncertainty estimates uncertainty in absolute $\\log K$ predictions. Propagated ad hoc uncertainty estimates uncertainty in a difference constructed from those predictions. Direct DS2 uncertainty estimates uncertainty in a model trained for $\\Delta\\log K$. These quantities are not interchangeable, and DS1 PICP or ACE values are not compared with DS2 PICP or ACE as if they measured the same prediction problem.")
    s("")
    s("Evidential models predict NIG parameters and use Student-$t$ NIG intervals. The heteroscedastic MLP has the shared tabular backbone $[512,512,256]$ and two output quantities: a predictive mean and an input-dependent log observation variance. The variance output is transformed to a positive variance by exponentiation of a log-variance clamped to $[-8,8]$. Monte Carlo dropout keeps dropout active at prediction (30 forward passes) and is a stochastic-regularization estimator (MC dropout only), not a full Bayesian neural network. `BAYES_FULL` is a fixed-prior variational model over all Bayesian linear-layer weights. SNGP uses a spectrally normalized MLP representation and a random-Fourier-feature output head. Sparse Gaussian process regression is a sparse variational Gaussian process (SVGP) with no neural feature-learning backbone.")
    s("")
    s("### S2.4 Workhorse uncertainty decomposition")
    s("")
    s("The ensemble decomposition is an application of the law of total variance (LoTV). The between-member term reflects variation among ensemble predictions; the input-dependent variance term is learned within each heteroscedastic member; the within-member term is the Monte Carlo variance of dropout means. The displayed ensemble variance uses the implementation’s `ddof=0` convention. At prediction, each heteroscedastic member uses $T=30$ Monte Carlo dropout forward passes, with dropout modules kept stochastic (`enable_mc_dropout`). The Table S2.1 `samples` column records this prediction-time count; it is not ensemble size. Frozen fold-score artifacts store $\\sigma_{\\mathrm{aleatoric}}$, $\\sigma_{\\mathrm{epistemic,within}}$, $\\sigma_{\\mathrm{epistemic,between}}$, and $\\sigma_{\\mathrm{total}}$, and the three variance components combine as written below.")
    s("")
    s("For workhorse ensemble member $m$, let $\\mu_{m,t}(x)$ be the predicted mean on Monte Carlo dropout pass $t$ and let $s_{m,t}(x)$ be the corresponding clamped log observation variance. The member mean and member observation variance are")
    s("")
    s("$$")
    s("\\mu_m(x)=\\frac{1}{T}\\sum_{t=1}^{T}\\mu_{m,t}(x),")
    s("\\qquad")
    s("v_m(x)=\\frac{1}{T}\\sum_{t=1}^{T}\\exp\\bigl(s_{m,t}(x)\\bigr).")
    s("$$")
    s("")
    s("The within-member component for member $m$ is the Monte Carlo variance of the dropout means,")
    s("")
    s("$$")
    s("\\sigma_{\\mathrm{epistemic,within},m}^{2}(x)")
    s("=")
    s("\\frac{1}{T}\\sum_{t=1}^{T}")
    s("\\bigl(\\mu_{m,t}(x)-\\mu_m(x)\\bigr)^{2}.")
    s("$$")
    s("")
    s("The ensemble mean is")
    s("")
    s("$$")
    s("\\bar{\\mu}(x)=\\frac{1}{M}\\sum_{m=1}^{M}\\mu_m(x).")
    s("$$")
    s("")
    s("The aleatoric component is the mean predicted observation variance,")
    s("")
    s("$$")
    s("\\sigma_{\\mathrm{aleatoric}}^{2}(x)")
    s("=")
    s("\\frac{1}{M}\\sum_{m=1}^{M}v_m(x).")
    s("$$")
    s("")
    s("The averaged within-member Monte Carlo component is")
    s("")
    s("$$")
    s("\\sigma_{\\mathrm{epistemic,within}}^{2}(x)")
    s("=")
    s("\\frac{1}{M}\\sum_{m=1}^{M}")
    s("\\sigma_{\\mathrm{epistemic,within},m}^{2}(x).")
    s("$$")
    s("")
    s("The between-member component is")
    s("")
    s("$$")
    s("\\sigma_{\\mathrm{epistemic,between}}^{2}(x)")
    s("=")
    s("\\frac{1}{M}\\sum_{m=1}^{M}")
    s("\\bigl(\\mu_m(x)-\\bar{\\mu}(x)\\bigr)^{2}.")
    s("$$")
    s("")
    s("The total predictive variance is")
    s("")
    s("$$")
    s("\\sigma_{\\mathrm{total}}^{2}(x)")
    s("=")
    s("\\sigma_{\\mathrm{aleatoric}}^{2}(x)")
    s("+")
    s("\\sigma_{\\mathrm{epistemic,within}}^{2}(x)")
    s("+")
    s("\\sigma_{\\mathrm{epistemic,between}}^{2}(x).")
    s("$$")
    s("")
    s("The reported $\\sigma_{\\mathrm{total}}$ is the square root of this variance. It is a predictive-risk score, not an applicability-domain score. The five-member workhorse ensemble uses independently trained members under the same fold-local training protocol; the ensemble mean and variance are combined using the decomposition above.")
    s("")
    s("The model-predicted $\\sigma_{\\mathrm{aleatoric}}$ is an input-dependent component learned by the heteroscedastic model. It is not an experimentally measured laboratory-noise variance and should not be equated with the descriptive within-condition disagreement in M4.")
    s("")
    s("Supplementary Figure S9 shows standard-deviation components (Panel A) and variance components (Panel B). Panel A standard deviations do not add linearly. Panel B variances add to total predictive variance as written above. These are model-internal uncertainty components, not laboratory-noise measurements.")
    s("")
    s("## S3. Complete DS1 results")
    s("")
    s("This section reports the complete ligand-disjoint absolute-binding benchmark. It answers how difficult $\\log K$ prediction is under the unseen-ligand regime used in the main text. It is supporting evidence for Methods 2.1–2.2, not the selectivity workhorse comparison.")
    s("")
    s("DS1 is a supporting absolute-binding benchmark and is not used to define the DS2 workhorse. After the official-split audit in S1, these ligand-disjoint $R^2$ values are the non-leaked $\\log K$ estimates used here; they are far below the official-split $R^2$ near 0.93 and indicate that ligand-disjoint absolute binding remains a difficult regression task. DS1 uncertainty-aware models enter the later manuscript only as the ad hoc differencing baseline in S13; they are not treated as DS2 uncertainty models. On metal-complex records, the RDKit MLP was the strongest point-model comparator, while the MC-dropout MLP ensemble had the lowest RMSE among uncertainty-aware models. The full long-form fold summary is `manuscripts/final/tables/supplementary_complete_model_results.csv` and the fold breakdown is `supplementary_fold_breakdowns.csv`.")
    s("")
    s("**Table S3.1. DS1 point models, metal-complex scope, fold mean ± SD.**")
    s(md_table(ds1p, ["display_name", "rmse", "r2", "n_valid_folds", "metric_status"], {"r2": "$R^2$"}))
    s("")
    s("**Table S3.2. DS1 uncertainty-aware models, metal-complex scope, fold mean ± SD.**")
    s(note_block("DS1 PICP@90 and ACE are absolute $\\log K$ row-level calibration on held-out metal-complex rows. They are not DS1 MoP pair metrics and are not comparable to DS2 $\\Delta\\log K$ coverage."))
    s(md_table(ds1u, ["display_name", "rmse", "r2", "picp_90", "ace", "interval_method", "n_valid_folds", "uq_status"], {"r2": "$R^2$"}))
    s("")
    s("Restricted pooled MAPE is `not_computed`. Do not mix all-heldout DS1 rows with metal-complex headline rows; both scopes exist in the long table.")
    s("")
    si.extend(si_figure("S1", "Complete DS1 metal-complex RMSE and $R^2$ for active models.", "figS1_ds1_complete.png"))
    si.extend(si_figure("S2", "DS1 point-model metal-complex RMSE and $R^2$.", "figS2_ds1_point.png"))
    si.extend(si_figure("S3", "DS1 uncertainty-aware models on held-out metal-complex rows. RMSE and $R^2$ summarize absolute $\\log K$ prediction; PICP@90 and ACE summarize row-level calibration of nominal 90% intervals for absolute $\\log K$. These are not DS2 pair-level selectivity metrics.", "figS3_ds1_uq.png"))
    s("## S4. Complete DS2 model comparison")
    s("")
    s("These tables document the breadth of the evaluated model grid; they are not a claim that the models form a single leaderboard. The main manuscript uses the model grid only to establish that several direct-UQ models are operationally usable and that downstream results are conditional on one representative model. Metric definitions are in S2.")
    s("")
    s("Pair-level and row-level metrics are **not interchangeable**. Full-data DS2 all-pair MoP is the primary application scope. Pair-level tables in this section report MoP PICP@90 and MoP ACE. Row-level PICP@90 and ACE remain the operational screen and are reported with the secondary row-weighted diagnostics in S4.3. Restricted chemical-subset results are deferred to Section S13 and are not used for workhorse selection. Row-level RMSE and $R^2$ are **secondary row-weighted diagnostics** that weight heavily sampled pairs more strongly. Negative and unstable MoP $R^2$ means that pair-specific mean selectivity is poorly explained; the operational route is interval-usable MoP error plus uncertainty-ordered subsetting and ranking, not a high-$R^2$ claim.")
    s("")
    ree_case = ree[ree.display_name.astype(str) == wh_name].copy() if "display_name" in ree.columns else ree
    if ree_case.empty:
        ree_case = ree
    s("### S4.1 Primary all-pair pair-level comparison")
    s("")
    s("**Table S4.1. DS2 all-pair pair-level accuracy and MoP interval calibration.**")
    s(note_block("Primary full-data DS2 all-pair scope. All-pair MoP RMSE is the primary accuracy metric. All-pair PICP@90 and ACE here are pair-equal MoP summaries on the same supported pairs. Row-level PICP@90 and ACE used for the operational screen are Table S4.2. All-pair MoP $R^2$ is auxiliary. Scope = `all_pairs`. Source = `manuscripts/final/tables/ds2_all_pairs.csv`."))
    s(md_table(allp, ["display_name", "mop_rmse", "mop_r2", "sign_accuracy", "mop_picp_90", "mop_ace"], {"display_name": "model", "mop_r2": "All-pair MoP $R^2$ (aux)", "mop_picp_90": "all-pair MoP PICP@90", "mop_ace": "all-pair MoP ACE"}))
    s("")
    si.extend(si_figure("S4", "DS2 all-pair pair-level comparison.", "figS4_ds2_all_pairs.png", "MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. Negative values indicate poor and unstable within-pair explanatory performance under this aggregation."))
    s("### S4.2 Primary all-pair pair-level comparison: interpretation")
    s("")
    s("Table S4.1 reports the primary all-pair pair-level comparison. All-pair MoP RMSE is the headline accuracy metric; MoP $R^2$ is auxiliary, and MoP PICP@90 and ACE provide pair-equal interval diagnostics. The secondary row-weighted diagnostics used for the operational interval-usability screen are reported separately in Section S4.3.")
    s("")
    s("### S4.3 Secondary row-weighted diagnostics")
    s("")
    s("**Table S4.2. Secondary row-weighted diagnostics, all-pair scope.**")
    s(note_block("Secondary row-weighted diagnostics only, including the all-pair row-level PICP@90 and ACE used for the operational screen. Scope = `all_pairs`. Source = `manuscripts/final/tables/ds2_all_pairs.csv`."))
    s(md_table(allp, ["display_name", "row_rmse", "row_r2", "sign_accuracy", "picp_90", "ace"], {"display_name": "model", "row_r2": "row-level $R^2$", "picp_90": "row-level PICP@90", "ace": "row-level ACE"}))
    s("")
    s("### S4.4 High-selectivity row estimand")
    s("")
    s("These two estimands (S4.4 and S4.5) are retained as audit tables documenting scope sensitivity; they are not treated as independent primary outcomes and are not used for workhorse selection.")
    s("")
    s("**Table S4.3. High-selectivity row-scope pair-level metrics.**")
    s(note_block("Scope = high_selectivity_row; $|y|>1$ inside pairs with ≥3 high-selectivity ligands. Monitoring scope, not a selection rule. Pair-level columns, including MoP PICP@90 and MoP ACE. Source = `manuscripts/final/tables/ds2_highsel_row.csv`."))
    s(md_table(hsrow, ["display_name", "mop_rmse", "mop_r2", "sign_accuracy", "mop_picp_90", "mop_ace"], {"display_name": "model", "mop_r2": "MoP $R^2$", "mop_picp_90": "MoP PICP@90", "mop_ace": "MoP ACE"}))
    s("")
    s("**Table S4.4. High-selectivity row-scope secondary row-weighted diagnostics.**")
    s(note_block("Same high_selectivity_row scope as Table S4.3, but secondary row-weighted diagnostics only, including row-level PICP@90 and ACE."))
    s(md_table(hsrow, ["display_name", "row_rmse", "row_r2", "picp_90", "ace"], {"display_name": "model", "row_r2": "row $R^2$", "picp_90": "row-level PICP@90", "ace": "row-level ACE"}))
    s("")
    si.extend(si_figure("S5", "DS2 high-selectivity row-scope comparison.", "figS5_ds2_highsel_row.png", "MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. Negative values indicate poor and unstable within-pair explanatory performance under this aggregation."))
    s("### S4.5 High-selectivity pair-membership estimand")
    s("")
    s("**Table S4.5. High-selectivity pair-membership pair-level metrics.**")
    s(note_block("Scope = high_selectivity_all_pair_membership. Monitoring scope, not a selection rule. Pair-level columns, including MoP PICP@90 and MoP ACE. Source = `manuscripts/final/tables/ds2_highsel_pair.csv`."))
    s(md_table(hspair, ["display_name", "mop_rmse", "mop_r2", "sign_accuracy", "mop_picp_90", "mop_ace"], {"display_name": "model", "mop_r2": "MoP $R^2$", "mop_picp_90": "MoP PICP@90", "mop_ace": "MoP ACE"}))
    s("")
    s("**Table S4.6. High-selectivity pair-membership secondary row-weighted diagnostics.**")
    s(md_table(hspair, ["display_name", "row_rmse", "row_r2", "picp_90", "ace"], {"display_name": "model", "row_r2": "row $R^2$", "picp_90": "row-level PICP@90", "ace": "row-level ACE"}))
    s("")
    si.extend(si_figure("S6", "DS2 high-selectivity pair-membership comparison.", "figS6_ds2_highsel_pair.png", "MoP $R^2$ is an auxiliary diagnostic, not the primary model-selection or downstream-ranking criterion. Negative values indicate poor and unstable within-pair explanatory performance under this aggregation."))
    s("**Chemprop DS2 native UQ diagnostics.**")
    s("")
    s(md_table(chem_d, ["fold_id", "n_rows_total", "n_point_valid", "n_uq_valid", "n_uq_invalid", "uncertainty_status", "checkpoint_unchanged"]))
    s("")
    s(f"Total invalid native NIG rows = {invalid_total}. Pair-metric status = `{getattr(chem_note, 'pair_metric_status', chem_note.ranking_status)}`. UQ status = `{chem_note.uq_status}`. PICP/ACE are conditional on valid native-UQ rows.")
    s("")
    s("## S5. Direct-UQ screening and workhorse-selection bridge")
    s("")
    s("This section links the complete DS2 model comparison in Section S4 to the representative-model selection procedure in Section S6. Point models provide predictions but no predictive uncertainty and therefore cannot serve as downstream uncertainty-based workhorses. Direct-UQ models are screened using the study-defined row-level interval-usability convention: five valid outer-fold summaries, complete native uncertainty output, row-level PICP@90 between 0.85 and 0.95, and row-level ACE no greater than 0.10.")
    s("")
    s("The screening criteria are operational conventions defined for this study; they are not externally validated calibration thresholds and do not establish a universal applicability-domain boundary. Passing the screen is necessary but not sufficient for selection. Among screen-passing direct-UQ models, Section S6 applies the prespecified RMSE–PICP–ACE Pareto procedure to identify one fixed representative model for downstream risk-ordering and acquisition analyses.")
    s("")
    s("The selected heteroscedastic MLP ensemble is therefore a fixed operational workhorse rather than a universally optimal architecture. Its selection does not imply that it has the lowest all-pair MoP RMSE, nor was it selected using pair-acquisition event precision@$k$, true hit@$k$, or retrospective regret@$k$.")
    s("")
    s("## S6. Representative direct-UQ model context and workhorse choice")
    s("")
    s("This section records how the representative direct-UQ model was chosen. It answers which models passed the operational interval-usability screen and why nearby alternatives were not unique optima. The selected model is carried forward because the downstream analysis requires one fixed model and because it satisfies the study’s operational interval-usability convention. Nearby models are not treated as scientifically inferior. The corresponding scatter plot is main-text Figure 2.")
    s("")
    s("Figure 2 plots all-pair MoP RMSE against pair-equal MoP ACE and MoP PICP@90. The shaded regions are descriptive MoP reference regions only (left: MoP $\\mathrm{ACE}\\le 0.20$; right: $0.85\\le\\mathrm{PICP@90}\\le 0.95$). They are not labeled as a screen or eligibility region. Eligibility is determined separately from unrounded five-fold row-level PICP@90 and ACE values ($0.85\\le\\mathrm{PICP@90}\\le 0.95$ and $\\mathrm{ACE}\\le 0.10$) and cannot be read from the plotted MoP coordinates. The MoP ACE and MoP PICP@90 shading is included to provide pair-level visual context; neither shaded region is the operational eligibility gate. Star marks the selected representative model. Diamond = Chemprop evidential (partial native UQ). Cross = models not interval-usable under the row-level operational screen.")
    s("")
    s("Rule file: `results/model_audit/workhorse_selection_rule.json`. Operational selection: `results/manuscript_final/workhorse_selection_accuracy_calibration.json`. Candidates: `results/manuscript_final/workhorse_selection_candidates.csv`.")
    s("")
    s("Procedure: operational interval-usability screen on five-fold mean all-pair row-level PICP@90 and mean all-pair row-level ACE → RMSE–PICP–ACE Pareto. The Pareto comparison treats lower all-pair MoP RMSE, lower row-level ACE, and closer-to-nominal row-level PICP@90 as preferable, subject to the operational screen. Model A dominates model B if it is no worse on all three objectives and strictly better on at least one, where the PICP objective is $|\\mathrm{PICP@90}-0.90|$ computed from the same row-level all-pair PICP@90 used by the screen. The selected model is the Pareto-front member with the smallest Euclidean distance to the ideal point after min-max normalization among the Pareto set. Remaining ties use all-pair sign accuracy, then all-pair MoP $R^2$, then a prespecified operational-simplicity ranking. Combined $|\\mathrm{PICP@90}-0.90|+\\mathrm{ACE}$ is not a selection key. Workhorse selection uses the all-pair scope only. No display-rounding tie-breaker was required for the reported candidates. This is not a formal statistical test. Row-level RMSE cannot enter selection. Complete native UQ is required for operational eligibility; Chemprop remains a scientific comparator. The PICP and ACE limits were study-defined operational conventions, not externally validated universal thresholds.")
    s("")
    s("**Table S6.1. Operational workhorse selection on the all-pair scope.**")
    s(note_block("Direct DS2 models only. calibration_usable, all-pair MoP RMSE, all-pair MoP PICP@90 and ACE, all-pair row-level PICP@90 and ACE (operational screen), all-pair MoP $R^2$ (auxiliary), selection status. Main-text Table 2 is a compact screen-passer subset. Ad hoc DS1-difference comparators are Section S13 and are omitted here. The screen uses row-level $\\Delta\\log K$ coverage. Passing the screen is necessary, not sufficient: `screen-pass, Pareto-dominated` means the model met the interval-usability convention but was dominated on the RMSE–PICP–ACE Pareto front. " + source_note("manuscripts/final/tables/supplementary_workhorse_selection.csv")))
    sel_si_disp = sel_si.copy()
    status_map = {
        "dominated_screen_passer": "screen-pass, Pareto-dominated",
        "selected": "selected",
        "candidate": "screen-pass",
        "screened_out": "screened out",
        "scientific_comparator_partial_uq_not_operational_candidate": "scientific comparator; partial UQ",
        "point_baseline_not_operationally_eligible": "point baseline; no UQ",
    }
    if "selection_status" in sel_si_disp.columns:
        sel_si_disp["selection_status"] = sel_si_disp["selection_status"].map(lambda x: status_map.get(str(x), str(x)))
    s(md_table(sel_si_disp, ["display_name", "calibration_usable", "all_pair_mop_rmse", "all_pair_picp_90", "all_pair_ace", "all_pair_row_picp_90", "all_pair_row_ace", "all_pair_mop_r2", "selection_status"], {"display_name": "model", "all_pair_picp_90": "MoP PICP@90", "all_pair_ace": "MoP ACE", "all_pair_row_picp_90": "row-level PICP@90", "all_pair_row_ace": "row-level ACE", "all_pair_mop_r2": "all-pair MoP $R^2$ (aux)", "selection_status": "status"}))
    s("")
    s("The prior manuscript’s obsolete PICP@$90\\ge 0.70$ gate was **not re-imposed** and was not used in the corrected analysis. Fold breakdowns for the selected workhorse are in `supplementary_fold_breakdowns.csv` filtered to the selected model id.")
    s("")
    s("**Table S6.2. Fold-wise differences versus the selected workhorse (interval-usable models only).**")
    s(note_block("Descriptive fold-wise Δ (comparator − selected full-data workhorse). Positive all-pair MoP RMSE delta means the comparator is worse on that fold. Not a nested significance test and not a selection key. Source = `manuscripts/final/tables/supplementary_fold_paired_differences.csv`."))
    if len(paired):
        paired_cols = [c for c in ["display_name", "delta_all_pair_mop_rmse", "n_folds_comparator_worse_rmse", "delta_all_pair_picp_90", "delta_all_pair_ace"] if c in paired.columns]
        s(md_table(paired, paired_cols, {"display_name": "model", "delta_all_pair_mop_rmse": "Δ all-pair MoP RMSE", "n_folds_comparator_worse_rmse": "folds with higher all-pair RMSE", "delta_all_pair_picp_90": "Δ PICP@90", "delta_all_pair_ace": "Δ ACE"}))
    s("")
    si.extend(si_figure("S7", "All-pair fold stability of the selected workhorse.", "figS7_workhorse_fold_stability.png", "Fold-local all-pair MoP RMSE and auxiliary MoP $R^2$ are shown for the five held-out folds of the selected heteroscedastic MLP ensemble. MoP $R^2$ is auxiliary and does not determine workhorse selection."))
    s("**Table S6.3. Workhorse MoP pair validity (invalid = constant-target pair $R^2$).**")
    s(note_block("Source = `manuscripts/final/tables/supplementary_mop_pair_validity.csv`. Displayed MoP $R^2$ values are rounded to three decimals; full precision remains in the artifact. Pair counts are fold-local integers. These unfiltered pair counts are the pairs contributing to full-retention MoP, not the 50% retained-MoP counts in Table 3 and not the top-$k$ eligible-pair counts. Scope = `all_pairs`. MoP $R^2$ excludes constant-target pairs only."))
    if len(mop_valid):
        mop_show = exclude_ree_scope(mop_valid[mop_valid.metric_name == "mop_r2"].copy())
        for col in ["n_pairs_retained", "n_pairs_valid", "n_pairs_invalid"]:
            if col in mop_show.columns:
                mop_show[col] = pd.to_numeric(mop_show[col], errors="coerce").round(0).astype("Int64")
        if "value" in mop_show.columns:
            mop_show["value"] = pd.to_numeric(mop_show["value"], errors="coerce").map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "NA")
        s(md_table(mop_show, ["fold_id", "scope", "n_pairs_retained", "n_pairs_valid", "n_pairs_invalid", "value"], {"value": "MoP $R^2$"}))
    s("")
    s("## S7. Retention and risk-ordering diagnostics")
    s("")
    s("This section reports the fold-local retention study. It answers whether model-derived uncertainty orders held-out error better than the tested molecular-distance tools. It supports main-text Table 3 and Figure 3. The tested distance scores provided weaker empirical risk ordering than $\\sigma_{\\mathrm{total}}$ in the primary all-pair retention analysis. The RDKit feature-centroid Mahalanobis score produced an intermediate reduction, whereas Tanimoto nearest-neighbor distance was closer to random-retention behavior. The tested distance scores were not sufficiently effective to serve as the primary risk-ordering rule in this task. These analyses are empirical error-ordering diagnostics, not coverage guarantees or certified acceptance boundaries.")
    s("")
    s("Fixed-retention scores rank held-out rows by predicted unreliability (`configs/experiments/ad_retention_spec.yaml`; artifacts in `results/downstream_cv/ad_retention/`). Ranking is performed within each analysis scope; at fraction $r$ the retained set is the $\\lfloor r n\\rfloor$ lowest-risk rows, with ties broken by record identifier.")
    s("")
    s("All retention analyses in this SI are empirical error-ordering analyses. They do not provide coverage guarantees, accept/reject thresholds, or universal candidate-quality scores.")
    s("")
    s("Chemprop evidential model = the `GRAPH_DMPNN_EVID` checkpoint; Chemprop native UQ = native NIG interval output; Chemprop representation-space score = embedding distance. These three objects are not interchangeable.")
    s("")
    s("### S7.1 Retention hypothesis and implementation")
    s("")
    s("(H1) The selected DS2 workhorse’s predictive uncertainty identifies held-out DS2 subsets with lower prediction error at fixed retention than random retention; DS1 retention analyses are supporting diagnostics. Pair-acquisition hypotheses H2–H5 are stated in S10.")
    s("")
    s("Unless otherwise qualified, $\\sigma_{\\mathrm{total}}$ is the selected workhorse’s fold-local total predictive uncertainty or total predictive standard deviation, including model-related and input-dependent predictive-variance components as represented by the selected model. Native Chemprop $\\sigma_{\\mathrm{total}}$ is a separate model-specific output and is not pooled with the workhorse score.")
    s("")
    s("### S7.2 Retention implementation")
    s("")
    s("At retention fraction $r$, the retained set is the $\\lfloor r n\\rfloor$ lowest-risk rows in the held-out fold, with ties broken by record identifier. Pair-support counts for fixed-retention MoP are the number of pairs with at least three retained ligands. The canonical pair identifier is bookkeeping only. Random retention uses 20 repeats with seed 0 inside each held-out fold.")
    s("")
    s("The 13 invalid Chemprop native-NIG rows affect Chemprop native UQ, not necessarily the availability of Chemprop representation-space scores or point predictions. Chemprop distance-based results are therefore reported separately from Chemprop native-UQ results. Chemprop embedding distances are distances in a learned molecular representation. Embeddings were taken from the training-fold Chemprop evidential checkpoint; held-out ligands were scored against that training-fold representation after training-fold standardization.")
    s("")
    s("**Table S7.1. Canonical fold-local retention-curve AUC inventory across DS1 and DS2.**")
    s(note_block("Table S7.1 inventories the canonical retention-curve AUC values reported in the manuscript and selected sensitivity analyses. For DS1, the endpoint is row-level RMSE; for DS2, the endpoint is pair-level MoP RMSE. Lower values indicate stronger empirical ordering of lower-error retained subsets. These are not formal risk–coverage AURCs. Grid = $\{0.10,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00\}$. Distance and pair-density scores use outer-training data only; model-output uncertainty uses the fold-local model output; random retention has no fit partition. Chemprop representation-space scores also use outer-training embeddings after training-fold standardization. Chemprop evidential model = checkpoint; Chemprop native UQ = native NIG interval output; Chemprop representation-space score = embedding distance. Native Chemprop $\\sigma_{\\mathrm{total}}$ is not pooled with the workhorse score. " + source_note("results/downstream_cv/ad_retention/retention_auc_summary.csv", "manuscripts/final/tables/retention_auc_summary.csv")))
    if len(ad_aurc):
        aurc_show = ad_aurc[
            ad_aurc.metric_name.astype(str).isin(
                ["retention_curve_auc_rmse", "retention_curve_auc_mop_rmse"]
            )
        ].copy()
        if "score_role" in aurc_show.columns:
            keep_role = (aurc_show.score_role.astype(str) == "primary") | (
                (aurc_show.dataset.astype(str) == "DS2") & (aurc_show.family.astype(str) == "workhorse")
            )
            aurc_show = aurc_show[keep_role]
        if "n_valid_folds" in aurc_show.columns:
            aurc_show = aurc_show[pd.to_numeric(aurc_show.n_valid_folds, errors="coerce").fillna(0) >= 5]
        keep_scope = (
            ((aurc_show.dataset == "DS1") & (aurc_show.scope == "metal_complex") & (aurc_show.metric_name == "retention_curve_auc_rmse"))
            | ((aurc_show.dataset == "DS2") & (aurc_show.scope.isin(["all_pairs"])) & (aurc_show.metric_name == "retention_curve_auc_mop_rmse"))
        )
        aurc_show = aurc_show[keep_scope].copy()
        aurc_show["score"] = [s10_score_label(f, sc) for f, sc in zip(aurc_show["family"], aurc_show["score_name"])]
        aurc_show["family"] = aurc_show["family"].map(family_display_name)
        aurc_show["endpoint"] = aurc_show["metric_name"].map(
            {
                "retention_curve_auc_rmse": "row-level RMSE",
                "retention_curve_auc_mop_rmse": "pair-level MoP RMSE",
            }
        ).fillna(aurc_show["metric_name"])
        aurc_show["retention-curve AUC"] = [_fmt_msd(a, b, n) for a, b, n in zip(aurc_show.fold_mean, aurc_show.fold_sd, aurc_show.n_valid_folds)]
        s(md_table(aurc_show, ["dataset", "family", "scope", "score", "endpoint", "retention-curve AUC", "n_valid_folds"]))
    s("")
    s("**Table S7.2. Score definitions (fold-local fit partitions).** Here, NN denotes nearest neighbor and $k$NN denotes $k$-nearest neighbor. Chemprop Euclidean-NN and Chemprop embedding-centroid Mahalanobis are Chemprop representation-space scores computed after training-fold standardization. `mahalanobis_feature_centroid` is the machine-readable alias for RDKit feature-centroid Mahalanobis distance (`rdkit_feature_mahalanobis_centroid`); it is not the Chemprop embedding centroid score. The obsolete label `mahalanobis_joint` refers to the same RDKit centroid score. Chemprop native UQ is the native NIG interval output and is distinct from the Chemprop evidential checkpoint.")
    if len(ad_defs):
        def_cols = [c for c in ad_defs.columns if c in {"score_name", "method_name", "display_name", "family", "definition", "fit_partition", "orientation", "score_role", "chemprop_only", "availability"}]
        if not def_cols:
            def_cols = list(ad_defs.columns)[:8]
        s(md_table(ad_defs, def_cols, {"score_name": "score"}))
    s("")
    s("**Table S7.3. Ten-percent retention comparison versus random retention.** RMSE is summarized by an error-enrichment ratio $E^{\\mathrm{RMSE}}_{10}=\\mathrm{RMSE}_{\\mathrm{random},10\\%}/\\mathrm{RMSE}_{\\mathrm{score},10\\%}$ (values above one are better). Random subsets are drawn inside each held-out fold (20 repeats, seed 0) and summarized as fold mean ± SD.")
    s(note_block("This is an error-enrichment ratio, not a percentage change and not a hit-rate enrichment measure; values above one indicate lower retained-subset RMSE than random retention. Empirical error-ordering summary; see S7 for interpretation. DS1 metal_complex is the headline DS1 scope; all_heldout is retained as a secondary diagnostic and includes protonation records. These rows should not be compared as the same estimand. Chemprop evidential model = checkpoint; Chemprop native UQ = native NIG interval output; Chemprop representation-space score = embedding distance."))
    if len(ad_enrich):
        enr = exclude_ree_scope(ad_enrich.copy())
        if "n_valid_folds" in enr.columns:
            enr = enr[pd.to_numeric(enr.n_valid_folds, errors="coerce").fillna(0) >= 5]
        if "metric_name" in enr.columns:
            enr = enr[~enr.metric_name.astype(str).str.contains("mop_rmse") | (enr.dataset == "DS2")]
        enr = enr.copy()
        if "score_name" in enr.columns:
            if "family" in enr.columns:
                enr["score"] = [s10_score_label(f, sc) for f, sc in zip(enr["family"], enr["score_name"])]
            else:
                enr["score"] = enr["score_name"].map(score_display_name)
        if "family_display" in enr.columns:
            enr["family"] = enr["family_display"]
        elif "family" in enr.columns:
            enr["family"] = enr["family"].map(family_display_name)
        if {"fold_mean", "fold_sd", "n_valid_folds"}.issubset(enr.columns):
            enr["mean ± SD"] = [_fmt_msd(a, b, n) for a, b, n in zip(enr.fold_mean, enr.fold_sd, enr.n_valid_folds)]
        ecols = [c for c in ["dataset", "family", "scope", "score", "metric_name", "comparison_kind", "mean ± SD", "n_valid_folds"] if c in enr.columns]
        if "score" not in enr.columns:
            ecols = [c for c in ["dataset", "model_id", "family", "scope", "score_name", "metric_name", "fold_mean", "fold_sd", "n_valid_folds"] if c in enr.columns]
        s(md_table(enr, ecols, {"score_name": "score", "metric_name": "summary"}))
    s("")
    s("The expanded 50% retention inventory is retained as a machine-readable artifact and is not rendered as a supplementary table. " + source_note("results/downstream_cv/ad_retention/retention_50pct.csv", "manuscripts/final/tables/ad_retention_50pct.csv") + " The primary all-pair MoP retention summaries are Table 3 and Supplementary Table S7.1. Supplementary Table S7.3 provides the ten-percent error-enrichment comparison, while Table S8.1 reports the separate row-level retention diagnostic.")
    s("")
    s("The exploratory predicted-magnitude shortlist-overlap analysis has no canonical completed result and is not reported as a table.")
    s("")
    s("Chemprop embedding distances are distances in a learned molecular representation. Embeddings were taken from the training-fold `GRAPH_DMPNN_EVID` checkpoint; held-out ligands were scored against that training-fold representation after training-fold standardization. DS1 embeddings are complex-graph representations; DS2 embeddings are ligand-graph representations. They are not reported for MLP, SNGP, or GPR models and are not a transferable molecular-distance baseline. Leakage audit: `results/downstream_cv/ad_retention/leakage_audit.csv`.")
    s("")
    si.extend(si_figure(
        "S8",
        "All-pair representation-space retention sensitivity analysis.",
        "figS8_chemprop_embedding_retention.png",
        "Thick lines are $\\sigma_{\\mathrm{total}}$, Tanimoto nearest neighbor, and random retention. Chemprop embedding distances are muted sensitivity analyses and are not pooled with the selected workhorse uncertainty score.",
    ))
    si.extend(si_figure(
        "S9",
        "Workhorse uncertainty components; standard-deviation components and variance components.",
        "figS9_uncertainty_decomposition.png",
        "Selected workhorse = Heteroscedastic MLP ensemble (`DS2_RDKIT_MLPENS5_HETERO`). Bars are fold means. The variance identity is applied at the row level before averaging. Panel A standard deviations do not add linearly. Panel B variances add to total predictive variance as defined in Section S2.4.",
    ))
    s("")
    s("## S8. Pair-context and support diagnostics")
    s("")
    s("This section answers whether pair support, medium strata, or tested molecular-distance scores explain held-out error. It is supporting context for the main-text risk-ordering result. Pair density is a support diagnostic that helps explain why some pair contexts are harder than others; it is not a validated applicability domain.")
    s("")
    s("All retention analyses in this SI are empirical error-ordering analyses. They do not provide coverage guarantees, accept/reject thresholds, or universal candidate-quality scores.")
    s("")
    s("S8 reports contextual support diagnostics based on pair support, medium strata, and the association between ligand-level molecular-distance scores and row-level absolute residual. The primary fixed-retention inventory is S7. Pair-level error-ordering metrics (Table S7.1) and row-level retention-error diagnostics are distinguished below.")
    s("")
    si.extend(si_figure("S10", "Risk-score orientation audit; higher values indicate greater predicted risk.", "figS10_ad_orientation.png", "All score orientations were verified before retention analysis; higher scores indicate greater predicted risk."))
    si.extend(si_figure("S11", "All-pair MoP retention error curves; primary scores emphasized.", "figS11_retention_all_pairs.png", "Thick lines are $\\sigma_{\\mathrm{total}}$, RDKit feature-centroid Mahalanobis distance, Tanimoto nearest-neighbor distance, and random retention. Component and sensitivity scores are thinner muted lines. Empirical error-ordering scores, not out-of-distribution (OOD) detectors."))
    s("**Table S8.1. Fold-local row-level retention-curve AUC over the 10–100% grid.**")
    s(note_block("Empirical error-ordering summary; see S7 for interpretation. Same fold-local ranking procedure as Table S7.1 and the same 10–100% grid $\\{0.10,\\ldots,1.00\\}$, but the curve is row RMSE rather than MoP RMSE (a different estimand). Scope = all-pair DS2 and DS1 metal-complex. " + source_note("results/downstream_cv/ad_retention/retention_auc_summary.csv", "manuscripts/final/tables/retention_auc_summary.csv") + " Endpoint field = `retention_curve_auc_rmse`."))
    row_aurc_col = "row_rmse_at_50" if "row_rmse_at_50" in row_ad_tbl.columns else "row_rmse_at_50pct_retention"
    s(md_table(exclude_ree_scope(row_ad_tbl), ["scope", "score_name", "fit_partition", "row_aurc", row_aurc_col, "n_valid_folds"], {"score_name": "score", "row_aurc": "row AUC", row_aurc_col: "row RMSE@50%"}))
    s("")
    s("Although $\\sigma_{\\mathrm{total}}$ and $\\sigma_{\\mathrm{epistemic,within}}$ can be numerically close in the retention summaries, they are distinct model outputs. Their close ranking performance reflects similar fold-local ordering in this analysis, not an identity of the underlying uncertainty definitions. For the selected heteroscedastic-ensemble workhorse, $\\sigma_{\\mathrm{total}}$ is assembled from the available fold-local predictive components, including a between-member term.")
    s("")
    s("Tanimoto: outer-train ligands only. RDKit feature-centroid Mahalanobis: outer-train Ledoit–Wolf on descriptors that include ligand, metal, medium, solvent, and temperature variables (`mahalanobis_joint` is the same score). Pair density: outer-train pair counts. Chemprop embedding distances are learned-representation scores. Score orientation is higher = greater predicted risk. The tested fingerprint, descriptor-space, and learned-representation distances are tested molecular-distance tools, not operational risk scores.")
    s("")
    s("**Medium strata.** The metal–medium seen/unseen analysis was identifiable, but the unseen stratum was small and was not used as a headline estimate. Electrolyte-seen (`electrolyte_seen`) and ligand–medium-seen (`ligand_medium_seen`) comparisons were non-identifiable / `not_identifiable` under the ligand-disjoint split. **No conclusion about medium novelty was drawn.** The all-pair unseen RMSE is not evidence of general medium extrapolation.")
    s("")
    s("**Table S8.2. Metal–medium seen/unseen (workhorse).**")
    s(note_block("Stratum = whether the (species pair, electrolyte) pair was observed in outer-train. Unseen is a small slice. " + source_note("results/downstream_cv/ad/ad_metal_medium_strata_summary.csv", "manuscripts/final/tables/supplementary_metal_medium_strata.csv")))
    if len(mm):
        mm_show = exclude_ree_scope(mm[mm.metric_name.isin(["n_rows", "n_pairs", "rmse", "mop_rmse", "picp_90"])].copy())
        s(md_table(mm_show, ["scope", "stratum_label", "metric_name", "display", "n_valid_folds"], {"stratum_label": "stratum", "metric_name": "metric"}))
    s("")
    s("**Table S8.3. Pair-context support bins (workhorse, all-pairs).** This is a metal-pair data-support breakdown of the data-collection structure, not a formal chemistry analysis and not a validated applicability domain. The first category is **0 training ligands** and is a distinct pair-context extrapolation regime; it is not merged with `1-5` and is not complete chemical novelty. MoP RMSE in that bin is partial (`n_valid_folds=2`) and is not a headline. Counts `n_test_rows` / `n_test_pairs` are fold means of test rows or test pairs assigned to each training-density bin; they are not unique-pair counts pooled across folds and are not retained MoP-pair counts. " + source_note("results/downstream_cv/pair_density/pair_density_summary.csv", "manuscripts/final/tables/supplementary_pair_density_bins.csv"))
    if len(dens_bins):
        dens_metrics = [m for m in ["rmse", "mop_rmse", "mean_sigma_total", "n_test_rows", "n_test_pairs", "n_rows", "n_pairs"] if m in set(dens_bins.metric_name)]
        dens_show = dens_bins[dens_bins.metric_name.isin(dens_metrics)].copy()
        dens_show["bin"] = dens_show["bin"].astype(str).replace({"0": "0 training ligands"})
        s(md_table(dens_show, ["bin", "metric_name", "display", "n_valid_folds"], {"metric_name": "metric"}))
    s("")
    si.extend(si_figure("S12", "All-pair pair-support density and workhorse diagnostics.", "figS12_pair_density.png", "All-pair DS2 workhorse; bin 0 denotes zero training ligands. The zero-training-ligand bin is a distinct pair-context extrapolation regime and is not complete chemical novelty."))
    s("**Table S8.4. Fold-wise association between ligand-level molecular-distance score and row-level absolute residual.**")
    s(
        note_block(
            "Descriptive association diagnostic for the selected workhorse (`DS2_RDKIT_MLPENS5_HETERO`) and the Chemprop evidential model (`DS2_GRAPH_DMPNN_EVID`). "
            "Each fold computes Pearson $r$ and Spearman $\\rho$ between the named molecular-distance score and the row-level absolute residual $|y-\\widehat y|$. "
            "RDKit feature-space extras (Mahalanobis NN, Mahalanobis $k$NN mean, Euclidean-NN, median Euclidean-to-all) are SI-only and are not plotted on retention-curve figures. "
            "Values are five-fold mean $\\pm$ SD. This is not a selection metric, not a retained-subset RMSE, and not a substitute for the fixed-retention error-ordering analysis. "
            "The tested molecular-distance associations were weak and did not provide a sufficiently strong basis for replacing the workhorse uncertainty score in the ligand-disjoint DS2 retention analysis. "
            + source_note(
                "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv",
                "manuscripts/final/tables/ad_distance_error_corr.csv",
            )
        )
    )
    if len(corr):
        show = exclude_ree_scope(corr.copy())
        if {"pearson_abs_residual", "spearman_abs_residual"}.issubset(show.columns):
            cols = [c for c in ["model_id", "scope", "score_display", "pearson_abs_residual", "spearman_abs_residual", "n_valid_folds"] if c in show.columns]
            s(
                md_table(
                    show,
                    cols,
                    {
                        "model_id": "model",
                        "score_display": "score",
                        "pearson_abs_residual": "Pearson $r$",
                        "spearman_abs_residual": "Spearman $\\rho$",
                    },
                )
            )
        elif {"metric_name", "fold_mean", "fold_sd"}.issubset(show.columns):
            show = show[show.metric_name.astype(str).isin(["pearson_abs_residual", "spearman_abs_residual"])].copy()
            show["display"] = [_fmt_msd(a, b, n) for a, b, n in zip(show.fold_mean, show.fold_sd, show.n_valid_folds)]
            if "score_display" not in show.columns:
                show["score_display"] = show["score_name"].map(score_display_name)
            s(
                md_table(
                    show,
                    ["scope", "score_display", "metric_name", "display", "n_valid_folds"],
                    {"score_display": "score", "metric_name": "metric", "display": "fold mean ± SD"},
                )
            )
    s("")
    s("**Table S8.5. Most frequent held-out pairs for the selected workhorse (descriptive).**")
    s(note_block("Pairs are ranked by the total number of held-out ligand rows across all five outer folds, using only test-side rows. Per-pair RMSE is an out-of-fold pooled quantity, not MoP RMSE and not a five-fold mean ± SD. Observed high-selectivity counts use held-out labels only. This table is descriptive and label-informed; it is not a selection rule. " + source_note("results/downstream_cv/pair_event_support/pair_event_support_frequent.csv", "manuscripts/final/tables/pair_event_support_frequent.csv")))
    if len(pair_event_frequent):
        freq = pair_event_frequent.copy()
        freq["per_pair RMSE (OOF pooled)"] = [f"{float(v):.3f}" if pd.notna(v) else "NA" for v in freq.out_of_fold_pooled_per_pair_rmse]
        freq["mean $\\sigma_{\\mathrm{total}}$"] = [f"{float(v):.3f}" if pd.notna(v) else "NA" for v in freq.mean_sigma_total]
        freq["high-sel. fraction"] = [f"{float(v):.3f}" if pd.notna(v) else "NA" for v in freq.high_selectivity_fraction]
        s(
            md_table(
                freq,
                [
                    "metal_species_pair",
                    "n_heldout_rows",
                    "n_folds_represented",
                    "n_high_selectivity_events",
                    "high-sel. fraction",
                    "per_pair RMSE (OOF pooled)",
                    "mean $\\sigma_{\\mathrm{total}}$",
                ],
                {
                    "metal_species_pair": "pair",
                    "n_heldout_rows": "held-out rows",
                    "n_folds_represented": "folds",
                    "n_high_selectivity_events": "high-sel. events",
                },
            )
        )
    s("")
    si.extend(si_figure("S13", "Held-out pair support versus high-selectivity event support.", "figS13_pair_event_support.png", "Out-of-fold held-out labels only. The plotted pooled per-pair RMSE is not MoP RMSE. The plot describes pair-specific data support and does not represent a chemical mechanism analysis."))
    if len(pair_event_summary):
        r0 = pair_event_summary.iloc[0]
        n_heldout_pairs = int(r0["n_pairs_with_heldout_rows"])
        event_col = f"n_pairs_gt_{int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)}_high_selectivity_events"
        if event_col in r0.index:
            n_gt_events = int(r0[event_col])
        else:
            n_gt_events = int(r0["n_pairs_gt_10_high_selectivity_events"])
        s(
            "Across the pooled out-of-fold labels, "
            + f"{n_heldout_pairs:,}"
            + " species-level pairs had held-out rows and "
            + f"{n_gt_events:,}"
            + " had "
            + pair_acq_event_elig_text().replace("held-out ", "high-selectivity ")
            + " when canonical ($y>1$) and reversed ($y<-1$) events were counted together. The descriptive pooled counts come from the registered pair-event-support artifacts. Source: `results/downstream_cv/pair_event_support/pair_event_support_summary.csv`. Per-pair event counts use `n_high_selectivity_events` in `results/downstream_cv/pair_event_support/pair_event_support_pooled.csv`. This pooled species-level count is not the fold-local ranking denominator in Table S10.1."
        )
        s("")
    s("Pair-level full-grid retention-curve AUC remains in Table S7.1.")
    s("## S9. Sign stability and interval interpretation")
    s("")
    s("This section answers whether 90% intervals resolve selectivity sign. It is supporting interpretation for the main-text claim that uncertainty ranks risk without certifying sign.")
    s("")
    s("Most 90% intervals overlap zero, so the intervals are not sign-certifying. Supplementary Figure S14 shows the sign-category fractions and corresponding sign accuracy.")
    s("")
    s("**Table S9.1. 90% interval sign categories (workhorse, all-pair scope).**")
    sign_show = sign[sign.metric_name.astype(str).isin(["fraction_rows", "sign_accuracy", "rmse", "mop_rmse"])].copy() if "metric_name" in sign.columns else sign
    sign_show = exclude_ree_scope(sign_show)
    if {"fold_mean", "fold_sd", "n_valid_folds"}.issubset(sign_show.columns):
        sign_show = sign_show.copy()
        sign_show["display"] = [_fmt_msd(a, b, n) for a, b, n in zip(sign_show.fold_mean, sign_show.fold_sd, sign_show.n_valid_folds)]
        s(md_table(sign_show, ["scope", "category", "metric_name", "display", "n_valid_folds"], {"metric_name": "metric", "display": "fold mean ± SD"}))
    else:
        s(md_table(sign_show, ["scope", "category", "metric_name", "fold_mean", "fold_sd", "n_valid_folds"]))
    s("")
    si.extend(si_figure("S14", "All-pair interval sign stability.", "figS14_sign_stability_detail.png", "Categories are based on whether the nominal 90% interval is entirely negative, entirely positive, or overlaps zero. The overlap-zero category dominates the all-pair scope."))
    s("## S10. Pair-acquisition analysis")
    s("")
    s("This section reports the sigma-only pair-acquisition study. This analysis does not ask whether the model is globally accurate. It asks whether, within a pair-specific candidate list, a ranking rule places useful event candidates near the top. The ranking is a retrospective held-out ranking simulation, not prospective experimental validation. Event precision@$k$ and true hit@$k$ are never interchangeable except at $k=1$. Retrospective regret@$k$ is reported on the same shortlists because those event metrics treat all events equally and do not measure the leftover gap to the best available ligand in a pair-task.")
    s("")
    s("Primary pair-acquisition comparisons. (H2) at least one prespecified sigma-only ranking rule, particularly the threshold-margin rule, recovers high-selectivity events ($\\Delta\\log K>1$) at higher event precision@$k$ and true hit@$k$ than random selection on the eligible pair-tasks. (H3) the prespecified sigma-only ranking rules improve near-zero event precision@$k$ and true hit@$k$ relative to random selection on the eligible pair-tasks, with the magnitude of improvement depending on the ranking rule and shortlist size. (H4) aggressive LCB-labeled uncertainty-penalty scores can help the near-zero task and hurt high-selectivity recovery because $\\sigma_{\\mathrm{total}}$ increases with $|\\widehat y|$. (H5) pair-tasks are eligible only when they have "
      + pair_acq_event_elig_text()
      + ", so reported event precision@$k$, true hit@$k$, and regret@$k$ exclude pair-tasks with at most "
      + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE))
      + " event"
      + ("" if int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE) == 1 else "s")
      + ". Retrospective regret@$k$ is a complementary post-ranking score on the same shortlists (Methods 2.6; Table S10.4; Figure 5 E--F); it was not used to change the ranking rules. Negative results are retained. No top-$k$ threshold is optimized on outer-test event precision@$k$, true hit@$k$, or regret@$k$. Legacy CSV field `hit_rate_at_k` is event precision@$k$; true hit@$k$ is `any_hit_at_k`. Retention hypothesis H1 is stated in S7. A supporting event-threshold sensitivity with high-selectivity events at $\\Delta\\log K>2$ and near-zero events at $|\\Delta\\log K|<1/2$ is Section S10.2 and Supplementary Figure S15; it does not replace Table 4 or Figure 5.")
    s("")
    s("Experiment artifacts: `configs/experiments/pair_acquisition_spec.yaml`; `results/downstream_cv/pair_acquisition/`.")
    s("")
    s("### S10.1 Sigma-only pair acquisition")
    s("")
    s("Experiment `sigma_pair_acquisition` ranks ligands inside each held-out metal-species pair using only $\\widehat y$ and $\\sigma_{\\mathrm{total}}$. Distance scores are not used as ranking inputs. A pair-task is eligible only if it has "
      + pair_acq_event_elig_text()
      + ": target-direction $\\Delta\\log K>1$ for high selectivity (canonical $y>1$ and reversed $y<-1$ as separate tasks) and $|\\Delta\\log K|<1$ for near-zero selectivity. Top-$k$ is $k\\in\\{1,5,10\\}$. A pair-task contributes to Precision@$k$, Hit@$k$, and retrospective regret@$k$ only when it has at least $k$ held-out rows. All ranking functions are oriented so that larger score values are ranked first; “more favorable” therefore means more favorable for the stated objective, not necessarily larger signed $\\Delta\\log K$. Ranking scores follow Methods 2.6. For an oriented high-selectivity task, the ranking scores are oriented predicted selectivity $\\widehat y^\\star$, oriented lower confidence bounds (LCB) $\\widehat y^\\star-z\\sigma_{\\mathrm{total}}$ with $z=1,2,3$, and the standardized threshold-margin score $(\\widehat y^\\star-1)/\\sigma_{\\mathrm{total}}$. For near-zero selection, candidates are ranked from low to high predicted magnitude. The predicted-magnitude score is $-|\\widehat y|$, the uncertainty-penalty variants are $-(|\\widehat y|+z\\sigma_{\\mathrm{total}})$, and the standardized margin is $(1-|\\widehat y|)/\\sigma_{\\mathrm{total}}$. In tables and figures, the prespecified LCB-labeled uncertainty-penalty rules appear as LCB-1, LCB-2, and LCB-3. The threshold-margin score is that specified standardized margin, not the Gaussian expected-improvement integral. Aggregation is a pair-equal mean inside each fold, then fold mean ± SD. We evaluate a retrospective held-out ranking simulation. Candidates are ranked before their held-out labels are used for scoring, but observed held-out event support is used to define eligibility; no prospective experiment is performed. Retrospective regret@$k$ is scored after ranking on the same shortlists (Methods 2.6); it does not change the ranking rule. The phrase “prospective-style” is used only as a structural analogy for that ranking-before-label-scoring design; this analysis does not constitute prospective experimental validation. Eligibility is $n_{\\mathrm{event}}>"
      + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE))
      + "$.")
    s("")
    if len(pair_acq_elig):
        s("**Table S10.1. Eligible pair counts.**")
        s(note_block("High-selectivity rows count metal-species pairs with "
          + pair_acq_event_elig_text()
          + " in that orientation. `high_both_orientations` counts pair-tasks (a pair can contribute both orientations). Low-selectivity has one task per pair because $|y|$ is orientation-invariant. " + source_note("results/downstream_cv/pair_acquisition/pair_acquisition_eligibility_summary.csv", "manuscripts/final/tables/pair_acquisition_eligibility_summary.csv")))
        elig_show = pair_acq_elig.copy()
        elig_show["display"] = [_fmt_msd(a, b, n, digits=1) for a, b, n in zip(elig_show.fold_mean, elig_show.fold_sd, elig_show.n_valid_folds)]
        s(md_table(elig_show, ["cohort", "metric_name", "display", "n_valid_folds"], {"metric_name": "metric", "display": "fold mean ± SD"}))
        s("")
    if len(pair_acq_cmp):
        s("**Table S10.2. Event precision@$k$ at $k=1,5,10$.**")
        s(note_block("Event precision@$k$ is the fraction of the top $k$ ranked ligands that are observed events. This is the legacy `hit_rate_at_k` field, previously displayed as hit@$k$. High-selectivity event = observed target-direction $\\Delta\\log K>1$. Near-zero event = observed $|\\Delta\\log K|<1$. Random is 20 within-pair permutations (seed 0). $n@k$ is the fold-mean number of pair-tasks with at least $k$ held-out rows (the denominator for that $k$). Eligible pair-task counts at $k=1$ are Table S10.1. True hit@$k$ is Table 4 and Table S10.3. Retrospective regret@$k$ is Table 4 and Table S10.4. " + source_note("results/downstream_cv/pair_acquisition/pair_acquisition_comparison.csv", "manuscripts/final/tables/pair_acquisition_comparison.csv")))
        cmp_show = pair_acq_cmp[pair_acq_cmp.orientation.astype(str).isin(["both", "near_zero"])].copy()
        if "strategy" in cmp_show.columns:
            cmp_show["strategy"] = cmp_show["strategy"].map(
                lambda x: {
                    "expected_improvement": "threshold-margin score",
                    "y_pred": "predicted value",
                    "lcb_1sigma": "LCB-1",
                    "lcb_2sigma": "LCB-2",
                    "lcb_3sigma": "LCB-3",
                    "random": "random",
                }.get(str(x), str(x))
            )
        cmp_show["precision@1"] = [_fmt_msd(a, b) for a, b in zip(cmp_show.hit_rate_at_1, cmp_show.hit_rate_at_1_sd)]
        cmp_show["precision@5"] = [_fmt_msd(a, b) for a, b in zip(cmp_show.hit_rate_at_5, cmp_show.hit_rate_at_5_sd)]
        cmp_show["precision@10"] = [_fmt_msd(a, b) for a, b in zip(cmp_show.hit_rate_at_10, cmp_show.hit_rate_at_10_sd)]
        for k in (1, 5, 10):
            mean_c = f"n_tasks_defined_at_{k}"
            sd_c = f"n_tasks_defined_at_{k}_sd"
            if mean_c in cmp_show.columns:
                cmp_show[f"n@{k}"] = [_fmt_msd(a, b, digits=1) for a, b in zip(cmp_show[mean_c], cmp_show[sd_c])]
            else:
                cmp_show[f"n@{k}"] = [_fmt_msd(a, b, digits=1) for a, b in zip(cmp_show.n_tasks_fold_mean, cmp_show.n_tasks_fold_sd)]
        s(md_table(cmp_show, ["task", "strategy", "precision@1", "precision@5", "precision@10", "n@1", "n@5", "n@10"]))
        s("")
    if len(pair_acq_hit):
        s("**Table S10.3. True hit@$k$ at $k=1,5,10$.**")
        s(note_block("True hit@$k$ is $1[\\sum_{j=1}^{k}I_j>0]$. Eligible pair-tasks have "
          + pair_acq_event_elig_text()
          + ". A pair-task contributes to a given $k$ only if it has at least $k$ held-out rows; $n@k$ is that pair-task count. Aggregation = pair-equal mean inside each fold, then fold mean ± SD. This is a retrospective held-out ranking simulation. Retrospective regret@$k$ on the same shortlists is Table 4 and Table S10.4. " + source_note("results/downstream_cv/pair_acquisition/pair_acquisition_hit_and_precision.csv", "manuscripts/final/tables/pair_acquisition_hit_and_precision.csv")))
        hit_show = pair_acq_hit[pair_acq_hit.orientation.astype(str).isin(["both", "near_zero"])].copy()
        if "strategy" in hit_show.columns:
            hit_show["strategy"] = hit_show["strategy"].map(
                lambda x: {
                    "expected_improvement": "threshold-margin score",
                    "y_pred": "predicted value",
                    "lcb_1sigma": "LCB-1",
                    "lcb_2sigma": "LCB-2",
                    "lcb_3sigma": "LCB-3",
                    "random": "random",
                }.get(str(x), str(x))
            )
        hit_show["hit@1"] = [_fmt_msd(a, b) for a, b in zip(hit_show.hit_at_1, hit_show.hit_at_1_sd)]
        hit_show["hit@5"] = [_fmt_msd(a, b) for a, b in zip(hit_show.hit_at_5, hit_show.hit_at_5_sd)]
        hit_show["hit@10"] = [_fmt_msd(a, b) for a, b in zip(hit_show.hit_at_10, hit_show.hit_at_10_sd)]
        for k in (1, 5, 10):
            mean_c = f"n_tasks_defined_at_{k}"
            sd_c = f"n_tasks_defined_at_{k}_sd"
            if mean_c in hit_show.columns:
                hit_show[f"n@{k}"] = [_fmt_msd(a, b, digits=1) for a, b in zip(hit_show[mean_c], hit_show[sd_c])]
            else:
                hit_show[f"n@{k}"] = [_fmt_msd(a, b, digits=1) for a, b in zip(hit_show.n_tasks_fold_mean, hit_show.n_tasks_fold_sd)]
        s(md_table(hit_show, ["task", "strategy", "hit@1", "hit@5", "hit@10", "n@1", "n@5", "n@10"]))
        s("")
    if len(pair_acq_regret):
        s("**Table S10.4. Retrospective regret@$k$ at $k=1,5,10$.**")
        s(note_block("Regret@$k$ is the leftover objective gap between the best ligand in the pair-task pool $\\mathcal{P}$ and the best ligand in the selected shortlist $B_k$ (Methods 2.6). High-selectivity regret uses target-direction $\\Delta\\log K$ (canonical $y$, reversed $-y$). Near-zero regret uses $|\\Delta\\log K|$. Lower is better. Zero means the shortlist contains a pool-optimal ligand, including tied optima. Contains optimum@$k$ is the pair-task fraction of those shortlists (equivalently, the fraction with regret@$k$=0). Pool optimum is the pair-equal mean of those best pool values and does not depend on the ranking rule. Rankings are the same shortlists as Table 4 and Tables S10.2 and S10.3. Event metrics treat all events equally; regret measures the leftover gap. A pair-task contributes to a given $k$ only if it has at least $k$ held-out rows; $n@k$ is that pair-task count. Aggregation = pair-equal mean inside each fold, then fold mean ± SD. Fold-level pair-equal means: `pair_acquisition_regret_fold.csv`. " + source_note("results/downstream_cv/pair_acquisition/pair_acquisition_regret.csv", "manuscripts/final/tables/pair_acquisition_regret.csv")))
        rg_show = pair_acq_regret[pair_acq_regret.orientation.astype(str).isin(["both", "near_zero"])].copy()
        if "strategy" in rg_show.columns:
            rg_show["strategy"] = rg_show["strategy"].map(
                lambda x: {
                    "expected_improvement": "threshold-margin score",
                    "y_pred": "predicted value",
                    "lcb_1sigma": "LCB-1",
                    "lcb_2sigma": "LCB-2",
                    "lcb_3sigma": "LCB-3",
                    "random": "random",
                }.get(str(x), str(x))
            )
        rg_show["regret@1"] = [_fmt_msd(a, b) for a, b in zip(rg_show.regret_at_1, rg_show.regret_at_1_sd)]
        rg_show["regret@5"] = [_fmt_msd(a, b) for a, b in zip(rg_show.regret_at_5, rg_show.regret_at_5_sd)]
        rg_show["regret@10"] = [_fmt_msd(a, b) for a, b in zip(rg_show.regret_at_10, rg_show.regret_at_10_sd)]
        if "selected_contains_optimum_at_1" in rg_show.columns:
            rg_show["contains opt@1"] = [_fmt_msd(a, b) for a, b in zip(rg_show.selected_contains_optimum_at_1, rg_show.selected_contains_optimum_at_1_sd)]
            rg_show["contains opt@5"] = [_fmt_msd(a, b) for a, b in zip(rg_show.selected_contains_optimum_at_5, rg_show.selected_contains_optimum_at_5_sd)]
            rg_show["contains opt@10"] = [_fmt_msd(a, b) for a, b in zip(rg_show.selected_contains_optimum_at_10, rg_show.selected_contains_optimum_at_10_sd)]
        if "pool_optimum" in rg_show.columns:
            rg_show["pool optimum"] = [_fmt_msd(a, b) for a, b in zip(rg_show.pool_optimum, rg_show.pool_optimum_sd)]
        for k in (1, 5, 10):
            mean_c = f"n_tasks_defined_at_{k}"
            sd_c = f"n_tasks_defined_at_{k}_sd"
            if mean_c in rg_show.columns:
                rg_show[f"n@{k}"] = [_fmt_msd(a, b, digits=1) for a, b in zip(rg_show[mean_c], rg_show[sd_c])]
            else:
                rg_show[f"n@{k}"] = [_fmt_msd(a, b, digits=1) for a, b in zip(rg_show.n_tasks_fold_mean, rg_show.n_tasks_fold_sd)]
        s(md_table(rg_show, ["task", "strategy", "regret@1", "regret@5", "regret@10", "contains opt@1", "contains opt@5", "contains opt@10", "pool optimum", "n@1", "n@5", "n@10"]))
        s("")
        hs_pool = pair_acq_regret[
            (pair_acq_regret.task.astype(str) == "high_selectivity")
            & (pair_acq_regret.orientation.astype(str).isin(["both"]))
        ]
        hs_pool_txt = "NA"
        if len(hs_pool) and {"pool_optimum", "pool_optimum_sd"}.issubset(hs_pool.columns):
            hs_pool_txt = _fmt_msd(hs_pool.iloc[0].pool_optimum, hs_pool.iloc[0].pool_optimum_sd)
        s(
            f"The fold-aggregated high-selectivity pool optimum is {hs_pool_txt}, well above the event threshold of 1, so high-selectivity regret measures the gap to the strongest available observed candidate rather than merely the gap to threshold qualification. "
            "Near-zero regret is the excess absolute selectivity relative to the closest-to-zero ligand in the eligible pool. Both regret quantities are expressed in $\\Delta\\log K$ units."
        )
        s("")
    s("The corresponding ranking figure is main-text Figure 5 (`fig05_pair_acquisition.png`), with high selectivity in the left column (A, C, E) and near-zero selectivity in the right column (B, D, F); rows are true hit@$k$, event precision@$k$, and retrospective regret@$k$. Main-text Table 4 reports the compact $k=1,5$ subset, including regret@$k$. Eligible pair-tasks have "
      + pair_acq_event_elig_text()
      + ". Ranking uses only predicted selectivity and $\\sigma_{\\mathrm{total}}$. Error bars are fold SD after a pair-equal mean. Source = `results/downstream_cv/pair_acquisition/pair_acquisition_summary.csv`.")
    s("")
    if len(pair_acq_bins):
        s("**Table S10.5. Workhorse RMSE and $\\sigma_{\\mathrm{total}}$ versus $|\\widehat y|$.**")
        s(note_block("RMSE of bin $S$ is $\\sqrt{\\mathrm{mean}_{i\\in S}(y_i-\\widehat y_i)^2}$ on held-out rows in that fold, then fold mean ± SD. It aggregates the same residuals used as $|e_i|$ in Table S10.6; it is not a per-row error and not pair-level MoP RMSE. The $|\\widehat y|<1$ and $|\\widehat y|\\ge 1$ rows are the two operational regimes. " + source_note("results/downstream_cv/pair_acquisition/pair_acquisition_calibration_bin_summary.csv", "manuscripts/final/tables/pair_acquisition_calibration_bin_summary.csv")))
        keep_bins = {
            "abs_yhat_0-0.5",
            "abs_yhat_0.5-1",
            "abs_yhat_1-2",
            "abs_yhat_2-4",
            "abs_yhat_4-8",
            "abs_yhat_>8",
            "abs_yhat_lt_1",
            "abs_yhat_ge_1",
        }
        bin_show = pair_acq_bins[pair_acq_bins.bin_id.astype(str).isin(keep_bins) & pair_acq_bins.metric_name.astype(str).isin(["rmse", "mean_sigma", "n_rows", "picp_90"])].copy()
        bin_show["display"] = [_fmt_msd(a, b, n) for a, b, n in zip(bin_show.fold_mean, bin_show.fold_sd, bin_show.n_valid_folds)]
        s(md_table(bin_show, ["bin_id", "metric_name", "display", "n_valid_folds"], {"bin_id": "bin", "metric_name": "metric", "display": "fold mean ± SD"}))
        s("")
    if len(pair_acq_rho):
        s("**Table S10.6. Rank correlations of $\\sigma_{\\mathrm{total}}$ and $|\\widehat y|$ with the row-level absolute residual.**")
        s(note_block("The residual on row $i$ is $e_i=y_i-\\widehat y_i$ with $y=\\Delta\\log K$. The absolute residual is $|e_i|=|y_i-\\widehat y_i|$. This is a per-row quantity, not RMSE. RMSE of a set $S$ is $\\sqrt{\\mathrm{mean}_{i\\in S}e_i^2}$, computed inside each held-out fold (Table S10.5). Each fold computes a rank correlation on all held-out rows; the displayed value is fold mean ± SD. The analysis is descriptive and does not claim formal conditional independence. No threshold-sweep curve is reported. " + source_note("results/downstream_cv/pair_acquisition/pair_acquisition_calibration_rho_summary.csv", "manuscripts/final/tables/pair_acquisition_calibration_rho_summary.csv")))
        rho_show = pair_acq_rho.copy()
        rho_show["display"] = [_fmt_msd(a, b, n) for a, b, n in zip(rho_show.fold_mean, rho_show.fold_sd, rho_show.n_valid_folds)]
        s(md_table(rho_show, ["metric_name", "display", "n_valid_folds"], {"metric_name": "pair", "display": "fold mean ± SD"}))
        s("")
        s("Because $\\sigma_{\\mathrm{total}}$ increases with predicted selectivity magnitude, we compared its rank association with absolute residuals against the corresponding association for $|\\widehat y|$. Across held-out rows, $\\sigma_{\\mathrm{total}}$ had a stronger association with $|y-\\widehat y|$ than $|\\widehat y|$ did. This supports interpreting $\\sigma_{\\mathrm{total}}$ as a composite predictive-risk score, while not implying that it is a purely epistemic uncertainty measure.")
        s("")
    if len(mag_adj):
        s("**Table S10.7. Magnitude-adjusted risk ordering for the selected workhorse.**")
        s(
            note_block(
                "Primary estimand: fold-local partial rank correlation $\\rho(\\sigma_{\\mathrm{total}},|y-\\widehat y|\\mid|\\widehat y|)$ on the same held-out fold-local predictions used for the primary retention analysis. Concatenated held-out rows are not used as a headline. Pairwise rank correlations are recomputed from the same fold scores and match Table S10.6. The 50% retained-subset MoP RMSE by $|\\widehat y|$ (higher $|\\widehat y|$ = greater predicted risk) is a sensitivity comparator only; it is not a canonical retention score and is not written into `fold_retention.csv`. The association is treated as remaining substantial when the five-fold mean partial correlation is at least 0.20 and the mean minus one fold SD remains positive. This does not establish causal independence. "
                + source_note(
                    "results/downstream_cv/pair_acquisition/magnitude_adjusted_risk_summary.csv",
                    "manuscripts/final/tables/magnitude_adjusted_risk_summary.csv",
                )
            )
        )
        mag_labels = {
            "rho_sigma_abs_error": r"$\rho(\sigma_{\mathrm{total}},\lvert y-\widehat y\rvert)$",
            "rho_sigma_abs_yhat": r"$\rho(\sigma_{\mathrm{total}},\lvert\widehat y\rvert)$",
            "rho_abs_yhat_abs_error": r"$\rho(\lvert\widehat y\rvert,\lvert y-\widehat y\rvert)$",
            "rho_partial_sigma_error_given_abs_yhat": r"$\rho(\sigma_{\mathrm{total}},\lvert y-\widehat y\rvert\mid\lvert\widehat y\rvert)$",
            "mop_rmse_at_50pct_abs_yhat": r"all-pair MoP RMSE at 50% retention by $\lvert\widehat y\rvert$",
            "retention_curve_auc_mop_rmse_abs_yhat": r"MoP RMSE retention-curve AUC by $\lvert\widehat y\rvert$",
        }
        mag_show = mag_adj.copy()
        order = {
            "rho_sigma_abs_error": 0,
            "rho_sigma_abs_yhat": 1,
            "rho_abs_yhat_abs_error": 2,
            "rho_partial_sigma_error_given_abs_yhat": 3,
            "mop_rmse_at_50pct_abs_yhat": 4,
            "retention_curve_auc_mop_rmse_abs_yhat": 5,
        }
        mag_show["_ord"] = mag_show.metric_name.astype(str).map(order).fillna(9)
        mag_show = mag_show.sort_values("_ord").drop(columns=["_ord"])
        mag_show["quantity"] = mag_show.metric_name.astype(str).map(lambda x: mag_labels.get(x, x))
        mag_show["display"] = [_fmt_msd(a, b, n) for a, b, n in zip(mag_show.fold_mean, mag_show.fold_sd, mag_show.n_valid_folds)]
        s(md_table(mag_show, ["quantity", "display", "n_valid_folds"], {"display": "fold mean ± SD"}))
        s("")
        s(
            "Ranking held-out rows by $|\\widehat y|$ reduced 50% retained-subset all-pair MoP RMSE relative to full retention, but $\\sigma_{\\mathrm{total}}$ remained the lower-error ranking in Table 3. Combined with the partial rank correlation, this supports an additional empirical risk-ordering contribution from $\\sigma_{\\mathrm{total}}$ under the present task without implying universal independence of $|\\widehat y|$."
        )
        s("")
    s("The corresponding magnitude figure is main-text Figure 4 (`fig04_predicted_magnitude.png`).")
    s("")
    def _acq_bin(bin_id, metric):
        if pair_acq_bins.empty:
            return "NA"
        hit = pair_acq_bins[(pair_acq_bins.bin_id.astype(str) == bin_id) & (pair_acq_bins.metric_name.astype(str) == metric)]
        if hit.empty:
            return "NA"
        return _fmt_msd(hit.iloc[0].fold_mean, hit.iloc[0].fold_sd, hit.iloc[0].n_valid_folds)

    s(
        "Near-zero recovery is easier than high-selectivity recovery mainly because row-level RMSE is smaller at low predicted magnitude "
        f"(Table S10.5: RMSE {_acq_bin('abs_yhat_lt_1', 'rmse')} for $|\\widehat y|<1$ versus {_acq_bin('abs_yhat_ge_1', 'rmse')} for $|\\widehat y|\\ge 1$) "
        "and because eligible low-selectivity pairs already have a high event rate. "
        "$\\sigma_{\\mathrm{total}}$ increases with $|\\widehat y|$ and is associated with the row-level absolute residual $|y-\\widehat y|$ (Table S10.6). "
        "The magnitude-adjusted partial rank correlation remains in Table S10.7. "
        "That scaling helps lower-confidence-bound scores on the near-zero task, where a large $\\sigma$ correctly marks a predicted-near-zero ligand as unreliable. "
        "The same scaling hurts aggressive LCB-$2$/$3$ on the high-selectivity task, because large $\\sigma$ at large $|\\widehat y|$ demotes true high-selectivity ligands. "
        "Interval ACE is not uniformly better at low $|y|$; the operational advantage is lower row-level RMSE plus $\\sigma$–magnitude coupling rather than a molecular-distance applicability domain. These RMSE values are not pair-level MoP RMSE."
    )
    s("")
    s("### S10.2 Event-threshold sensitivity ($\\Delta\\log K>2$ and $|\\Delta\\log K|<1/2$)")
    s("")
    s(
        "Experiment `pair_acquisition_threshold_2_half` repeats the sigma-only ranking protocol on the same stored fold-local scores after changing only the event definitions. High-selectivity events are target-direction $\\Delta\\log K>2$ (canonical $y>2$, reversed $y<-2$). Near-zero events are $|\\Delta\\log K|<1/2$. Eligibility, event labels, and the threshold-margin ranking use these cutoffs, so the threshold-margin scores are $(\\widehat y-2)/\\sigma$ and $(1/2-|\\widehat y|)/\\sigma$. Predicted-value and LCB rankings do not use the event cutoff directly; they change only through the revised eligible pair-task pools. The threshold-margin ranking uses the revised cutoff explicitly. This supporting sensitivity does not replace Table 4 or Figure 5. Artifacts: `configs/experiments/pair_acquisition_threshold_sensitivity_spec.yaml`; `results/downstream_cv/pair_acquisition_tau_2_0p5/`."
    )
    s("")
    if len(pair_acq_sens_hit):
        s("**Table S10.8. True hit@$k$, event precision@$k$, and retrospective regret@$k$ at the stricter event thresholds (fold mean ± SD).**")
        s(
            note_block(
                "Compact $k=1,5$ subset in the same layout as Table 4. High-selectivity event = observed target-direction $\\Delta\\log K>2$. Near-zero event = observed $|\\Delta\\log K|<1/2$. The threshold-margin score is $(\\widehat y-2)/\\sigma$ or $(1/2-|\\widehat y|)/\\sigma$. Rankings, splits, and stored fold scores are otherwise the same as the primary study. A pair-task contributes to a given $k$ only if it has at least $k$ held-out rows; $n@k$ is that pair-task count. Aggregation = pair-equal mean inside each fold, then fold mean ± SD. This supporting sensitivity does not replace Table 4 or Figure 5. Source: `results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_hit_and_precision.csv` and `results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_regret.csv`; manuscript rendering copies: `manuscripts/final/tables/pair_acquisition_tau2_half_hit_and_precision.csv` and `manuscripts/final/tables/pair_acquisition_tau2_half_regret.csv`."
            )
        )
        tbl_s = pair_acq_compact_md(pair_acq_sens_hit, pair_acq_sens_regret)
        if tbl_s:
            s(tbl_s)
        s("")
    if len(pair_acq_sens_elig):
        s("**Table S10.9. Eligible pair counts at the stricter event thresholds.**")
        s(
            note_block(
                "Same eligibility rule as Table S10.1 ($n_{\\mathrm{event}}>"
                + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE))
                + "$), with high-selectivity events at target-direction $\\Delta\\log K>2$ and near-zero events at $|\\Delta\\log K|<1/2$. `high_both_orientations` counts oriented pair-tasks. This table does not replace Table S10.1. "
                + source_note(
                    "results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_eligibility_summary.csv",
                    "manuscripts/final/tables/pair_acquisition_tau2_half_eligibility_summary.csv",
                )
            )
        )
        elig_s = pair_acq_sens_elig.copy()
        elig_s["display"] = [_fmt_msd(a, b, n, digits=1) for a, b, n in zip(elig_s.fold_mean, elig_s.fold_sd, elig_s.n_valid_folds)]
        s(md_table(elig_s, ["cohort", "metric_name", "display", "n_valid_folds"], {"metric_name": "metric", "display": "fold mean ± SD"}))
        s("")
    si.extend(
        si_figure(
            "S15",
            "Supporting event-threshold sensitivity for pair acquisition.",
            "figS15_pair_acquisition_threshold_sensitivity.png",
            "High-selectivity events are target-direction $\\Delta\\log K>2$; near-zero events are $|\\Delta\\log K|<1/2$. The plotted ranking families and panel layout match Figure 5 (left column high selectivity; right column near-zero). This supporting sensitivity does not replace Figure 5 or Table 4. Table S10.8 reports the compact $k=1,5$ subset.",
        )
    )
    write_si_s11(s, exp)
    write_si_s12(s, exp, status)
    s("## S13. Supplementary formulation and pair-frequency diagnostics")
    s("")
    s("The analyses in this section are restricted diagnostics. They appear after the all-pair model, retention, support, sign, acquisition, additional-diagnostic, and artifact-inventory sections. They are not the primary DS2 evaluation and do not reopen that argument. The primary evaluation uses all eligible metal pairs and MoP metrics. These restricted analyses illustrate pair-frequency structure and examine a formulation-specific direct-versus-propagated uncertainty comparison. Figure callouts for this section are collected in S13.4 in numerical order.")
    s("")
    s("A rare-earth-element pair is a pair in which both metal species are rare-earth elements. We use “REE–REE” as shorthand for this restricted subset below.")
    s("")
    s("### S13.1 Restricted rare-earth-element pair definition")
    s("")
    s(f"The restricted rare-earth-element pair (REE–REE) subset contains {sc['n_ree_rows']:,} DS2 rows and {sc['n_ree_species_pairs']} species-level pairs. {sc['n_ree_highsel_rows']:,} of those rows also satisfy $|y|>1$ (composition count only; combined REE∩high-selectivity CV is `not_computed`). These counts describe a chemically restricted family and are not a second general benchmark.")
    s("")
    s("**Table S13.1. Selected-workhorse restricted rare-earth-element pair metrics.**")
    s(note_block("Restricted formulation diagnostic for the selected workhorse only. This scope is not used for workhorse selection and is not a second general benchmark. The complete all-model REE grid remains in `manuscripts/final/tables/ds2_ree_pairs.csv`. PICP@90 and ACE here are pair-equal MoP summaries (`mop_picp_90`, `mop_ace`). Minimum pair support for MoP = 3 held-out ligands. Fold aggregation = mean ± SD. " + source_note("manuscripts/final/tables/ds2_ree_pairs.csv")))
    s(md_table(ree_case, ["display_name", "mop_rmse", "mop_r2", "sign_accuracy", "mop_picp_90", "mop_ace"], {"display_name": "model", "mop_r2": "MoP $R^2$ (aux)", "mop_picp_90": "MoP PICP@90", "mop_ace": "MoP ACE"}))
    s("")
    s("**Table S13.2. Secondary row-weighted diagnostics, restricted rare-earth-element pair subset.**")
    s(note_block("Workhorse-only restricted slice. These are not competing primary estimates and were not used for workhorse selection. Row-level PICP@90 and ACE are REE-scope row-weighted $\\Delta\\log K$ coverage. Scope = `ree_pairs`."))
    s(md_table(ree_case, ["display_name", "row_rmse", "row_r2", "sign_accuracy", "picp_90", "ace"], {"display_name": "model", "row_r2": "row-level $R^2$", "picp_90": "row-level PICP@90", "ace": "row-level ACE"}))
    s("")
    s("Fold-local MoP RMSE and auxiliary MoP $R^2$ for the selected workhorse on this restricted subset are shown in Supplementary Figure S19.")
    s("")
    s("### S13.2 Pair-frequency, support, retention, and sign-stability statistics")
    s("")
    s("The restricted rare-earth-element pair subset is useful for illustrating the uneven frequency and support structure of particular chemical pair families. These examples are descriptive and should not be substituted for the all-pair MoP analysis. Workhorse REE retained-pair counts at full retention are 119 / 120 / 120 / 113 / 122. Restricted MoP retention curves are Supplementary Figure S17.")
    s("")
    s("**Table S13.3. Restricted rare-earth-element pair-level retention diagnostics.**")
    s(note_block("Pair-level retention-curve AUC in the REE–REE scope. Restricted diagnostic. Empirical pair-level error-ordering summary; see S7 for interpretation. Not used for workhorse selection and not a second general benchmark. " + source_note("results/downstream_cv/ad_retention/retention_auc_summary.csv", "manuscripts/final/tables/retention_auc_summary.csv") + " Endpoint field = `retention_curve_auc_mop_rmse`."))
    ree_pair = pair_ad_tbl[pair_ad_tbl.scope == "ree_pairs"] if "scope" in pair_ad_tbl.columns else ad[ad.scope == "ree_pairs"]
    s8_aurc = "pair_aurc" if "pair_aurc" in ree_pair.columns else "aurc"
    s8_mae = "mop_rmse_at_50" if "mop_rmse_at_50" in ree_pair.columns else "rmse_at_50pct_retention"
    if "score_display" not in ree_pair.columns:
        ree_pair = ree_pair.copy()
        ree_pair["score_display"] = ree_pair["score_name"].map(score_display_name)
    s(md_table(ree_pair, ["score_display", s8_aurc, s8_mae, "n_valid_folds"], {"score_display": "score", "pair_aurc": "MoP RMSE retention-curve AUC", "aurc": "MoP RMSE retention-curve AUC", "mop_rmse_at_50": "MoP RMSE@50%", "rmse_at_50pct_retention": "MoP RMSE@50%"}))
    s("")
    if len(ad50):
        s("**Table S13.4. Restricted rare-earth-element 50% retention comparison.**")
        s(note_block("Selected workhorse only. Restricted diagnostic; not used for workhorse selection. Full MoP RMSE is the unfiltered 100% REE–REE workhorse MoP RMSE. Pair counts are REE pairs contributing to retained MoP at 50%. Source: `results/downstream_cv/ad_retention/retention_50pct.csv`."))
        d2 = ad50.copy()
        if "primary_comparison" in d2.columns:
            d2 = d2[d2.primary_comparison.astype(bool)]
        d2 = d2[
            (d2.dataset == "DS2")
            & (d2.model_id.astype(str) == str(numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO")))
            & (d2.scope == "ree_pairs")
            & (d2.metric_name == "mop_rmse")
            & (d2.score_name.astype(str).isin(["sigma_total", "mahalanobis_feature_centroid", "tanimoto_nn_distance"]))
        ]
        score_order = {"sigma_total": 0, "mahalanobis_feature_centroid": 1, "tanimoto_nn_distance": 2}
        d2 = d2.copy()
        d2["_ord"] = d2.score_name.astype(str).map(score_order)
        d2 = d2.sort_values("_ord")
        rows2 = []
        for _, m in d2.iterrows():
            n = int(m.n_valid_folds)
            pair_mean = getattr(m, "pairs_contributing_to_MoP_mean", None)
            pair_sd = getattr(m, "pairs_contributing_to_MoP_sd", None)
            rows2.append(
                {
                    "score": score_display_name(m.score_name),
                    "MoP RMSE@50%": _fmt_msd(m.retained_mean, m.retained_sd, n),
                    "full MoP RMSE": fmt.get("wh_ree_mop_rmse", "NA"),
                    "pairs contributing to retained MoP at 50%": _fmt_msd(pair_mean, pair_sd, n, digits=1),
                    "relative RMSE change": "NA" if pd.isna(m.relative_change) else f"{100.0 * float(m.relative_change):+.1f}%",
                }
            )
        if rows2:
            s(md_table(pd.DataFrame(rows2), list(rows2[0].keys())))
        s("")
    ree_sign = sign[sign.scope.astype(str) == "ree_pairs"].copy() if "scope" in sign.columns else pd.DataFrame()
    ree_pos = ree_sign[(ree_sign.category.astype(str) == "entirely_positive") & (ree_sign.metric_name.astype(str) == "fraction_rows")] if len(ree_sign) else pd.DataFrame()
    ree_pos_pct = f"{100.0 * float(ree_pos.iloc[0].fold_mean):.1f}%" if len(ree_pos) else "NA"
    s("**Table S13.5. Restricted rare-earth-element pair 90% interval sign categories.**")
    s(note_block(f"Descriptive interval-sign diagnostic for the restricted rare-earth-element pair subset. The entirely-positive category is {ree_pos_pct} of REE–REE rows. This table is not a primary accuracy result and is not used for workhorse selection. Source = `manuscripts/final/tables/sign_stability_summary.csv`."))
    if len(ree_sign):
        ree_sign_show = ree_sign[ree_sign.metric_name.astype(str).isin(["fraction_rows", "sign_accuracy", "rmse", "mop_rmse"])].copy()
        if {"fold_mean", "fold_sd", "n_valid_folds"}.issubset(ree_sign_show.columns):
            ree_sign_show["display"] = [_fmt_msd(a, b, n) for a, b, n in zip(ree_sign_show.fold_mean, ree_sign_show.fold_sd, ree_sign_show.n_valid_folds)]
            s(md_table(ree_sign_show, ["scope", "category", "metric_name", "display", "n_valid_folds"], {"metric_name": "metric", "display": "fold mean ± SD"}))
        else:
            s(md_table(ree_sign_show, ["scope", "category", "metric_name", "fold_mean", "fold_sd", "n_valid_folds"]))
        s("")
    s("### S13.3 Direct versus propagated interval comparison")
    s("")
    s("Experiment `adhoc_differencing` compares direct DS2 selectivity modeling with ad hoc differencing of DS1 absolute-binding predictions. The comparison is a formulation diagnostic aligned on DS2 ligand-disjoint outer folds, held-out ligands, the three-ligand pair-support rule, and five-fold mean ± SD aggregation. The numerical inventory uses the restricted rare-earth-element pair subset and is not a second general model-selection grid. Older manuscript numbers are not reused.")
    s("")
    s("#### S13.3.1 Ad hoc difference construction")
    s("")
    s("For each held-out DS2 pair $(X,Y,L)$, the ad hoc estimate is $\\widehat{\\Delta\\log K}_{\\mathrm{ad\\,hoc}}=\\widehat{\\log K}(X,L)-\\widehat{\\log K}(Y,L)$, using the DS1 checkpoint that held ligand $L$ out. The reverse orientation is $\\widehat{\\Delta\\log K}_{Y>X}=-\\widehat{\\Delta\\log K}_{X>Y}$. Held-out targets are not used in fitting or in forming the difference. DS1 and DS2 fold identifiers are not interchangeable; alignment is by ligand identity.")
    s("")
    s("The ad hoc uncertainty comparison is restricted to DS1 models with complete fold-local predictive uncertainty. Propagated difference uncertainty is $\\sigma_{\\Delta}=\\sqrt{\\sigma_X^2+\\sigma_Y^2}$ from total predictive standard deviations. This independent-error expression does not include a covariance term. It is a diagnostic uncertainty for the difference of two absolute-binding predictions. It is not assumed to be calibrated uncertainty for a directly modeled selectivity target, and it is not called DS2 uncertainty. No covariance term is invented.")
    s("")
    s("Point-model ad hoc differencing is a contextual baseline only and is not pooled with uncertainty-aware models. Two named ad hoc comparators on the same held-out ligands as the direct DS2 grid are included in the operational CSV: RDKit MLP (point; no PICP/ACE) and Chemprop evidential (MoP PICP/ACE of propagated $\\sqrt{\\sigma_X^2+\\sigma_Y^2}$ intervals). Those restricted-subset MoP PICP/ACE values are not the row-weighted propagated diagnostics in Table S13.6 and Figure S16, and they are not workhorse-selection metrics.")
    s("")
    s("#### S13.3.2 Summary results")
    s("")
    s("**Table S13.6. Direct versus propagated uncertainty summary.**")
    s(note_block("Grouped by method class. Not a workhorse-selection grid. Primary metric = MoP RMSE on the aligned comparison subset. Supporting = MoP $R^2$ and sign accuracy. Propagated PICP@90 and ACE are diagnostic only. " + source_note("results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_comparison.csv", "manuscripts/final/tables/ad_hoc_selectivity_comparison.csv")))
    if adhoc_table_full:
        s(adhoc_table_full)
    elif adhoc_table:
        s(adhoc_table)
    s("")
    s("Under this restricted rare-earth-element pair comparison, uncertainty-aware ad hoc models "
      + adhoc_phrase
      + " the selected direct DS2 workhorse. This result is a formulation diagnostic and is not an all-pair DS2 model comparison. The strongest ad hoc uncertainty-aware model was Chemprop evidential ("
      + adhoc_best_uq
      + "); the ad hoc Evidential MLP ensemble was "
      + adhoc_evid_ens
      + ". Row-weighted propagated ad hoc 90% intervals were poorly calibrated (PICP@90 near 1; ACE typically $>0.3$). That is an empirical result for this independent-error diagnostic, not a hidden failure. Supplementary Figures S16 and S18 summarize the aligned comparison and the corresponding interval calibration.")
    s("")
    s("#### S13.3.3 Fold-level results")
    s("")
    s("**Table S13.7. Fold-level direct/ad hoc results.**")
    s(note_block("Each cell is one ligand-disjoint outer fold. Full metric inventory, including sign accuracy, MoP $R^2$, row RMSE, and propagated PICP/ACE, is in `results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_fold_metrics.csv`."))
    if len(adhoc_fold):
        ff = adhoc_fold.copy()
        ff = ff[(ff.selectivity_method.astype(str) == "ad_hoc_ds1_difference") & (ff.scope.astype(str) == "ree_pairs")]
        if len(ff):
            fold_metric = "mop_rmse" if "mop_rmse" in ff.columns else "mop_mae"
            wide = ff.pivot_table(index=["source_model", "source_model_id", "method_class"], columns="outer_fold", values=fold_metric, aggfunc="first").reset_index()
            wide = wide.rename(columns={"source_model": "model", "source_model_id": "model id", "method_class": "method class"})
            s(md_table(wide, list(wide.columns)))
    s("")
    s("### S13.4 Restricted-figure block")
    s("")
    s("The following figures belong to this restricted diagnostic and are placed here in document order as Supplementary Figures S16–S19. They are not part of the primary all-pair Results sequence.")
    s("")
    si.extend(si_figure(
        "S16",
        "Direct DS2 intervals versus propagated ad hoc intervals on the restricted REE–REE diagnostic.",
        "figS16_ad_hoc_interval_calibration.png",
        "Direct intervals are native intervals for $\\Delta\\log K$; ad hoc intervals are propagated diagnostics from two absolute-binding predictions under an independent-error assumption. This restricted diagnostic is descriptive and is not used for workhorse selection or the primary all-pair conclusions.",
    ))
    si.extend(si_figure("S17", "Restricted REE–REE diagnostic MoP retention curves.", "figS17_retention_ree.png", "Restricted diagnostic; not the primary all-pair retention result and not used for workhorse selection."))
    si.extend(si_figure(
        "S18",
        "Direct versus propagated intervals on the restricted REE–REE diagnostic.",
        "figS18_direct_vs_ad_hoc_selectivity.png",
        "Direct DS2 intervals and propagated ad hoc intervals are compared here as a restricted diagnostic. The analysis is not the primary performance evaluation, does not replace all-pair MoP metrics, and is not used to select the workhorse or establish the all-pair risk-ordering result.",
    ))
    si.extend(si_figure(
        "S19",
        "Fold stability on the restricted REE–REE diagnostic.",
        "figS19_ree_fold_stability.png",
        "Fold-local MoP RMSE and auxiliary MoP $R^2$ for the selected workhorse. This restricted diagnostic is not used for workhorse selection or for the primary all-pair conclusions.",
    ))
    s("### S13.5 Interpretation and limitations")
    s("")
    s("The restricted rare-earth-element pair diagnostic is consistent with the motivation for attaching uncertainty directly to $\\Delta\\log K$, but it should be interpreted only as a formulation-specific diagnostic. Direct DS2 training changes the target and its representation; it does not guarantee removal of the underlying measurement variability. The primary conclusions come from the all-pair DS2 MoP and downstream ranking analyses. Chemprop REE–REE PICP or ACE values are not part of the main Results sequence.")
    s("")
    s("These restricted analyses illustrate pair-frequency structure and a formulation-specific direct-versus-propagated uncertainty comparison. They are not the primary DS2 evaluation, were not used for workhorse selection, and were not used to establish the all-pair risk-ordering result. Complete all-model REE grids remain in `manuscripts/final/tables/ds2_ree_pairs.csv`. Fold-level ad hoc metrics remain in `results/downstream_cv/ad_hoc_selectivity/ad_hoc_selectivity_fold_metrics.csv`.")
    s("")

    _prose_out_dir().joinpath("supplementary_information.md").write_text("\n".join(si) + "\n", encoding="utf-8")

    extra_claims = [
        {
            "claim_id": "hplus|metal_complex|r2|with",
            "dataset": "DS1",
            "model_id": "DS1_GRAPH_DMPNN_POINT",
            "experiment_id": "hplus_policy",
            "scope": "metal_complex",
            "metric_name": "r2_train_with_H",
            "value_display": numbers["hplus_r2_with"],
            "artifact_path": "results/canonical/hplus_decision.json",
            "fold_level_source": "results/canonical/hplus_decision.json",
            "aggregation_method": "fold_mean_plus_sd_ddof1",
            "status": "canonical_cv_result",
        },
        {
            "claim_id": "hplus|metal_complex|r2|without",
            "dataset": "DS1",
            "model_id": "DS1_GRAPH_DMPNN_POINT",
            "experiment_id": "hplus_policy",
            "scope": "metal_complex",
            "metric_name": "r2_train_without_H",
            "value_display": numbers["hplus_r2_without"],
            "artifact_path": "results/canonical/hplus_decision.json",
            "fold_level_source": "results/canonical/hplus_decision.json",
            "aggregation_method": "fold_mean_plus_sd_ddof1",
            "status": "canonical_cv_result",
        },
        {
            "claim_id": "cmi_split|test|mae",
            "dataset": "DS1",
            "model_id": "M4_GRAPH_DMPNN_POINT_cmisplit",
            "experiment_id": "cmi_split_verification",
            "scope": "cmi_test_all_records",
            "metric_name": "mae",
            "value_display": f"{cmi_test['mae']:.3f}",
            "artifact_path": "results/stage1/stage1a_cmisplit_results.json",
            "aggregation_method": "single_split_historical",
            "status": "historical_result",
        },
        {
            "claim_id": "ds1|chemprop_point|metal_complex|mae",
            "dataset": "DS1",
            "model_id": "DS1_GRAPH_DMPNN_POINT",
            "experiment_id": "cmi_split_verification",
            "scope": "metal_complex",
            "metric_name": "mae",
            "value_display": chem_ds1_mae,
            "artifact_path": "results/canonical/ds1_chemprop_ligand_disjoint_mae.json",
            "fold_level_source": "results/cv_predictions/fold_test_predictions/DS1_GRAPH_DMPNN_POINT/",
            "aggregation_method": "fold_mean_plus_sd_ddof1",
            "status": "canonical_cv_result",
        },
        {
            "claim_id": "cmi_split|train_test|shared_ligands",
            "dataset": "DS1",
            "model_id": "M4_GRAPH_DMPNN_POINT_cmisplit",
            "experiment_id": "cmi_split_verification",
            "scope": "cmi_official_assignment",
            "metric_name": "n_ligands_shared_train_test",
            "value_display": str(cmi_overlap.get("n_ligands_shared_train_test", "")),
            "artifact_path": "results/canonical/cmi_split_ligand_overlap.json",
            "aggregation_method": "assignment_count",
            "status": "historical_result",
        },
        {
            "claim_id": "cmi_released_weights|test|fold_mean_mae",
            "dataset": "DS1",
            "model_id": "CMI_M4_BEST_MODEL_RELEASED",
            "experiment_id": "cmi_released_weight_eval",
            "scope": "cmi_test_all_records",
            "metric_name": "mae_fold_mean",
            "value_display": f"{float((cmi_repro.get('released_weights') or {}).get('evaluations', {}).get('cmi_test', {}).get('fold_mean_mae', float('nan'))):.3f}" if (cmi_repro.get("released_weights") or {}).get("evaluations") else "NA",
            "artifact_path": "results/stage1/cmi_reproduction_results.json",
            "aggregation_method": "mean_of_five_foldwise_metrics",
            "status": "historical_result",
        },
        {
            "claim_id": "cmi_released_weights|ext|ensemble_mae",
            "dataset": "DS1",
            "model_id": "CMI_M4_BEST_MODEL_RELEASED",
            "experiment_id": "cmi_released_weight_eval",
            "scope": "cmi_ext_all_records",
            "metric_name": "mae_ensemble",
            "value_display": (
                f"{float((((cmi_repro.get('released_weights') or {}).get('evaluations') or {}).get('cmi_ext') or {}).get('ensemble', {}).get('mae', float('nan'))):.3f}"
                if (((cmi_repro.get("released_weights") or {}).get("evaluations") or {}).get("cmi_ext") or {}).get("ensemble")
                else "NA"
            ),
            "artifact_path": "results/stage1/cmi_reproduction_results.json",
            "aggregation_method": "five_fold_ensemble",
            "status": "historical_result",
        },
        {
            "claim_id": "cmi_architecture_retrain|test|ensemble_mae",
            "dataset": "DS1",
            "model_id": "M4_GRAPH_DMPNN_POINT_cmiarch",
            "experiment_id": "cmi_architecture_retrain",
            "scope": "cmi_test_all_records",
            "metric_name": "mae_ensemble",
            "value_display": (
                f"{float((((cmi_repro.get('architecture_retrain') or {}).get('evaluations') or {}).get('cmi_test') or {}).get('ensemble', {}).get('mae', float('nan'))):.3f}"
                if str((cmi_repro.get("architecture_retrain") or {}).get("status", "")).lower() == "complete"
                else "NA"
            ),
            "artifact_path": "results/stage1/cmi_reproduction_results.json",
            "aggregation_method": "five_fold_ensemble",
            "status": "historical_result",
        },
        {
            "claim_id": "cmi_architecture_retrain|ext|ensemble_mae",
            "dataset": "DS1",
            "model_id": "M4_GRAPH_DMPNN_POINT_cmiarch",
            "experiment_id": "cmi_architecture_retrain",
            "scope": "cmi_ext_all_records",
            "metric_name": "mae_ensemble",
            "value_display": (
                f"{float((((cmi_repro.get('architecture_retrain') or {}).get('evaluations') or {}).get('cmi_ext') or {}).get('ensemble', {}).get('mae', float('nan'))):.3f}"
                if str((cmi_repro.get("architecture_retrain") or {}).get("status", "")).lower() == "complete"
                else "NA"
            ),
            "artifact_path": "results/stage1/cmi_reproduction_results.json",
            "aggregation_method": "five_fold_ensemble",
            "status": "historical_result",
        },
    ]
    for r in sign.itertuples(index=False):
        extra_claims.append(
            {
                "claim_id": f"sign|{r.scope}|{r.category}|{r.metric_name}",
                "dataset": "DS2",
                "model_id": numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO"),
                "experiment_id": "sign_stability",
                "scope": r.scope,
                "metric_name": f"{r.category}:{r.metric_name}",
                "value_display": f"{float(r.fold_mean):.3f} ± {float(r.fold_sd):.3f}",
                "fold_mean": r.fold_mean,
                "fold_sd": r.fold_sd,
                "n_valid_folds": r.n_valid_folds,
                "artifact_path": "results/downstream_cv/sign_stability/sign_stability_summary.csv",
                "aggregation_method": "fold_mean_plus_sd_ddof1",
                "status": "canonical_fold_result",
            }
        )
    for r in dens.itertuples(index=False):
        extra_claims.append(
            {
                "claim_id": f"pair_density|{r.scope}|{r.bin}|{r.metric_name}",
                "dataset": "DS2",
                "model_id": numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO"),
                "experiment_id": "pair_density",
                "scope": r.scope,
                "metric_name": f"{r.bin}:{r.metric_name}",
                "value_display": (
                    f"{float(r.fold_mean):.3f} ± {float(r.fold_sd):.3f}"
                    if pd.notna(r.fold_sd)
                    else f"{float(r.fold_mean):.3f}"
                )
                + (f" (n_valid={int(r.n_valid_folds)})" if int(r.n_valid_folds) != 5 else ""),
                "fold_mean": r.fold_mean,
                "fold_sd": r.fold_sd,
                "n_valid_folds": r.n_valid_folds,
                "artifact_path": "results/downstream_cv/pair_density/pair_density_summary.csv",
                "aggregation_method": "fold_mean_plus_sd_ddof1",
                "status": "canonical_fold_result" if int(r.n_valid_folds) == 5 else "partial_result",
            }
        )
    if len(corr):
        if {"pearson_mean", "spearman_mean"}.issubset(corr.columns):
            for r in corr.itertuples(index=False):
                extra_claims.append(
                    {
                        "claim_id": f"ad_distance_error_corr|{r.scope}|{r.score_name}|pearson_abs_residual",
                        "dataset": "DS2",
                        "model_id": numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO"),
                        "experiment_id": "ad_distance_error_corr",
                        "scope": r.scope,
                        "metric_name": f"{r.score_name}:pearson_abs_residual",
                        "value_display": r.pearson_abs_residual,
                        "fold_mean": r.pearson_mean,
                        "fold_sd": r.pearson_sd,
                        "n_valid_folds": r.n_valid_folds,
                        "artifact_path": "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv",
                        "aggregation_method": "fold_mean_plus_sd_ddof1",
                        "status": "canonical_fold_result",
                    }
                )
                extra_claims.append(
                    {
                        "claim_id": f"ad_distance_error_corr|{r.scope}|{r.score_name}|spearman_abs_residual",
                        "dataset": "DS2",
                        "model_id": numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO"),
                        "experiment_id": "ad_distance_error_corr",
                        "scope": r.scope,
                        "metric_name": f"{r.score_name}:spearman_abs_residual",
                        "value_display": r.spearman_abs_residual,
                        "fold_mean": r.spearman_mean,
                        "fold_sd": r.spearman_sd,
                        "n_valid_folds": r.n_valid_folds,
                        "artifact_path": "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv",
                        "aggregation_method": "fold_mean_plus_sd_ddof1",
                        "status": "canonical_fold_result",
                    }
                )
        elif {"metric_name", "fold_mean"}.issubset(corr.columns):
            for r in corr.itertuples(index=False):
                extra_claims.append(
                    {
                        "claim_id": f"ad_distance_error_corr|{r.scope}|{r.score_name}|{r.metric_name}",
                        "dataset": "DS2",
                        "model_id": numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO"),
                        "experiment_id": "ad_distance_error_corr",
                        "scope": r.scope,
                        "metric_name": f"{r.score_name}:{r.metric_name}",
                        "value_display": (
                            f"{float(r.fold_mean):.3f} ± {float(r.fold_sd):.3f}"
                            if pd.notna(r.fold_sd)
                            else f"{float(r.fold_mean):.3f}"
                        ),
                        "fold_mean": r.fold_mean,
                        "fold_sd": r.fold_sd,
                        "n_valid_folds": r.n_valid_folds,
                        "artifact_path": "results/downstream_cv/ad_distance_error_corr/ad_distance_error_corr_summary.csv",
                        "aggregation_method": "fold_mean_plus_sd_ddof1",
                        "status": "canonical_fold_result",
                    }
                )
    claims_path = MAN / "claim_registry.csv"
    base = pd.read_csv(claims_path)
    extra_df = pd.DataFrame(extra_claims)
    merged = pd.concat([base, extra_df], ignore_index=True).drop_duplicates(subset=["claim_id"])
    token_mask = pd.Series(False, index=merged.index)
    for col in merged.columns:
        if merged[col].dtype == object:
            token_mask = token_mask | merged[col].astype(str).str.contains("spearman", case=False, na=False)
    allow_corr = merged["experiment_id"].astype(str).eq("ad_distance_error_corr") if "experiment_id" in merged.columns else False
    merged = merged.loc[~token_mask | allow_corr].copy()
    merged.to_csv(claims_path, index=False)
    merged.to_csv(MSRC / "manuscript_claim_registry.csv", index=False)

    _out = _prose_out_dir()
    print("wrote", _out / "main_manuscript.md")
    print("wrote", _out / "supplementary_information.md")
    if _out != MAN:
        print(
            "note: curated prose left untouched; set "
            f"{PROSE_OVERWRITE_ENV}=1 to overwrite manuscripts/final/*.md",
            flush=True,
        )
    print("claims", len(merged))


if __name__ == "__main__":
    main()
