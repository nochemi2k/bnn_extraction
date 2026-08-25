"""Main-article body: CMI protocol, DS1/DS2 UQ, AD comparison, sigma-only pair acquisition.

SUPERSEDED FOR PROSE. This module builds the generated draft consumed by
`write_manuscript.py`, which now writes to manuscripts/final/_generated/. The submitted
main text is hand-maintained in manuscripts/final/main_manuscript_iz.md; edits belong
there, not here. Known divergences of this module from the submitted text: it emits five
main figures and the historical figure ids, it lacks the reference list and the
Associated Content section, and it predates the Methods corrections to the ACE
definition, the Pareto tie-breakers, the retention support rule, and the provenance of
the RDKit feature-median Mahalanobis score.
"""
from __future__ import annotations

import pandas as pd

from uq_remake.downstream.protocol import PAIR_ACQ_EVENT_MIN_EXCLUSIVE, pair_acq_event_elig_text, score_display_name
from uq_remake.manuscript.markdown import fmt_msd as _fmt_msd, md_table, note_block, pair_acq_compact_md


def _row(df: pd.DataFrame, name: str):
    hit = df[df.display_name == name]
    if hit.empty:
        raise KeyError(name)
    if "regime" in hit.columns:
        direct = hit[hit["regime"].astype(str).str.lower().eq("direct")]
        if len(direct):
            return direct.iloc[0]
    return hit.iloc[0]


def _cell(df: pd.DataFrame, name: str, col: str) -> str:
    if df is None or getattr(df, "empty", True) or "display_name" not in df.columns or col not in df.columns:
        return "NA"
    hit = df[df.display_name == name]
    if hit.empty:
        return "NA"
    return str(hit.iloc[0][col])


ROLE_DISPLAY = {
    "selected operational accuracy-calibration compromise": "selected operational accuracy–interval-usability compromise",
    "calibration-usable candidate": "interval-usable candidate",
    "screened out by operational calibration-usability screen": "screened out by operational interval-usability screen",
}
STATUS_ROLE = {
    "selected": "selected operational accuracy–interval-usability compromise",
    "candidate": "interval-usable candidate",
    "dominated_screen_passer": "screen-pass, Pareto-dominated",
    "screened_out": "screened out by operational interval-usability screen",
    "scientific_comparator_partial_uq_not_operational_candidate": "scientific comparator; partial native UQ",
    "point_baseline_not_operationally_eligible": "point baseline; no predictive UQ",
}


def build_main_markdown(ctx: dict) -> list[str]:
    fmt = ctx["fmt"]
    ds1 = ctx["ds1"]
    ds2 = ctx["ds2"]
    sc = ctx["sc"]
    numbers = ctx["numbers"]
    ds1_test = ctx["ds1_test"]
    ds2_test = ctx["ds2_test"]
    wh = ctx["wh"]
    ranked = ctx["ranked"]
    chem_note = ctx["chem_note"]
    invalid_total = ctx["invalid_total"]
    ad50 = ctx["ad50"]
    ad50_val = ctx["ad50_val"]
    sign_val = ctx["sign_val"]

    def sign_pct(scope, cat, metric="fraction_rows"):
        txt = sign_val(scope, cat, metric)
        if not txt or txt == "NA" or "±" not in txt:
            return txt
        mean_s, sd_s = txt.split("±", 1)
        return f"{100.0 * float(mean_s.strip()):.1f}% ± {100.0 * float(sd_s.strip()):.1f}%"
    ds1p = ctx.get("ds1p", pd.DataFrame())
    ds1u = ctx.get("ds1u", pd.DataFrame())
    cmi_test = ctx.get("cmi_test", {})
    cmi_ext = ctx.get("cmi_ext", {})
    pub = ctx.get("pub", {})
    overlap = ctx.get("cmi_overlap") or {}
    pair_acq_cmp = ctx.get("pair_acq_cmp", pd.DataFrame())
    pair_acq_hit = ctx.get("pair_acq_hit", pd.DataFrame())
    pair_acq_sum = ctx.get("pair_acq_sum", pd.DataFrame())
    pair_acq_regret = ctx.get("pair_acq_regret", pd.DataFrame())
    pair_acq_elig = ctx.get("pair_acq_elig", pd.DataFrame())
    wh_id = numbers.get("workhorse_model_id", "DS2_RDKIT_MLPENS5_HETERO")
    wh_name = numbers.get("workhorse_display", "Heteroscedastic MLP ensemble")
    others = [r for r in ranked if str(r.get("display")) != wh_name]
    others_by_rmse = sorted(others, key=lambda r: float(r.get("fold_mean")) if r.get("fold_mean") is not None else float("inf"))
    ad_aurc = ctx.get("ad_aurc", pd.DataFrame())
    corr = ctx.get("corr", pd.DataFrame())
    pair_acq_rho = ctx.get("pair_acq_rho", pd.DataFrame())
    mag_adj = ctx.get("mag_adj", pd.DataFrame())
    dens_disp = ctx.get("dens_disp")
    pair_event_sum = ctx.get("pair_event_summary", pd.DataFrame())
    adhoc_val = ctx.get("adhoc_val")
    allp = ctx.get("allp", pd.DataFrame())

    def allp_cell(name, col="mop_rmse"):
        if allp is None or getattr(allp, "empty", True) or "display_name" not in allp.columns or col not in allp.columns:
            return "NA"
        hit = allp[allp.display_name.astype(str) == name]
        if hit.empty:
            return "NA"
        return str(hit.iloc[0][col])

    def corr_spearman(score, scope="all_pairs"):
        if corr is None or getattr(corr, "empty", True):
            return "NA"
        hit = corr[(corr.scope.astype(str) == scope) & (corr.score_name.astype(str) == score)]
        if "model_id" in corr.columns:
            scoped = hit[hit.model_id.astype(str) == wh_id]
            hit = scoped if len(scoped) else hit
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        if "spearman_abs_residual" in hit.columns and pd.notna(r.spearman_abs_residual):
            return str(r.spearman_abs_residual)
        mean = r.spearman_mean if "spearman_mean" in hit.columns else None
        sd = r.spearman_sd if "spearman_sd" in hit.columns else None
        n = int(r.n_valid_folds) if "n_valid_folds" in hit.columns and pd.notna(r.n_valid_folds) else 5
        return _fmt_msd(mean, sd, n)

    def rho_val(metric="rho_sigma_abs_error"):
        if pair_acq_rho is None or getattr(pair_acq_rho, "empty", True):
            return "NA"
        hit = pair_acq_rho[pair_acq_rho.metric_name.astype(str) == metric]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        return _fmt_msd(r.fold_mean, r.fold_sd, int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5)

    def mag_val(metric="rho_partial_sigma_error_given_abs_yhat"):
        if mag_adj is None or getattr(mag_adj, "empty", True):
            return "NA"
        hit = mag_adj[mag_adj.metric_name.astype(str) == metric]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        return _fmt_msd(r.fold_mean, r.fold_sd, int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5)

    def mag_substantial() -> bool:
        if mag_adj is None or getattr(mag_adj, "empty", True):
            return False
        hit = mag_adj[mag_adj.metric_name.astype(str) == "rho_partial_sigma_error_given_abs_yhat"]
        if hit.empty:
            return False
        mean = float(hit.iloc[0].fold_mean)
        sd = float(hit.iloc[0].fold_sd) if pd.notna(hit.iloc[0].fold_sd) else float("nan")
        if pd.isna(mean) or pd.isna(sd):
            return False
        return abs(mean) >= 0.20 and (mean - sd) > 0

    def auc_val(scope, score, metric="retention_curve_auc_mop_rmse"):
        if ad_aurc is None or getattr(ad_aurc, "empty", True):
            return "NA"
        hit = ad_aurc[
            (ad_aurc.dataset.astype(str) == "DS2")
            & (ad_aurc.model_id.astype(str) == wh_id)
            & (ad_aurc.scope.astype(str) == scope)
            & (ad_aurc.score_name.astype(str) == score)
            & (ad_aurc.metric_name.astype(str) == metric)
        ]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        return _fmt_msd(r.fold_mean, r.fold_sd, int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5)

    def retention_table(scope, full_label):
        if not len(ad50):
            return ""
        d2 = ad50.copy()
        if "primary_comparison" in d2.columns:
            d2 = d2[d2.primary_comparison.astype(bool)]
        d2 = d2[
            (d2.dataset == "DS2")
            & (d2.model_id.astype(str) == wh_id)
            & (d2.scope == scope)
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
                    "full MoP RMSE": full_label,
                    "pairs contributing to retained MoP at 50%": _fmt_msd(pair_mean, pair_sd, n, digits=1),
                    "relative RMSE change": "NA" if pd.isna(m.relative_change) else f"{100.0 * float(m.relative_change):+.1f}%",
                    "retention-curve AUC": auc_val(scope, m.score_name),
                }
            )
        if not rows2:
            return ""
        return md_table(pd.DataFrame(rows2), list(rows2[0].keys()))

    def elig_val(cohort, metric="n_eligible_pairs"):
        if pair_acq_elig is None or getattr(pair_acq_elig, "empty", True):
            return "NA"
        hit = pair_acq_elig[
            (pair_acq_elig.cohort.astype(str) == cohort)
            & (pair_acq_elig.metric_name.astype(str) == metric)
        ]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        return _fmt_msd(r.fold_mean, r.fold_sd, int(r.n_valid_folds) if pd.notna(r.n_valid_folds) else 5, digits=1)

    def acq_metric(df, task, orientation, strategy, k, prefix):
        if df is None or getattr(df, "empty", True):
            return "NA"
        hit = df[
            (df.task.astype(str) == task)
            & (df.orientation.astype(str) == orientation)
            & (df.strategy.astype(str) == strategy)
        ]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        mean_c = f"{prefix}_at_{int(k)}"
        sd_c = f"{prefix}_at_{int(k)}_sd"
        return _fmt_msd(r[mean_c] if mean_c in hit.columns else None, r[sd_c] if sd_c in hit.columns else None, 5)

    def acq_hit(task, orientation, strategy, k):
        return acq_metric(pair_acq_hit, task, orientation, strategy, k, "hit")

    def acq_precision(task, orientation, strategy, k):
        if pair_acq_hit is not None and not getattr(pair_acq_hit, "empty", True) and "event_precision_at_1" in pair_acq_hit.columns:
            return acq_metric(pair_acq_hit, task, orientation, strategy, k, "event_precision")
        return acq_metric(pair_acq_cmp, task, orientation, strategy, k, "hit_rate")

    def acq_n_at(task, orientation, strategy, k):
        src = pair_acq_hit if pair_acq_hit is not None and not getattr(pair_acq_hit, "empty", True) else pair_acq_cmp
        if src is None or getattr(src, "empty", True):
            return "NA"
        hit = src[
            (src.task.astype(str) == task)
            & (src.orientation.astype(str) == orientation)
            & (src.strategy.astype(str) == strategy)
        ]
        if hit.empty:
            return "NA"
        r = hit.iloc[0]
        mean_c = f"n_tasks_defined_at_{int(k)}"
        sd_c = f"n_tasks_defined_at_{int(k)}_sd"
        if mean_c in r.index and pd.notna(r[mean_c]):
            return _fmt_msd(r[mean_c], r[sd_c] if sd_c in r.index else None, 5, digits=1)
        return _fmt_msd(r.n_tasks_fold_mean, r.n_tasks_fold_sd, 5, digits=1)

    def acq_sum_metric(metric, task, orientation, strategy, k=None, digits=3):
        sources = [pair_acq_sum, pair_acq_regret]
        for src in sources:
            if src is None or getattr(src, "empty", True):
                continue
            hit = src[
                (src.task.astype(str) == task)
                & (src.orientation.astype(str) == orientation)
                & (src.strategy.astype(str) == strategy)
            ]
            if hit.empty:
                continue
            if "metric_name" in hit.columns:
                part = hit[hit.metric_name.astype(str) == metric]
                if k is not None and "k" in part.columns:
                    part = part[pd.to_numeric(part.k, errors="coerce") == int(k)]
                if part.empty:
                    continue
                r = part.iloc[0]
                n = int(r.n_valid_folds) if "n_valid_folds" in part.columns and pd.notna(r.n_valid_folds) else 5
                return _fmt_msd(r.fold_mean, r.fold_sd, n, digits=digits)
            mean_c = f"{metric}_at_{int(k)}" if k is not None else metric
            sd_c = f"{mean_c}_sd"
            if mean_c not in hit.columns:
                continue
            r = hit.iloc[0]
            n = int(r.n_valid_folds) if "n_valid_folds" in hit.columns and pd.notna(r.n_valid_folds) else 5
            return _fmt_msd(r[mean_c], r[sd_c] if sd_c in hit.columns else None, n, digits=digits)
        return "NA"

    def acq_regret(task, orientation, strategy, k):
        return acq_sum_metric("regret", task, orientation, strategy, k)

    def acq_table_compact():
        return pair_acq_compact_md(pair_acq_hit, pair_acq_regret)

    wh_disp = wh.copy()
    if "selection_status" in wh_disp.columns:
        mapped = []
        for status, role in zip(wh_disp["selection_status"], wh_disp["operational_role"]):
            mapped.append(STATUS_ROLE.get(str(status), ROLE_DISPLAY.get(str(role), str(role))))
        wh_disp["operational_role"] = mapped
    else:
        wh_disp["operational_role"] = wh_disp["operational_role"].map(lambda x: ROLE_DISPLAY.get(str(x), x))
    if "calibration_usable" in wh_disp.columns:
        wh_disp["calibration_usable"] = wh_disp["calibration_usable"].map(
            lambda x: "yes" if str(x).strip().lower() in {"true", "1"} else ("no" if str(x).strip().lower() in {"false", "0"} else x)
        )

    n_shared = overlap.get("n_ligands_shared_train_test", "NA")
    n_test_lig = overlap.get("n_test_ligands", "NA")
    frac_lig = overlap.get("frac_test_ligands_seen_in_train")
    frac_row = overlap.get("frac_test_rows_with_train_ligand")
    frac_lig_s = f"{100.0 * float(frac_lig):.1f}%" if frac_lig is not None else "NA"
    frac_row_s = f"{100.0 * float(frac_row):.1f}%" if frac_row is not None else "NA"
    n_test_rows_cmi = overlap.get("n_test_rows", 4338)
    n_ext_lig = overlap.get("n_ext_ligands", 40)

    cmi_test_mae = f"{float(cmi_test['mae']):.3f}" if cmi_test else "0.802"
    cmi_test_r2 = f"{float(cmi_test['r2']):.3f}" if cmi_test else "0.918"
    cmi_ext_mae = f"{float(cmi_ext['mae']):.3f}" if cmi_ext else "1.278"
    cmi_ext_r2 = f"{float(cmi_ext['r2']):.3f}" if cmi_ext else "0.894"
    pub_test_mae = str(pub.get("cmi_published_m4_test_mae", "0.599"))
    pub_test_r2 = str(pub.get("cmi_published_m4_test_r2", "0.934"))
    pub_ext_mae = str(pub.get("cmi_published_m4_ext_mae", "0.834"))
    pub_ext_r2 = str(pub.get("cmi_published_m4_ext_r2", "0.942"))
    cmi_repro = ctx.get("cmi_repro") or {}
    wt = ((cmi_repro.get("released_weights") or {}).get("evaluations") or {})
    wt_test = wt.get("cmi_test") or {}
    wt_ext = wt.get("cmi_ext") or {}
    wt_test_ens = wt_test.get("ensemble") or {}
    wt_ext_ens = wt_ext.get("ensemble") or {}
    arch_rt = cmi_repro.get("architecture_retrain") or {}
    arch_ev = (arch_rt.get("evaluations") or {}) if str(arch_rt.get("status", "")).lower() == "complete" else {}
    arch_test = (arch_ev.get("cmi_test") or {}).get("ensemble") or {}
    arch_ext = (arch_ev.get("cmi_ext") or {}).get("ensemble") or {}
    arch_test_fold = arch_ev.get("cmi_test") or {}
    cmi_arch = ((cmi_repro.get("released_weights") or {}).get("architecture") or {})

    chem_ds1_rmse = _cell(ds1p, "Chemprop point", "rmse")
    chem_ds1_mae = ctx.get("chem_ds1_mae") or numbers.get("fmt", {}).get("ds1_chemprop_mae", "NA")
    chem_ds1_r2 = _cell(ds1p, "Chemprop point", "r2")
    mlp_ds1_rmse = _cell(ds1p, "RDKit MLP", "rmse")
    mlp_ds1_r2 = _cell(ds1p, "RDKit MLP", "r2")
    mcd_ens_ds1_rmse = _cell(ds1u, "MC-dropout MLP ensemble", "rmse")
    mcd_ens_ds1_r2 = _cell(ds1u, "MC-dropout MLP ensemble", "r2")

    compact_status = {"selected", "candidate", "dominated_screen_passer"}
    if "selection_status" in wh.columns:
        mask = wh["selection_status"].astype(str).isin(compact_status)
        if "regime" in wh.columns:
            mask = mask & wh["regime"].astype(str).str.lower().eq("direct")
        wh_compact = wh_disp.loc[mask].copy()
    else:
        wh_compact = wh_disp.copy()

    main: list[str] = []
    a = main.append

    a("# Pair-level selectivity, prediction risk, and practical ligand prioritization")
    a("")
    a("## Abstract")
    a("")
    abstract = (
        "We study ligand prioritization for pair-specific metal selectivity using task-specific modeling representations derived from a heterogeneous literature collection. The source collection, literature M4, is distinguished from DS1, an absolute-binding representation, and DS2, a relative-selectivity representation whose target is $\\Delta\\log K$. Because metal-pair support is uneven, the primary DS2 evaluation uses ligand-disjoint validation and all-pair mean-of-pairs metrics rather than relying on row-level averages. A fixed heteroscedastic MLP ensemble is used as an operational workhorse for downstream analyses. Its direct uncertainty provides stronger retained-subset error ordering than the tested molecular-distance scores. Uncertainty and predicted selectivity magnitude are positively associated with held-out error, while acquisition performance depends on the objective: threshold-margin ranking favors high-selectivity event recovery, whereas incorporating uncertainty into the ranking substantially improves near-zero event precision over random selection, with only small differences among the uncertainty-aware rules at larger shortlist sizes. Retrospective regret@$k$ further shows that recovering an event is not the same as capturing the best held-out ligand in a pair-task. Restricted rare-earth-element pair analyses are reported only as supplementary formulation diagnostics. The results support task-aligned evaluation and empirical uncertainty ranking, but do not establish a universal applicability-domain boundary or prospective experimental performance."
    )
    a(abstract)
    a("")
    a("## 1. Introduction")
    a("")
    a("A useful ligand-prioritization model must answer two different questions: which metal pair a ligand is likely to favor, and whether that prediction is reliable enough to justify an experiment. Predicting absolute binding constants is relevant to the first question only indirectly. The experimental decision is usually pair-specific: given metals $X$ and $Y$, which ligands should be tested because they may produce a desired relative preference, and which predictions should be deferred because their error risk is high?")
    a("")
    a("$$")
    a("\\Delta\\log K(X,Y,L)=\\log K(X,L)-\\log K(Y,L).")
    a("$$")
    a("")
    a("We use $\\Delta\\log K$ as a relative binding-preference target. It is not a process-level separation factor.")
    a("")
    a("The literature source for this study is a heterogeneous compilation of reported binding measurements. That compilation is not itself the modeling dataset. Absolute-binding examples and pairwise-selectivity examples are constructed from the same source under different inclusion rules and with different targets, so they are not interchangeable subsets of rows. Because metal-pair support is uneven, a row-level metric gives more influence to pairs with many observations and can describe the most heavily sampled pairs rather than the quality of pair-specific decisions. We therefore evaluate selectivity primarily by giving each eligible metal pair one contribution to the summary.")
    a("")
    a("Even when a point model provides a useful average prediction, it does not indicate which individual pair predictions are safe to use. Predictive uncertainty quantification (UQ) is used here to order predictions by expected error risk, not to certify the sign of $\\Delta\\log K$. Direct modeling of $\\Delta\\log K$ attaches uncertainty to the quantity used by the experimental decision. Molecular distance is a tempting alternative risk score, but fingerprint, descriptor-space, and learned-representation distances must be tested against held-out residuals rather than accepted as applicability-domain evidence.")
    a("")
    a("Official-split accuracy can also mislead. The Critical Materials Institute (CMI) assignment of the literature M4 collection (Karunaratne et al., https://doi.org/10.1021/acs.jcim.5c01546) contains substantial ligand reuse: "
      + frac_lig_s
      + " of official test ligands overlap with training and "
      + frac_row_s
      + " of official test rows involve a ligand seen in training. The protocol used here is five-fold ligand-disjoint grouped $K$-fold cross-validation (GroupKFold).")
    a("")
    a("The scientific contribution is therefore task alignment rather than the discovery of a universally optimal architecture. The study defines the modeling datasets clearly from literature M4; aligns evaluation with pair-specific decisions through all-pair mean-of-pairs (MoP) metrics; tests whether direct DS2 uncertainty orders held-out prediction risk; and tests whether different acquisition objectives require different ranking rules.")
    a("")
    a("## 2. Methods")
    a("")
    a("### 2.1 Literature M4 and task-specific modeling datasets")
    a("")
    a("The starting collection, referred to here as literature M4, is a heterogeneous compilation of reported binding measurements from the literature. M4 is the source collection rather than the final modeling dataset. Before modeling, the records were normalized to the fields required by the study, including ligand identity, metal identity, the reported binding quantity, and the available experimental-condition metadata such as solvent or medium.")
    a("")
    a("Normalization follows the implemented CMI-table pipeline. Ligand SMILES are taken as the largest organic fragment by heavy-atom count (otherwise the largest non-metal fragment). Condition fields include temperature, ionic strength, electrolyte identity decoded from solvent molecular weight and density, ligand charge, complex charge, and metal oxidation state. Records are quarantined when ligand parsing fails, the ligand or metal identity is empty, the parsed metal disagrees with the CMI atomic number, solvent molecular weight or density is missing, temperature or ionic strength is missing, or the electrolyte is unknown. The "
      + str(ds1.get("n_external_records", 398))
      + "-record, "
      + str(ds1.get("n_external_ligands", n_ext_lig))
      + "-ligand CMI external file is retained only for the historical protocol comparison in Table 1. It is excluded from DS1 and DS2 modeling tables, is not used to construct project folds, and does not contribute to any all-pair MoP or downstream ranking metric.")
    a("")
    a("DS1 and DS2 are task-specific modeling representations derived from the normalized M4 records. DS1 represents the absolute-binding prediction task: its examples retain the absolute binding target and the inclusion rules used for that task. Under the implemented protonation-inclusion policy, DS1 keeps protonation records in the modeling table. Each normalized literature measurement remains a separate DS1 training example with its reported $\\log K$ value; repeated measurements sharing an experiment key are not collapsed before DS1 training. Consequently, DS1 retains within-condition disagreement present in the literature collection rather than replacing it with a single consensus label. Held-out DS1 accuracy is restricted to metal-complex records. DS2 represents the relative-selectivity task: its examples are formed only when the measurements needed to compare two metals satisfy the study’s pair-construction and matching rules, and its target is $\\Delta\\log K$.")
    a("")
    a("Thus, an M4 record is an individual reported binding measurement, a DS1 example is an absolute-binding modeling example derived from the normalized records, and a DS2 example is a ligand-associated comparison between two metals. DS1 and DS2 are therefore not interchangeable subsets of rows; they encode different prediction targets and different requirements for forming a valid example.")
    a("")
    a("DS2 construction uses only metal-complex records (protonation / H$^+$ rows never enter DS2). Before pairing, metal-complex records are grouped by ligand identity, temperature, ionic strength, electrolyte identity, metal element, and metal oxidation state. Within each such species-level group, repeated $\\log K$ values are replaced by their median. The resulting species-level records are then paired when two metal species share ligand identity, temperature, ionic strength, and electrolyte identity. The $\\Delta\\log K$ target is then calculated from the two aggregated species-level median values. Thus, DS2 does not pair every raw repeated measurement with every measurement for the other metal; it first forms one species-level value for each matched condition and then calculates the ordered difference.")
    a("")
    a("The aggregation key includes metal identity and oxidation state, whereas the pairing key requires the two resulting species records to share the non-metal experimental fields. This distinction prevents measurements for different metal species from being collapsed into one absolute-binding value before the pair is formed.")
    a("")
    a("Each valid comparison is represented as an ordered pair. Reversing the metal order reverses the target sign:")
    a("")
    a("$$")
    a("\\Delta\\log K(Y,X,L)=-\\Delta\\log K(X,Y,L).")
    a("$$")
    a("")
    a("Thus, canonical and reversed high-selectivity directions are separate ranking tasks in the acquisition analysis, even though they arise from the same underlying metal-pair relationship. Electrolyte identity is used for matching and grouping but is excluded from the tabular predictor.")
    a("")
    a("A species-level pair preserves the metal element and oxidation state on both sides of the comparison; it is therefore more specific than an unordered element pair. The "
      + f"{ds2['n_metal_species_pairs']:,}"
      + " species-level pairs and "
      + f"{ds2['n_metal_element_pairs']:,}"
      + " element pairs reported for DS2 are different counts and should not be treated as interchangeable. For example, Fe:2.0--Co:2.0 and Fe:3.0--Co:2.0 are distinct species-level pairs even though they correspond to the same unordered element pair Fe–Co.")
    a("")
    a("The data flow is therefore:")
    a("")
    a("$$")
    a("\\text{literature M4}")
    a("\\rightarrow")
    a("\\text{normalized records}")
    a("\\rightarrow")
    a("\\{\\text{DS1 absolute-binding examples},\\text{DS2 pair-selectivity examples}\\}.")
    a("$$")
    a("")
    a("DS2 is not the full M4 collection and should not be interpreted as a direct synonym for the literature dataset. It is a selectivity-focused representation constructed from M4-derived measurements.")
    a("")
    a("The distinction matters for interpreting the results. Performance on DS1 concerns recovery of an absolute binding quantity from heterogeneous literature observations. Performance on DS2 concerns relative metal-pair selectivity. The two tasks inherit information from the same literature source but have different targets, example-construction rules, and sources of uncertainty.")
    a("")
    a("The compiled literature records also contain within-condition disagreement: nominally matching records can report substantially different $\\log K$ values even when ligand, metal, solvent, and medium match at the level of the available metadata. We treat within-condition disagreement as descriptive internal aleatoric uncertainty. Because replicate spread is not identifiable under the ligand-disjoint folds, we do not estimate a separate laboratory-noise variance. This noise component limits DS1 absolute-binding accuracy and may propagate into DS2 when two absolute measurements are differenced; some error may cancel, while some may amplify. Direct DS2 training changes the target and its representation, but it does not guarantee removal of the underlying measurement variability.")
    a("")
    a(f"DS1 contains **{ds1['n_records']:,}** records and **{ds1['n_unique_ligands']:,}** unique ligands. DS2 contains **{ds2['n_pairs_rows']:,}** ligand-level rows associated with **{ds2['n_metal_species_pairs']:,}** species-level pairs. The primary DS2 benchmark is the full-data all-pair scope. Feature dimensions, protonation policy, and fold sizes are in Supplementary Sections S1–S2.")
    a("")
    a("### 2.2 Validation regime")
    a("")
    a("The official assignment and the ligand-disjoint evaluation answer different questions. The official assignment preserves the original benchmark structure and may permit ligand overlap between training and evaluation records. The ligand-disjoint evaluation assigns all records associated with a ligand to one side of each split, so that held-out predictions concern ligands unavailable during training. We use the ligand-disjoint evaluation for the main generalization and downstream risk-ordering analyses.")
    a("")
    a("The ligand-disjoint analysis uses five outer folds. Predictions are generated for held-out records, and pair-level metrics are calculated within each fold before being summarized across folds. This procedure prevents heavily represented pairs from determining the primary headline aggregation, because each eligible pair contributes one value to MoP. The split itself does not make pair support uniform. Inner validation is a ligand-disjoint subset of the outer-training ligands. Fold-local preprocessing is fit on outer-train features only. Headline values are fold means ± SD. Concatenated held-out rows, including out-of-fold pooled diagnostics, are not used as headline estimates.")
    a("")
    a(
        "On the official CMI assignment, "
        + str(n_shared)
        + " of "
        + str(n_test_lig)
        + " official test ligands ("
        + frac_lig_s
        + ") already appear in train, and "
        + frac_row_s
        + " of official test rows use a ligand seen in train. Under ligand-disjoint DS1 evaluation, Chemprop point had metal-complex MAE "
        + chem_ds1_mae
        + " and $R^2$ "
        + chem_ds1_r2
        + ". The official-split comparison is Table 1. Shared-architecture retrains train a Chemprop 1.6.1 D-MPNN from random initialization and do not load CMI weights. Released CMI `best_model` checkpoints are scored on the official files, and a from-scratch retrain uses the architecture recovered from those checkpoints (hidden size 2400, depth 6; S1.5.2)."
    )
    a("")
    a("**Table 1. Official CMI assignment versus ligand-disjoint DS1 Chemprop.**")
    a(
        note_block(
            "Published test metrics are five-fold CV averages. The released-weight test row reports the mean of five fold-wise metrics on the same open official test file; the released-weight external row is the five-fold ensemble, matching the published external protocol. Shared-architecture retrains use Chemprop 1.6.1 with hidden size 300 and depth 3 and do not load CMI `.pt` files. CMI architecture (hidden 2400, depth 6) is recovered from those checkpoints; the from-scratch architecture-matched retrain is the five-fold ensemble in Table 1 and Table S1.8. MAE is the error metric reported by Karunaratne et al. Official CMI columns use all records; the ligand-disjoint Chemprop row is metal-complex MAE (fold mean ± SD). DS1 headline tables remain RMSE (Table S3.1). Source: `results/stage1/stage1a_cmisplit_results.json`, `results/stage1/cmi_reproduction_results.json`, `results/canonical/cmi_split_ligand_overlap.json`, and `results/canonical/ds1_chemprop_ligand_disjoint_mae.json`."
        )
    )
    table1_rows = [
        {
            "source": "published CMI M4 test",
            "estimand": "5-fold CV average",
            "ligand-disjoint": "no (ligand overlap)",
            "MAE": pub_test_mae,
            "$R^2$": pub_test_r2,
        },
    ]
    if wt_test.get("fold_mean_mae") is not None:
        table1_rows.append(
            {
                "source": "CMI released weights, official test file",
                "estimand": "mean of 5 fold-wise metrics on open test file",
                "ligand-disjoint": f"no ({frac_lig_s} ligands shared)",
                "MAE": f"{float(wt_test['fold_mean_mae']):.3f}",
                "$R^2$": f"{float(wt_test['fold_mean_r2']):.3f}",
            }
        )
    table1_rows.append(
        {
            "source": "this work, shared Chemprop 1.6.1, official test file",
            "estimand": "from-scratch hidden 300 / depth 3, open test file",
            "ligand-disjoint": f"no ({frac_lig_s} ligands shared)",
            "MAE": cmi_test_mae,
            "$R^2$": cmi_test_r2,
        }
    )
    if arch_test:
        table1_rows.append(
            {
                "source": "this work, CMI architecture, official test file",
                "estimand": "from-scratch hidden 2400 / depth 6, 5-fold ensemble",
                "ligand-disjoint": f"no ({frac_lig_s} ligands shared)",
                "MAE": f"{float(arch_test['mae']):.3f}",
                "$R^2$": f"{float(arch_test['r2']):.3f}",
            }
        )
    table1_rows.append(
        {
            "source": "published CMI M4 external",
            "estimand": "ensemble on 40-ligand holdout",
            "ligand-disjoint": "yes (external only)",
            "MAE": pub_ext_mae,
            "$R^2$": pub_ext_r2,
        }
    )
    if wt_ext_ens:
        table1_rows.append(
            {
                "source": "CMI released weights, official external file",
                "estimand": "5-fold ensemble of released .pt files",
                "ligand-disjoint": "yes (external only)",
                "MAE": f"{float(wt_ext_ens['mae']):.3f}",
                "$R^2$": f"{float(wt_ext_ens['r2']):.3f}",
            }
        )
    table1_rows.append(
        {
            "source": "this work, shared Chemprop 1.6.1, official external file",
            "estimand": "same from-scratch hidden 300 / depth 3 model",
            "ligand-disjoint": "yes (external only)",
            "MAE": cmi_ext_mae,
            "$R^2$": cmi_ext_r2,
        }
    )
    if arch_ext:
        table1_rows.append(
            {
                "source": "this work, CMI architecture, official external file",
                "estimand": "from-scratch hidden 2400 / depth 6, 5-fold ensemble",
                "ligand-disjoint": "yes (external only)",
                "MAE": f"{float(arch_ext['mae']):.3f}",
                "$R^2$": f"{float(arch_ext['r2']):.3f}",
            }
        )
    table1_rows.append(
        {
            "source": "this work, DS1 Chemprop point",
            "estimand": "5-fold ligand-disjoint CV",
            "ligand-disjoint": "yes",
            "MAE": chem_ds1_mae,
            "$R^2$": chem_ds1_r2,
        }
    )
    a(
        md_table(
            pd.DataFrame(table1_rows),
            ["source", "estimand", "ligand-disjoint", "MAE", "$R^2$"],
        )
    )
    a("")
    a("**Figure 1. Study design and decision workflow.** Literature M4 is normalized into task-specific modeling representations, including DS1 for absolute binding and DS2 for pairwise selectivity. The primary analysis evaluates DS2 across eligible metal pairs under ligand-disjoint validation, uses mean-of-pairs (MoP) as the pair-level summary, tests uncertainty as an empirical risk-ordering variable, and evaluates objective-specific retrospective ranking. Restricted chemical subsets are not substituted for the all-pair evaluation.")
    a("")
    a("![Figure 1](figures/fig01_study_design.png)")
    a("")
    a("### 2.3 Primary target and mean-of-pairs metrics")
    a("")
    a("The main prediction target is relative selectivity, $\\Delta\\log K$, for a metal pair. Because the practical decision is pair-specific and pair support is uneven, the primary evaluation unit is the metal pair rather than the individual record.")
    a("")
    a("For each eligible metal pair $p$, let $\\mathrm{RMSE}_p$ be the root mean squared error (RMSE) calculated over the held-out ligands for that pair. The mean-of-pairs RMSE is")
    a("")
    a("$$")
    a("\\mathrm{MoP\\text{-}RMSE}")
    a("=")
    a("\\frac{1}{|\\mathcal{P}_{\\mathrm{eligible}}|}")
    a("\\sum_{p\\in\\mathcal{P}_{\\mathrm{eligible}}}")
    a("\\mathrm{RMSE}_p.")
    a("$$")
    a("")
    a("Each eligible pair therefore receives equal weight, regardless of the number of held-out ligands associated with it. By contrast, row-level RMSE weights individual records and consequently gives greater influence to heavily sampled pairs.")
    a("")
    a("Within an outer fold, a pair contributes to MoP only when it has the required held-out-ligand support; in the present analysis, this requires at least three held-out ligands. A model-level all-pair headline is reported only when all five outer folds provide a valid all-pair MoP summary.")
    a("")
    a("All-pair MoP RMSE is the primary metric for the pair-specific decision. Row-level RMSE, row-level $R^2$, MoP $R^2$, prediction interval coverage probability (PICP) at the nominal 90% level, and average coverage error (ACE) are complementary diagnostics and are not treated as interchangeable measures of model quality. MoP $R^2$ is the mean of within-pair $R^2$ values and is auxiliary because it is unstable for pairs with limited target variation. Row-level PICP and ACE are row-weighted interval diagnostics; MoP PICP/ACE are pair-equal interval diagnostics. Retained-subset MoP RMSE is the risk-ordering endpoint. Event precision@$k$ is the fraction of selected ligands that are events. True hit@$k$ is the shortlist discovery probability: whether at least one selected ligand is an event. Retrospective regret@$k$ is the leftover objective gap between the best ligand in a pair-task pool and the best ligand in the selected shortlist; it is not interchangeable with the event metrics.")
    a("")
    a("### 2.4 Models and workhorse selection")
    a("")
    a("Point models provide $\\widehat y$ but no predictive interval. UQ models provide both a prediction and an uncertainty estimate. The heteroscedastic multilayer perceptron (MLP) has the shared tabular backbone $[512,512,256]$ and two output quantities: a predictive mean and an input-dependent log observation variance. The variance output is transformed to a positive variance by exponentiation of a clamped log-variance. The five-member workhorse ensemble uses independently trained members under the same fold-local training protocol; member predictions also include a within-member Monte Carlo dropout contribution from 30 stochastic forward passes with dropout kept active at prediction. The ensemble mean and variance are combined using the law of total variance, as defined in Supplementary Section S2.4. The reported $\\sigma_{\\mathrm{total}}$ is the square root of that total predictive variance. Exact Gaussian process regression (GPR) was computationally infeasible; Exact GPR numbers do not appear in active comparison tables. Dropout is **not** shared across MLP families. Chemprop evidential DS2 has partial native UQ (**"
      + str(invalid_total)
      + " invalid native Normal–Inverse-Gamma (NIG) rows**) and is not an operational candidate. Native Chemprop $\\sigma_{\\mathrm{total}}$ is a separate model-specific output and is not pooled with the selected-model score. Architectural details are in Supplementary Section S2. DS1 results remain a supporting absolute-binding benchmark in Supplementary Section S3.")
    a("")
    a("The downstream analyses require one fixed direct-UQ model, so we selected a representative model using the full-data DS2 all-pair scope. Here, “workhorse” means the fixed representative model carried into the downstream analyses. Eligibility is determined by the study-defined row-level interval-usability screen, $0.85\\le\\mathrm{PICP@90}\\le 0.95$ and $\\mathrm{ACE}\\le 0.10$, with five valid folds and complete native UQ. The MoP interval quantities are reported as pair-level diagnostics and are not the eligibility gate. Among eligible models, the RMSE–PICP–ACE Pareto comparison was used to identify a reasonable operational compromise. The Pareto comparison treats lower all-pair MoP RMSE, lower row-level ACE, and closer-to-nominal row-level PICP@90 as preferable, subject to the operational screen. Several models were close, and the selected heteroscedastic ensemble was not the lowest-RMSE model. We therefore use the term “workhorse” only to denote the fixed model used for downstream analyses, not to claim unique or universal superiority. Row-level RMSE was not used for selection. Selection did not use pair-acquisition event precision@$k$, true hit@$k$, or retrospective regret@$k$. Workhorse selection was exploratory and conditional, not a nested statistical test.")
    a("")
    a("A restricted supplementary diagnostic compares direct DS2 uncertainty with uncertainty propagated from two DS1 absolute-binding predictions. The propagated interval uses the independent-error expression $\\sigma_{\\Delta}=\\sqrt{\\sigma_X^2+\\sigma_Y^2}$ and is evaluated only on the restricted chemical subset described in Supplementary Section S13. This diagnostic is not used to select the workhorse and does not replace the all-pair MoP or retention analyses.")
    a("")
    a("### 2.5 Distance error ordering")
    a("")
    a("We tested whether three practical scores—$\\sigma_{\\mathrm{total}}$, RDKit feature-centroid Mahalanobis distance, and Tanimoto nearest-neighbor distance—could order held-out prediction error. The distance scores were computed from outer-training information only. Retention is performed at the held-out ligand-row level. After retaining the lowest-risk rows at a specified fraction, MoP RMSE is recalculated using only pairs with at least three retained ligands. Therefore, different scores can produce different retained pair populations even at the same nominal retention fraction. This directly tests whether a score can order held-out error. It does not define a universal acceptance threshold and does not establish a universal applicability-domain boundary.")
    a("")
    a("Retention-curve AUC is calculated over retained-subset MoP RMSE as a function of retained fraction; lower values indicate stronger concentration of low-error predictions at low retention fractions.")
    a("")
    a("The additional fingerprint, descriptor, and Chemprop-embedding scores were sensitivity analyses. Because the tested molecular-distance scores were weaker than the workhorse uncertainty score for the primary all-pair retention analysis, they are treated as comparative risk-ordering baselines rather than as operational applicability-domain proxies. Full score formulas and sensitivity inventories are in Supplementary Section S7.")
    a("")
    a("### 2.6 Pair acquisition")
    a("")
    a("A species-level pair is a metal-species relationship in DS2, preserving the element and oxidation state on both sides of the comparison. A pair-task is a species-level pair combined with an orientation and a specified acquisition objective. A high-selectivity pair can therefore contribute separate canonical and reversed pair-tasks, whereas the near-zero objective is orientation-invariant.")
    a("")
    a("High-selectivity and near-zero selectivity are different pair-task objectives. High-selectivity pair-tasks are oriented; near-zero pair-tasks are orientation-invariant. Pair-task eligibility requires more than one observed held-out event for the relevant objective. We evaluate a retrospective held-out ranking simulation. Candidates are ranked before their held-out labels are used for scoring, but observed held-out event support is used to define the evaluable task set; no prospective experiment is performed.")
    a("")
    a("High-selectivity: target-direction $\\Delta\\log K>1$. Near-zero: $|\\Delta\\log K|<1$. Eligibility: $n_{\\mathrm{event}}>"
      + str(int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE))
      + "$. A pair-task contributes to Precision@$k$, Hit@$k$, and retrospective regret@$k$ only when it has at least $k$ held-out rows.")
    a("")
    a("Ligands are ranked using only $\\widehat y$ and $\\sigma_{\\mathrm{total}}$. Distance scores are not ranking inputs.")
    a("")
    a("For an oriented high-selectivity task, define the oriented target and prediction as")
    a("")
    a("$$")
    a("y_i^\\star =")
    a("\\begin{cases}")
    a("y_i, & \\text{canonical direction},\\\\")
    a("-y_i, & \\text{reversed direction},")
    a("\\end{cases}")
    a("\\qquad")
    a("\\widehat y_i^\\star =")
    a("\\begin{cases}")
    a("\\widehat y_i, & \\text{canonical direction},\\\\")
    a("-\\widehat y_i, & \\text{reversed direction}.")
    a("\\end{cases}")
    a("$$")
    a("")
    a("The uncertainty is unchanged by reversing the target orientation, so $\\sigma_{\\mathrm{total},i}^\\star=\\sigma_{\\mathrm{total},i}$.")
    a("")
    a("For high selectivity, the ranking scores are oriented predicted selectivity, oriented lower confidence bounds (LCB), and the standardized threshold-margin score:")
    a("")
    a("$$")
    a("s_{\\mathrm{value},i}=\\widehat y_i^\\star,")
    a("\\qquad")
    a("s_{\\mathrm{LCB},z,i}=\\widehat y_i^\\star-z\\sigma_{\\mathrm{total},i},")
    a("\\qquad")
    a("s_{\\mathrm{margin},i}=")
    a("\\frac{\\widehat y_i^\\star-1}{\\sigma_{\\mathrm{total},i}},")
    a("\\quad z\\in\\{1,2,3\\}.")
    a("$$")
    a("")
    a("Higher values indicate a more favorable candidate for the high-selectivity objective.")
    a("")
    a("For near-zero selection, candidates are ranked from low to high predicted magnitude. The predicted-magnitude score is $-|\\widehat y_i|$, the uncertainty-aware LCB variants are $-(|\\widehat y_i|+z\\sigma_{\\mathrm{total},i})$, and the standardized margin is $(1-|\\widehat y_i|)/\\sigma_{\\mathrm{total},i}$. Higher values indicate a more favorable candidate for the near-zero objective.")
    a("")
    a("The threshold-margin score is that specified standardized margin, not the Gaussian expected-improvement integral. This is a sigma-only pair-acquisition evaluation: distance scores are not ranking inputs.")
    a("")
    a("Event precision@$k$ asks how many of the selected ligands are events. True hit@$k$ is the shortlist discovery indicator: it asks whether the shortlist contains at least one event. They coincide only at $k=1$. Both treat every observed event equally, so a shortlist that captures a weakly qualifying event can score as well as one that captures the strongest available ligand. Event precision@$k$ is")
    a("")
    a("$$")
    a("\\mathrm{Precision@}k")
    a("=")
    a("\\frac{1}{k}\\sum_{j=1}^{k} I_j,")
    a("$$")
    a("")
    a("where $I_j=1$ if the ranked ligand is an event. True hit@$k$ is")
    a("")
    a("$$")
    a("\\mathrm{Hit@}k")
    a("=")
    a("\\mathbf{1}")
    a("\\left[")
    a("\\sum_{j=1}^{k}I_j>0")
    a("\\right].")
    a("$$")
    a("")
    a("Retrospective regret@$k$ is therefore scored on the same within-pair rankings to measure the leftover objective gap, not event presence. Let $\\mathcal{P}$ be the held-out ligand pool of one pair-task, and let $B_k\\subseteq\\mathcal{P}$ be the top $k$ ranked ligands in that pair-task. Ranking is never pooled across pairs within a fold. For high selectivity, $y^{\\star}$ is the target-direction observed selectivity (canonical $y$, reversed $-y$), and")
    a("")
    a("$$")
    a("R_k^{\\mathrm{high}}")
    a("=")
    a("\\max_{i\\in\\mathcal{P}} y^{\\star}_i")
    a("-")
    a("\\max_{i\\in B_k} y^{\\star}_i.")
    a("$$")
    a("")
    a("For near-zero selectivity,")
    a("")
    a("$$")
    a("R_k^{\\mathrm{near}}")
    a("=")
    a("\\min_{i\\in B_k}|y_i|")
    a("-")
    a("\\min_{i\\in\\mathcal{P}}|y_i|.")
    a("$$")
    a("")
    a("Lower values are better. The units are $\\Delta\\log K$. $R_k$ is defined only when $|\\mathcal{P}|\\ge k$, the same support rule used for Precision@$k$ and Hit@$k$. Regret is nonnegative. Zero means that $B_k$ contains a pool-optimal ligand, including tied optima. The pool optimum itself does not depend on the ranking rule. Rankings, tie-breaking, and shortlists are unchanged from the hit@$k$ and precision@$k$ calculations; held-out labels identify the pool optimum for scoring only.")
    a("")
    a("Random ranking uses 20 within-pair permutations (seed 0). Metrics, including retrospective regret@$k$, are aggregated by pair-equal mean within each fold, then fold mean ± SD. The complete $k=1,5,10$ inventories are Tables S10.2 (event precision), S10.3 (true hit), and S10.4 (retrospective regret); Table 4 and Figure 5 report the main-text subset.")
    a("")
    a("## 3. Results")
    a("")
    a("### 3.1 Data representations and validation regime")
    a("")
    a("M4, DS1, and DS2 represent different stages and tasks in the data pipeline. M4 is the heterogeneous literature source, DS1 is the absolute-binding modeling representation, and DS2 is the pairwise selectivity representation. The main downstream analysis uses DS2 under ligand-disjoint validation because the intended decision is pair-specific ligand prioritization.")
    a("")
    a("Ligand-disjoint validation tests unseen-ligand generalization. All-pair MoP is the primary DS2 evaluation because pair support is uneven and the decision is pair-specific. Official-split accuracy is a different estimand and is reported in Table 1 only as a protocol comparison: published CMI M4 metrics, released `best_model` scores, a shared-architecture from-scratch retrain, and a from-scratch retrain with the recovered CMI architecture. Section 4.2 discusses that comparison.")
    a("")
    a("### 3.2 Row-level versus pair-level performance")
    a("")
    a("The relevant question is not whether the model predicts individual binding records accurately on average, but whether its errors are acceptably distributed across the metal pairs used for prioritization. The gap between row-level and MoP performance shows that favorable row-weighted performance does not establish reliable pair-specific behavior.")
    a("")
    a("For the selected heteroscedastic MLP ensemble, row-level $R^2$ is "
      + str(_row(wh, wh_name).all_pair_row_r2)
      + ", whereas all-pair MoP $R^2$ is "
      + fmt["wh_mop_r2"]
      + " and all-pair MoP RMSE is "
      + fmt["wh_mop_rmse"]
      + ".")
    a("")
    a("These values summarize different estimands rather than conflicting calculations: row-level $R^2$ is weighted by individual records, whereas MoP $R^2$ gives each eligible pair equal weight.")
    a("")
    a("We use the row-level/MoP contrast diagnostically. MoP $R^2$ is not treated as a standalone model-selection objective; the primary pair-level quantity is MoP RMSE, while interval metrics provide complementary information about uncertainty behavior. RDKit XGBoost has the lowest all-pair MoP RMSE among the point baselines ("
      + allp_cell("RDKit XGBoost")
      + "), but it provides no predictive uncertainty and therefore cannot be used for the downstream uncertainty-based retention analysis.")
    a("")
    a("### 3.3 Fixed workhorse and all-pair direct UQ")
    a("")
    a("The heteroscedastic MLP ensemble is fixed as the workhorse for the downstream analyses. This is an operational choice, not a claim that it is the universally best architecture. The purpose of fixing the model is to test whether its uncertainty score orders held-out prediction risk and supports pair acquisition without repeatedly changing the predictor. The resulting score is a composite predictive-risk signal; the analysis does not identify which physical or statistical source of uncertainty causes each high-risk prediction.")
    a("")
    a("Eligibility is determined by the study-defined row-level interval-usability screen. The MoP interval quantities are reported as pair-level diagnostics and are not the eligibility gate. The selected model passes the row-level screen with row-level PICP@90 and ACE equal to "
      + fmt["wh_picp"]
      + " and "
      + fmt["wh_ace"]
      + ". It is retained despite the lower MoP-RMSE values of some candidates because the downstream study requires one complete direct-UQ model selected under the prespecified interval-usability and Pareto procedure. This choice fixes the predictor before risk ordering; it is not evidence that the selected model dominates the grid. In particular, the Evidential MLP ensemble ("
      + allp_cell("Evidential MLP ensemble")
      + ") and the MC-dropout MLP ("
      + allp_cell("MC-dropout MLP")
      + ") have lower all-pair MoP RMSE than the selected workhorse ("
      + fmt["wh_mop_rmse"]
      + "). The subsequent experiments evaluate the usefulness of the selected model's uncertainty score; they do not validate the model-selection rule as a nested statistical procedure. The full model grid is in Supplementary Sections S4 and S6.")
    a("")
    a("**Figure 2. All-pair direct-UQ model context and representative-model choice.** The heteroscedastic MLP ensemble is fixed as the workhorse for downstream analyses because it provides an operationally suitable direct-UQ model under the study’s interval-usability convention; it is not presented as the universally optimal architecture. The plotted coordinates are pair-equal MoP diagnostics. Eligibility is determined separately from unrounded five-fold row-level PICP@90 and ACE values, so neither shaded region is the operational eligibility gate.")
    a("")
    a("![Figure 2](figures/fig02_workhorse_selection.png)")
    a("")
    a("**Table 2. Screen-passing direct-UQ models on the all-pair scope, including the selected representative workhorse.**")
    a(
        note_block(
            "Compact context only; not a full leaderboard. Rows are direct DS2 models that passed the operational interval-usability screen (selected, candidate, or screen-pass but Pareto-dominated). Primary accuracy = all-pair MoP RMSE. Displayed PICP@90 and ACE include pair-equal MoP summaries and the row-level PICP@90 and ACE used by the screen. Auxiliary = all-pair MoP $R^2$. The selected model is an operational accuracy–interval-usability compromise, not a uniquely optimal predictor. Full inventory: Supplementary Table S6.1. Source: `manuscripts/final/tables/main_ds2_operational_comparison.csv`."
        )
    )
    if len(wh_compact):
        a(
            md_table(
                wh_compact,
                [
                    "display_name",
                    "all_pair_mop_rmse",
                    "all_pair_picp_90",
                    "all_pair_ace",
                    "all_pair_row_picp_90",
                    "all_pair_row_ace",
                    "all_pair_mop_r2",
                    "operational_role",
                ],
                {
                    "display_name": "model",
                    "all_pair_mop_rmse": "All-pair MoP RMSE",
                    "all_pair_picp_90": "all-pair MoP PICP@90",
                    "all_pair_ace": "all-pair MoP ACE",
                    "all_pair_row_picp_90": "row-level PICP@90",
                    "all_pair_row_ace": "row-level ACE",
                    "all_pair_mop_r2": "all-pair MoP $R^2$ (aux)",
                    "operational_role": "role",
                },
            )
        )
    a("")
    a("### 3.4 All-pair risk ordering for the selected workhorse")
    a("")
    a("We next test the selected workhorse uncertainty score on all eligible DS2 metal pairs. The fixed-retention analysis asks whether a score can select a subset of held-out predictions with lower pair-level error than the unfiltered set or a random-retention reference.")
    a("")
    a("At 50% nominal retention, $\\sigma_{\\mathrm{total}}$ produced a MoP RMSE of "
      + ad50_val(ad50, "DS2", wh_id, "sigma_total", "mop_rmse", scope="all_pairs")
      + ", compared with "
      + ad50_val(ad50, "DS2", wh_id, "mahalanobis_feature_centroid", "mop_rmse", scope="all_pairs")
      + " for the RDKit feature-centroid Mahalanobis score, "
      + ad50_val(ad50, "DS2", wh_id, "tanimoto_nn_distance", "mop_rmse", scope="all_pairs")
      + " for Tanimoto nearest-neighbor distance, and "
      + fmt["wh_mop_rmse"]
      + " for the unfiltered selected-workhorse predictions.")
    a("")
    a("The result is conditional on the retained populations. At the same nominal retention, the ranking rules retain different numbers of MoP-eligible pairs. The support panel therefore accompanies retained-subset RMSE and prevents the comparison from being interpreted as evaluation over a common pair population.")
    a("")
    a("Among the primary all-pair scores evaluated for the selected workhorse, $\\sigma_{\\mathrm{total}}$ provided the strongest descriptive empirical ordering of held-out prediction error. The RDKit feature-centroid Mahalanobis score provided an intermediate reduction, while Tanimoto nearest-neighbor distance remained closer to the random-retention behavior. This supports model-derived uncertainty as the primary operational risk-ordering variable in this validation regime.")
    a("")
    a("**Table 3. Prespecified 50% retention comparison on full-data DS2 all-pair MoP RMSE.**")
    a("")
    a("These are retained-MoP support counts after filtering, not the unfiltered all-pair counts used at full retention.")
    a(
        note_block(
            "Selected workhorse only (`"
            + wh_id
            + "`). Retained-subset RMSE values measure error among predictions selected by each score; they are not estimates of overall model accuracy and do not constitute an acceptance guarantee. Full MoP RMSE is the unfiltered 100% all-pair selected-workhorse MoP RMSE. Relative RMSE change is $\\Delta_{\\mathrm{rel}}=(\\mathrm{RMSE}_{50\\%}-\\mathrm{RMSE}_{100\\%})/\\mathrm{RMSE}_{100\\%}$, computed from the five-fold mean retained RMSE and the five-fold mean unfiltered RMSE. Negative values indicate lower retained-subset error. Retention-curve AUC uses the canonical 11-point 10–100% grid; lower values indicate stronger concentration of low-error predictions at low retention fractions. Random retention is shown in Figure 3 as the no-ordering reference; Table 3 reports the three prespecified operational scores. Source: `results/downstream_cv/ad_retention/retention_50pct.csv` and `retention_auc_summary.csv`."
        )
    )
    tbl3 = retention_table("all_pairs", fmt["wh_mop_rmse"])
    if tbl3:
        a(tbl3)
    a("")
    a("**Figure 3. DS2 selected-workhorse all-pair fixed-retention risk ordering.** Each score is used to retain a nominal fraction of held-out DS2 predictions, after which MoP RMSE is evaluated on the resulting all-pair subset. Panel B counts species-level pairs with at least three retained ligands. It contextualizes MoP support and is not a performance metric. The black dashed curve is the fold-local random-retention baseline. Different scores retain different pair populations at the same nominal retention fraction. The result tests empirical error ordering for the stated all-pair dataset and validation regime.")
    a("")
    a("![Figure 3](figures/fig03_risk_ordering.png)")
    a("")
    a("### 3.5 Why uncertainty orders error")
    a("")
    a("Both held-out error and $\\sigma_{\\mathrm{total}}$ increase with predicted selectivity magnitude. The rank association between $\\sigma_{\\mathrm{total}}$ and absolute error was "
      + rho_val("rho_sigma_abs_error")
      + ", compared with "
      + mag_val("rho_abs_yhat_abs_error")
      + " between $|\\widehat y|$ and absolute error. After controlling for $|\\widehat y|$ using a fold-local partial rank correlation, the association between $\\sigma_{\\mathrm{total}}$ and absolute error remained positive at "
      + mag_val("rho_partial_sigma_error_given_abs_yhat")
      + ". Uncertainty therefore contains empirical risk-ordering information beyond predicted magnitude in this analysis, although this result does not establish conditional independence or causality.")
    a("")
    a("**Figure 4. Predicted magnitude, error, and uncertainty.** The fold-local partial rank association controls for predicted magnitude. The analysis is descriptive and does not establish conditional independence or causality.")
    a("")
    a("![Figure 4](figures/fig04_predicted_magnitude.png)")
    a("")
    a("### 3.6 Objective-specific pair acquisition")
    a("")
    a("Retention and acquisition answer different questions. Retention asks which predictions appear safer to use; acquisition asks which candidates best match a specified experimental objective. A score that is useful for rejecting risky predictions need not be optimal for discovering high-selectivity candidates.")
    a("")
    n_heldout_pairs = "NA"
    n_gt_events = "NA"
    if pair_event_sum is not None and getattr(pair_event_sum, "empty", True) is False:
        r0 = pair_event_sum.iloc[0]
        n_heldout_pairs = f"{int(r0['n_pairs_with_heldout_rows']):,}"
        event_col = f"n_pairs_gt_{int(PAIR_ACQ_EVENT_MIN_EXCLUSIVE)}_high_selectivity_events"
        if event_col in r0.index:
            n_gt_events = f"{int(r0[event_col]):,}"
        elif "n_pairs_gt_10_high_selectivity_events" in r0.index:
            n_gt_events = f"{int(r0['n_pairs_gt_10_high_selectivity_events']):,}"
    a(
        "Using out-of-fold held-out labels only, "
        + n_heldout_pairs
        + " species-level pairs had held-out rows; the descriptive pooled counts come from the registered pair-event-support artifacts. Across the pooled out-of-fold labels, "
        + n_gt_events
        + " species-level pairs had "
        + pair_acq_event_elig_text().replace("held-out ", "high-selectivity ")
        + " when canonical ($y>1$) and reversed ($y<-1$) events were counted together. Source: `results/downstream_cv/pair_event_support/pair_event_support_summary.csv`."
    )
    a("")
    a(
        "The pooled "
        + n_gt_events
        + "-pair count is descriptive. The ranking estimand is fold-local: canonical and reversed directions are separate oriented pair-tasks, and only pair-tasks with "
        + pair_acq_event_elig_text()
        + " in the relevant direction enter the ranking benchmark. After fold-local eligibility filtering, the ranking benchmark contained "
        + elig_val("high_both_orientations", "n_eligible_pair_tasks")
        + " eligible oriented high-selectivity pair-tasks per fold. Ranking metrics use the oriented pair-tasks. Because eligibility requires more than one observed held-out event, the evaluated pair-task set is label-informed. Rankings are nevertheless formed from out-of-fold predictions and uncertainty before held-out labels are used for event and regret scoring. The result is therefore a retrospective ranking evaluation, not prospective experimental validation."
    )
    a("")
    a(
        "For high-selectivity pair tasks, threshold margin gave the strongest short-list performance among the tested rules on the event metrics. At $k=5$, its true-hit rate was "
        + acq_hit("high_selectivity", "both", "expected_improvement", 5)
        + ", compared with "
        + acq_hit("high_selectivity", "both", "random", 5)
        + " for random ranking, and its event precision was "
        + acq_precision("high_selectivity", "both", "expected_improvement", 5)
        + ", compared with "
        + acq_precision("high_selectivity", "both", "random", 5)
        + " for random ranking."
    )
    a("")
    a(
        "For near-zero selection, uncertainty-aware and standardized-margin rules substantially improved event precision over random ranking, while the non-random strategies approached similar true-hit rates at larger shortlist sizes. For example, LCB-2 achieved precision@5 of "
        + acq_precision("low_selectivity", "near_zero", "lcb_2sigma", 5)
        + ", compared with "
        + acq_precision("low_selectivity", "near_zero", "random", 5)
        + " for random ranking."
    )
    a("")
    a(
        "Event precision@$k$ and true hit@$k$ credit any ligand that meets the event threshold. A shortlist can therefore recover an event while missing the strongest held-out ligand in that pair-task. Retrospective regret@$k$ measures that leftover objective gap on the same within-pair shortlists (Figure 5 E--F; Table 4; Table S10.4). Rankings are unchanged from the event metrics; held-out labels identify the pool optimum for scoring only. Lower regret is better, and zero means the shortlist contains a pool-optimal ligand."
    )
    a("")
    a(
        "At $k=5$, threshold-margin high-selectivity regret was "
        + acq_regret("high_selectivity", "both", "expected_improvement", 5)
        + ", compared with "
        + acq_regret("high_selectivity", "both", "random", 5)
        + " for random ranking. Predicted signed selectivity left a still smaller high-selectivity regret ("
        + acq_regret("high_selectivity", "both", "y_pred", 5)
        + ") even though threshold margin had the stronger event metrics. Near-zero LCB-2 regret@5 was "
        + acq_regret("low_selectivity", "near_zero", "lcb_2sigma", 5)
        + ", compared with "
        + acq_regret("low_selectivity", "near_zero", "random", 5)
        + " for random ranking; predicted magnitude was similar ("
        + acq_regret("low_selectivity", "near_zero", "y_pred", 5)
        + "). Residual regret remains, so recovering an event is not equivalent to capturing a pool-optimal ligand."
    )
    a("")
    a("The high-selectivity regret values are gaps to the strongest observed candidate in each eligible pool, not prediction errors and not distances from the event threshold. Near-zero regret instead measures excess absolute selectivity relative to the closest-to-zero available ligand.")
    a("")
    a("These results show that the ranking objective must be specified before choosing how predicted value and uncertainty are combined. They do not identify one universally optimal acquisition score. Ranking used only $\\widehat y$ and $\\sigma_{\\mathrm{total}}$. The selected workhorse was not identified using pair-acquisition event precision@$k$, true hit@$k$, or retrospective regret@$k$. The result applies only to pair-tasks with more than one held-out event and, for a given $k$, at least $k$ held-out rows. A supporting sensitivity with high-selectivity events at $\\Delta\\log K>2$ and near-zero events at $|\\Delta\\log K|<1/2$ is Table S10.8 and Supplementary Figure S15; it does not replace Table 4 or Figure 5.")
    a("")
    a("**Table 4. True hit@$k$, event precision@$k$, and retrospective regret@$k$ for the retrospective ranking benchmark (fold mean ± SD).**")
    a(
        note_block(
            "True hit@$k$ is the pair-task fraction of rankings that contain at least one observed event among the top $k$. Event precision@$k$ is the fraction of that shortlist that are events. The two event metrics coincide at $k=1$. Retrospective regret@$k$ is the leftover objective gap to the best ligand in the same pair-task pool (Methods 2.6); lower is better, and zero means the shortlist contains a pool-optimal ligand. Event metrics and regret are not interchangeable. Eligible pair-tasks have "
            + pair_acq_event_elig_text()
            + ". A pair-task contributes to a given $k$ only if it has at least $k$ held-out rows; $n@k$ is that pair-task count. High-selectivity event = observed target-direction $\\Delta\\log K>1$. Near-zero event = observed $|\\Delta\\log K|<1$. The threshold-margin score is $(\\widehat y^\\star-1)/\\sigma$ or $(1-|\\widehat y|)/\\sigma$, not the Gaussian expected-improvement integral. Random is 20 within-pair permutations (seed 0). Aggregation = pair-equal mean inside each fold, then fold mean ± SD. Complete $k=1,5,10$ inventories: Tables S10.2, S10.3, and S10.4. This table summarizes the retrospective ranking evaluation defined in Methods 2.6. Sources: `results/downstream_cv/pair_acquisition/pair_acquisition_hit_and_precision.csv` and `pair_acquisition_regret.csv`."
        )
    )
    tbl4 = acq_table_compact()
    if tbl4:
        a(tbl4)
    a("")
    a("**Figure 5. Objective-specific pair acquisition.** Ranking rules are evaluated separately for high-selectivity and near-zero objectives using true-hit rate, event precision, and retrospective regret at $k=1$, $5$, and $10$. High-selectivity events are target-direction $\\Delta\\log K>1$; near-zero events are $|\\Delta\\log K|<1$. The left column (A, C, E) is high selectivity and the right column (B, D, F) is near-zero selectivity. Rows are true hit@$k$, event precision@$k$, and retrospective regret@$k$. Panels E and F report retrospective regret@$k$ (lower is better) on the same pair-task shortlists. Table 4 reports the main-text subset, including retrospective regret@$k$; Table S10.4 reports the complete retrospective regret inventory. These panels summarize the retrospective held-out ranking evaluation defined in Methods 2.6. Ranking-function and retrospective regret@$k$ definitions are in Methods 2.6.")
    a("")
    a("![Figure 5](figures/fig05_pair_acquisition.png)")
    a("")
    a("## 4. Discussion")
    a("")
    a("### 4.1 Task alignment")
    a("")
    a("The central result is task alignment rather than a universally superior architecture. Literature M4, DS1, and DS2 encode different stages and prediction targets, and the distinction is necessary for interpreting their results. For the pair-specific DS2 decision, all-pair MoP evaluation gives each eligible metal pair equal weight and exposes behavior that row-level averages can obscure.")
    a("")
    a("The corresponding DS1 and DS2 composition, split, and model-grid inventories are provided in Supplementary Tables S1.1–S1.4 and S4.1–S4.4.")
    a("")
    a("### 4.2 Official-split Chemprop comparison")
    a("")
    if arch_test and arch_ext and wt_ext_ens and wt_test.get("fold_mean_mae") is not None:
        a(
            "Table 1 is a protocol comparison on the official CMI assignment, not the ligand-disjoint DS1 or DS2 headline. "
            "Three Chemprop 1.6.1 evaluations use the published complex SMILES and the same 16 additional CMI feature columns as the released additional-feature `best_model`. "
            "First, a from-scratch D-MPNN with the shared study architecture (hidden size 300, depth 3) does not load CMI `.pt` files and gives official-test MAE "
            + cmi_test_mae
            + " and $R^2$ "
            + cmi_test_r2
            + " and official-external MAE "
            + cmi_ext_mae
            + " and $R^2$ "
            + cmi_ext_r2
            + ". "
            "Second, scoring the released five-fold `best_model` checkpoints reproduces the published external ensemble (MAE "
            + f"{float(wt_ext_ens['mae']):.3f}"
            + ", $R^2$ "
            + f"{float(wt_ext_ens['r2']):.3f}"
            + ") and yields a mean of five fold-wise metrics on the open test file of MAE "
            + f"{float(wt_test['fold_mean_mae']):.3f}"
            + " and $R^2$ "
            + f"{float(wt_test['fold_mean_r2']):.3f}"
            + ", compared with the published five-fold CV-average test MAE "
            + pub_test_mae
            + " and $R^2$ "
            + pub_test_r2
            + ". "
            "Third, retraining from random initialization with the architecture recovered from those checkpoints (hidden size 2400, depth 6, three feed-forward layers of size 2400, batch size 50, 30 epochs, five random folds on the official train file) produces a five-fold ensemble MAE of "
            + f"{float(arch_test['mae']):.3f}"
            + " and $R^2$ "
            + f"{float(arch_test['r2']):.3f}"
            + " on the official test file and MAE "
            + f"{float(arch_ext['mae']):.3f}"
            + " and $R^2$ "
            + f"{float(arch_ext['r2']):.3f}"
            + " on the official external file."
        )
        a("")
        arch_fm = (
            " The architecture-matched test fold-mean MAE is "
            + f"{float(arch_test_fold['fold_mean_mae']):.3f}"
            + " versus the released-weight fold-mean MAE "
            + f"{float(wt_test['fold_mean_mae']):.3f}"
            + "."
            if arch_test_fold.get("fold_mean_mae") is not None
            else ""
        )
        a(
            "The weaker shared-architecture scores are therefore an architecture and training-protocol difference relative to Karunaratne et al., not a failure to evaluate the published weights."
            + arch_fm
            + " The architecture-matched retrain does not reproduce the published external ensemble. "
            "Chemprop 1.6.1 was used both to load the 1.5.2 checkpoints and to retrain. "
            + frac_lig_s
            + " of official test ligands already appear in train, so these numbers do not replace the ligand-disjoint DS1 Chemprop MAE of "
            + chem_ds1_mae
            + ". Protocol detail is in Supplementary Section S1.5.2 and Tables S1.6–S1.8."
        )
    else:
        a("Table 1 is a protocol comparison on the official CMI assignment. Architecture-matched retrain metrics are reported when the corresponding Stage-1 artifact is complete (S1.5.2).")
    a("")
    a("### 4.3 What the distance result means")
    a("")
    a("The tested molecular-distance methods were weaker empirical risk-ordering variables than model-derived uncertainty in this task. The RDKit feature-centroid Mahalanobis score produced an intermediate retained-subset improvement, while Tanimoto nearest-neighbor distance remained closer to random-retention behavior. The fold-wise distance–residual associations in Supplementary Table S8.4 provide a complementary descriptive diagnostic, while Supplementary Figure S11 shows the corresponding all-pair retention curves. We therefore do not use the tested distance scores as the primary operational risk rule. This is a comparative result for the tested representations, distances, dataset, split, and endpoint, not a universal claim that molecular similarity cannot predict error. The result is not a universal applicability-domain boundary.")
    a("")
    a("### 4.4 What uncertainty does and does not provide")
    a("")
    a("For the selected direct-UQ workhorse and the primary all-pair retention comparison, $\\sigma_{\\mathrm{total}}$ provided the strongest descriptive empirical ordering among the tested workhorse uncertainty and molecular-distance scores. Predictive uncertainty was useful because it supplied a model-specific risk ordering. It was not useful because it certified the sign of selectivity. Most intervals overlapped zero, including "
      + sign_pct("all_pairs", "overlaps_zero")
      + " of all-pair rows. Supplementary Table S9.1 and Supplementary Figure S14 show the full sign-category and sign-accuracy breakdown. Thus, the uncertainty score helps decide which predictions to trust or prioritize; it does not establish whether the underlying selectivity is positive or negative. The intervals are therefore not sign-certifying.")
    a("")
    a("### 4.5 Implications for experimental prioritization")
    a("")
    a("The acquisition results show that safer-prediction retention and objective-specific candidate selection are different problems. High-selectivity event recovery favored the threshold-margin score, whereas incorporating uncertainty into the ranking improves near-zero event precision over random selection; differences among the uncertainty-aware rules are small at larger shortlist sizes. Retrospective regret@$k$ shows a further distinction: event recovery is not the same as capturing the best held-out ligand in a pair-task. The complete $k=1,5,10$ inventories are given in Supplementary Tables S10.2–S10.4, and the stricter-threshold sensitivity is reported separately in Supplementary Table S10.8 and Figure S15. On the high-selectivity benchmark, predicted signed selectivity left a smaller residual gap to the pool optimum than threshold margin, even though threshold margin recovered events more often. The correct ranking function therefore depends on whether the next experiment is intended to find any qualifying event, to approach the strongest available candidate, or to exclude high-selectivity ligands.")
    a("")
    a("### 4.6 Limitations")
    a("")
    a("The conclusions are bounded by several features of the study. Literature M4 is heterogeneous, and repeated nominally matching records introduce target variability that was not identifiable as a separate replicate-spread estimand under the ligand-disjoint folds. Pair support is uneven, so the all-pair MoP results describe the supported-pair evaluation regime rather than uniform behavior over every possible metal pair. Five-fold ligand-disjoint validation is one generalization estimand; it does not replace a prospective campaign or a dedicated metal-pair holdout. The official-split Chemprop comparison in Table 1, including released-weight scoring and the architecture-matched retrain, is a historical protocol audit and is not a ligand-disjoint generalization estimate. Workhorse selection is exploratory and conditional on the tested model grid, interval-usability convention, and Pareto procedure. The tested molecular-distance definitions are also conditional design choices and were weaker than model-derived uncertainty in the primary retention comparison. Pair-acquisition eligibility uses observed held-out event support, making the evaluated task set label-informed. The rankings themselves use only out-of-fold predictions and uncertainty, while held-out labels are used afterward for event and regret scoring. The acquisition results, including retrospective regret@$k$, are therefore retrospective ranking evaluations rather than prospective experimental validation. Regret uses held-out labels to identify the pool-optimal ligand and is therefore not a quantity available at ranking time. Finally, $\\Delta\\log K$ is a relative binding-preference target rather than a process-level separation factor, and its uncertainty intervals generally do not certify the sign of selectivity.")
    a("")
    a("A restricted rare-earth-element pair analysis is provided in Supplementary Section S13 as a restricted formulation diagnostic. It is not part of the primary all-pair MoP evaluation, was not used for workhorse selection, and was not used to establish the all-pair risk-ordering result.")
    a("")
    a("## 5. Conclusions")
    a("")
    a("The central result is task alignment rather than a universally superior architecture. Literature M4 supplies heterogeneous measurements, while DS1 and DS2 encode different absolute-binding and pairwise-selectivity tasks. For DS2, ligand-disjoint all-pair MoP evaluation aligns the headline metric with the pair-specific decision and exposes behavior that row-level averages can obscure. For the fixed heteroscedastic ensemble, direct uncertainty provided the strongest descriptive ordering of held-out prediction risk among the primary workhorse and molecular-distance scores tested in this validation regime. The acquisition results show that safer-prediction retention and objective-specific candidate selection are distinct problems: threshold margin favors high-selectivity event recovery, whereas incorporating uncertainty into the ranking improves near-zero event precision over random selection; differences among the uncertainty-aware rules are small at larger shortlist sizes. Retrospective regret@$k$ shows that those event metrics do not replace the leftover gap to the best held-out ligand in a pair-task. These findings do not establish a universal applicability-domain boundary, sign-certifying intervals, or prospective experimental performance. A restricted rare-earth-element pair diagnostic is reported only in the Supplementary Information.")
    a("")
    a("## Data and artifact availability")
    a("")
    a("Canonical metrics: `results/cv_metrics/fold_metric_summary.csv`. Official-split audit: `results/stage1/stage1a_cmisplit_results.json`, `results/stage1/cmi_reproduction_results.json`, and `results/canonical/cmi_split_ligand_overlap.json`. Ad hoc versus direct selectivity: `results/downstream_cv/ad_hoc_selectivity/`. Workhorse selection: `results/manuscript_final/workhorse_selection_accuracy_calibration.json`. All-pair retention-analysis validation: `results/downstream_cv/ad_retention/` and `results/manuscript_final/ad_retention_validation.json`. Sigma-only pair acquisition: `results/downstream_cv/pair_acquisition/`. Manuscript tables: `manuscripts/final/tables/`. Figures: `manuscripts/final/figures/`.")
    a("")
    return main
