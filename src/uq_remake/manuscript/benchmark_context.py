"""The two facts the paper has to establish before it can ask its own question.

Both are read from frozen artifacts of the earlier analysis of this dataset.
No model is loaded, trained or rescored here.

  benchmark_reproduction.csv   What the published benchmark on this compilation
                               measures. Its own train/test assignment reuses
                               ligands heavily, so the published accuracy is
                               obtained in a partly familiar-ligand regime; the
                               released weights and an architecture-matched
                               retraining both reproduce it. Scoring the same
                               architecture under ligand-disjoint validation,
                               which is the regime a ligand designer is in,
                               gives a much larger error. The three rows are
                               three different questions, not three attempts at
                               one.

  selectivity_construction.csv Composing selectivity from two absolute-binding
                               predictions (the ad hoc route) against modelling
                               the pairwise target directly, with the point-only
                               ad hoc models carried as an accuracy reference
                               (they produce no interval, so their calibration
                               columns are empty). Point accuracy is comparable
                               across all three; the propagated intervals are
                               not, which is why the uncertainty has to be
                               attached to the quantity the decision consumes.
"""
import json
import os
import subprocess

import pandas as pd

T = os.environ.get("BNN_TABLES", os.path.join("manuscripts", "derived", "tables")) + os.sep
REPRO = os.path.join("results", "stage1", "cmi_reproduction_results.json")
OVERLAP = os.path.join("results", "canonical", "cmi_split_ligand_overlap.json")
DISJOINT = os.path.join("results", "canonical", "ds1_chemprop_ligand_disjoint_mae.json")
ADHOC = os.path.join("results", "downstream_cv", "ad_hoc_selectivity",
                     "ad_hoc_selectivity_comparison.csv")
DIRECT = os.path.join("results", "cv_metrics", "fold_metric_summary.csv")

# The direct model carried through the rest of the paper, and the ad hoc route
# with the best-calibrated propagated intervals, on the same all-pair scope.
WORKHORSE = "DS2_RDKIT_MLPENS5_HETERO"

# The direct grid is keyed by model id only. These are the same names the ad hoc
# file already carries for the matching DS1 architectures, so the two halves of
# the comparison are labelled alike.
DIRECT_NAME = {
    "DS2_RDKIT_MLPENS5_HETERO": "Heteroscedastic MLP ensemble",
    "DS2_RDKIT_MLP_HETERO": "Heteroscedastic MLP",
    "DS2_RDKIT_MLPENS5_EVID": "Evidential MLP ensemble",
    "DS2_RDKIT_MLP_EVID": "Evidential MLP",
    "DS2_RDKIT_MLPENS5_MCDROPOUT": "MC-dropout MLP ensemble",
    "DS2_RDKIT_MLP_MCDROPOUT": "MC-dropout MLP",
    "DS2_RDKIT_SNGPENS5": "SNGP ensemble",
    "DS2_RDKIT_SNGP": "SNGP",
    "DS2_RDKIT_MLP_BAYES_FULL": "Full variational BNN",
    "DS2_RDKIT_GPR_SPARSE": "Sparse variational GPR",
    "DS2_GRAPH_DMPNN_EVID": "Chemprop evidential",
}
AD_HOC_UQ = "ad hoc, difference of two absolute predictions"
AD_HOC_PT = "ad hoc, point models without an interval"
DIRECT_LBL = "direct, modelled on the pairwise target"


def benchmark_reproduction():
    r = json.load(open(REPRO))
    o = json.load(open(OVERLAP))
    d = json.load(open(DISJOINT))
    pub = r["cmi_published"]
    rw = r["released_weights"]["evaluations"]
    ar = r["architecture_retrain"]["evaluations"]
    rows = [
        dict(regime="published split, test partition",
             ligand_overlap_note=f"{o['n_ligands_shared_train_test']} of "
                                 f"{o['n_test_ligands']} test ligands also in training "
                                 f"({100 * o['frac_test_ligands_seen_in_train']:.1f}%); "
                                 f"{100 * o['frac_test_rows_with_train_ligand']:.1f}% of test rows",
             published_mae=pub["cmi_published_m4_test_mae"],
             published_r2=pub["cmi_published_m4_test_r2"],
             released_weights_mae=rw["cmi_test"]["fold_mean_mae"],
             released_weights_r2=rw["cmi_test"]["fold_mean_r2"],
             retrained_mae=ar["cmi_test"]["fold_mean_mae"],
             retrained_r2=ar["cmi_test"]["fold_mean_r2"]),
        dict(regime="published split, external holdout",
             ligand_overlap_note=f"none; {o['n_ext_ligands']} ligands absent from training",
             published_mae=pub["cmi_published_m4_ext_mae"],
             published_r2=pub["cmi_published_m4_ext_r2"],
             released_weights_mae=rw["cmi_ext"]["ensemble"]["mae"],
             released_weights_r2=rw["cmi_ext"]["ensemble"]["r2"],
             retrained_mae=ar["cmi_ext"]["ensemble"]["mae"],
             retrained_r2=ar["cmi_ext"]["ensemble"]["r2"]),
        dict(regime="ligand-disjoint cross-validation",
             ligand_overlap_note="none by construction, in every fold",
             published_mae=float("nan"), published_r2=float("nan"),
             released_weights_mae=float("nan"), released_weights_r2=float("nan"),
             retrained_mae=d["fold_mean"], retrained_r2=float("nan")),
    ]
    out = pd.DataFrame(rows)
    out["retrained_mae_fold_sd"] = [ar["cmi_test"]["fold_sd_mae"],
                                    float("nan"), d["fold_sd"]]
    out["n_train_ligands"] = o["n_train_ligands"]
    out["scope"] = "metal-complex records, absolute log K"
    return out


def selectivity_construction():
    """Ad hoc composition against direct modelling, all-pair scope, like for like."""
    a = pd.read_csv(ADHOC)
    a = a[a.scope == "all_pairs"]
    cols_in = ["source_model", "source_model_id", "mop_rmse", "mop_rmse_sd",
               "mop_picp_90", "mop_picp_90_sd", "mop_ace", "mop_ace_sd"]
    ad = a[a.method_class == "ad_hoc_ds1_uncertainty"][cols_in].copy()
    ad["construction"] = AD_HOC_UQ
    # Point models produce no interval. They are carried because they fix what
    # the accuracy of this construction is when nothing is asked of calibration,
    # which is the reference the interval-bearing models have to be read against.
    pt = a[a.method_class == "ad_hoc_ds1_point"][cols_in].copy()
    for c in ("mop_picp_90", "mop_picp_90_sd", "mop_ace", "mop_ace_sd"):
        pt[c] = float("nan")
    pt["construction"] = AD_HOC_PT

    f = pd.read_csv(DIRECT)
    f = f[(f.dataset == "DS2") & (f.scope == "all_pairs")
          & (f.metric_name.isin(["mop_rmse", "mop_picp_90", "mop_ace"]))]
    mean = f.pivot_table(index="model_id", columns="metric_name", values="fold_mean")
    sd = f.pivot_table(index="model_id", columns="metric_name", values="fold_sd")
    di = mean.join(sd, rsuffix="_sd").reset_index()
    di = di.rename(columns={"model_id": "source_model_id"})
    di = di.dropna(subset=["mop_ace"])
    di["source_model"] = di.source_model_id.map(DIRECT_NAME)
    assert di.source_model.notna().all(), sorted(set(di.source_model_id) - set(DIRECT_NAME))
    di["construction"] = DIRECT_LBL

    cols = ["construction", "source_model", "source_model_id", "mop_rmse",
            "mop_rmse_sd", "mop_picp_90", "mop_picp_90_sd", "mop_ace", "mop_ace_sd"]
    out = pd.concat([ad[cols], pt[cols], di[cols]], ignore_index=True)
    out["is_workhorse"] = out.source_model_id == WORKHORSE
    out["scope"] = "all_pairs"
    return out.sort_values(["construction", "mop_ace", "mop_rmse"]).reset_index(drop=True)


def _provenance(name, inputs, note):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unavailable"
    json.dump(dict(table=name, inputs=inputs, git_commit=commit, note=note),
              open(T + name + ".provenance.json", "w"), indent=2)


if __name__ == "__main__":
    b = benchmark_reproduction()
    b.to_csv(T + "benchmark_reproduction.csv", index=False)
    _provenance("benchmark_reproduction.csv", [REPRO, OVERLAP, DISJOINT],
                "what the published split measures, and the same architecture "
                "under ligand-disjoint validation")
    print("=== benchmark reproduction ===")
    for _, r in b.iterrows():
        print(f"  {r.regime:36s} published MAE {r.published_mae!s:>6.6} | "
              f"released {r.released_weights_mae!s:>6.6} | "
              f"retrained {r.retrained_mae!s:>6.6}")
        print(f"      overlap: {r.ligand_overlap_note}")

    s = selectivity_construction()
    s.to_csv(T + "selectivity_construction.csv", index=False)
    _provenance("selectivity_construction.csv", [ADHOC, DIRECT],
                "composing selectivity from two absolute predictions versus "
                "modelling the pairwise target directly, all-pair scope")
    print("\n=== selectivity construction, all-pair scope ===")
    for cls in s.construction.unique():
        print(f"  -- {cls}")
        for _, r in s[s.construction == cls].iterrows():
            star = " *" if r.is_workhorse else "  "
            print(f"   {star}{r.source_model:30s} MoP RMSE {r.mop_rmse:.3f}  "
                  f"PICP {r.mop_picp_90:.3f}  ACE {r.mop_ace:.3f}")
    for lbl, sel in (("ad hoc UQ", AD_HOC_UQ), ("ad hoc point", AD_HOC_PT), ("direct", DIRECT_LBL)):
        d = s[s.construction == sel]
        print(f"\n  {lbl:13s} n={len(d):2d}  MoP RMSE {d.mop_rmse.min():.3f}-{d.mop_rmse.max():.3f}"
              + ("" if d.mop_ace.isna().all() else f"  best ACE {d.mop_ace.min():.3f}"))
