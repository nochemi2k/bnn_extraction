"""Build M4 / DS1 / DS2 tables, folds, and fold-local features. No UQ training.

Use ``--verify`` to rebuild feature matrices in memory and compare them to the
saved fold files without writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake import REMAKE_ROOT


class _NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, cls=_NpEnc)


from uq_remake.data.build_ds2 import (
    build_ds2,
    build_ds2_rdkit_matrix,
    last_medium_key_audit,
    make_ds2_folds,
    save_ds2,
    summarize_ds2,
    write_chemprop_ds2_fold_csvs,
)
from uq_remake.data.build_m4 import (
    build_m4_table,
    make_project_folds,
    save_processed,
    write_chemprop_fold_csvs,
)
from uq_remake.data.descriptor_audit import (
    audit_rdkit_descriptors,
    save_descriptor_audit,
    save_rdkit_selection,
    select_rdkit_descriptors,
)
from uq_remake.data.feature_schema import (
    INCLUDE_SOLVENT_PROPERTIES,
    PINNED_RDKIT_VERSION,
    ds1_rdkit_dimension,
    ds2_rdkit_dimension,
    schema_dict,
    verify_feature_name_layout,
)
from uq_remake.data.metal_props import (
    METAL_PROPS_CONVENTION,
    build_metal_property_table,
    last_fallback_report,
    save_metal_property_reports,
)
from uq_remake.data.rdkit_features import (
    build_rdkit_matrix,
    fold_manifest,
    load_fold_features,
    preprocess_fold,
    rdkit_version,
    save_fold_features,
)


def summarize_replicates(df: pd.DataFrame) -> dict:
    multi = df[df["n_replicates_in_key"] >= 2].copy()
    keys = multi.drop_duplicates("experiment_key")
    summary = {
        "n_records": int(len(df)),
        "n_unique_ligands": int(df["ligand_smiles"].nunique()),
        "n_metals": int(df.loc[df["family"] == "metal_complex", "metal_el"].nunique()),
        "n_species_keys": int(df["species_key"].nunique()),
        "n_experiment_keys": int(df["experiment_key"].nunique()),
        "n_multi_replicate_keys": int(len(keys)),
        "replicate_spread_median_multi": float(keys["replicate_spread"].median()) if len(keys) else None,
        "replicate_spread_mean_multi": float(keys["replicate_spread"].mean()) if len(keys) else None,
        "stratum_counts_keys": keys["replicate_stratum"].value_counts().to_dict() if len(keys) else {},
        "hard_disagreement_rate_by_family": {},
        "family_counts": df["family"].value_counts().to_dict(),
        "cmi_external_excluded": True,
    }
    for fam in ("protonation", "metal_complex"):
        fam_keys = df[df["family"] == fam].drop_duplicates("experiment_key")
        fam_keys = fam_keys[fam_keys["n_replicates_in_key"] >= 2]
        if len(fam_keys) == 0:
            summary["hard_disagreement_rate_by_family"][fam] = None
        else:
            summary["hard_disagreement_rate_by_family"][fam] = float(
                (fam_keys["replicate_stratum"] == "hard_disagreement").mean()
            )
    multi = df[df["n_replicates_in_key"] >= 2]
    within_std = multi.groupby("experiment_key")["logK"].std()
    keys = multi.drop_duplicates("experiment_key")
    non_hard_keys = keys.loc[keys["replicate_stratum"] != "hard_disagreement", "experiment_key"]
    std_non_hard = within_std.loc[within_std.index.isin(set(non_hard_keys))]
    if len(std_non_hard):
        summary["aleatoric_noise_floor"] = float(std_non_hard.median())
        summary["aleatoric_noise_floor_mean_std"] = float(std_non_hard.mean())
        summary["replicate_spread_median_agree"] = (
            float(keys.loc[keys["replicate_stratum"] == "agree", "replicate_spread"].median())
            if (keys["replicate_stratum"] == "agree").any()
            else None
        )
        y = df.loc[df["family"] == "metal_complex", "logK"].to_numpy(dtype=float)
        var_y = float(np.var(y))
        sig = summary["aleatoric_noise_floor"]
        summary["metal_complex_logK_variance"] = var_y
        summary["r2_ceiling_from_noise_floor"] = float(1.0 - (sig ** 2) / var_y) if var_y > 0 else None
        hard_frac = float((keys["replicate_stratum"] == "hard_disagreement").mean())
        summary["hard_disagreement_key_fraction"] = hard_frac
    return summary


def _ligand_parse_summary(audit: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(audit)),
        "parse_status": audit["parse_status"].value_counts().to_dict() if len(audit) else {},
        "n_quarantine": int(audit["quarantine"].sum()) if "quarantine" in audit.columns else 0,
        "n_empty_ligand": int((audit["ligand_smiles"].astype(str) == "").sum()),
        "n_failed_parts": int((audit["n_failed_parts"] > 0).sum()),
        "n_discarded_organic_fragments_rows": int((audit["n_discarded_organic_fragments"] > 0).sum()),
        "n_multi_ligand_fragments": int((audit["n_ligand_fragments"] > 1).sum()),
        "n_metal_label_disagreement": int(audit["metal_label_disagreement"].sum())
        if "metal_label_disagreement" in audit.columns
        else 0,
        "convention": (
            "Ligand = largest organic fragment by heavy atoms; else largest non-metal fragment. "
            "Proton tokens are exact [H+]/[H]/H+ only (Hg/Ho/Hf are metals)."
        ),
    }


def _solvent_audit(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["_mw"] = work["solvent_mw"].round(3)
    work["_dens"] = work["solvent_density"].round(2)
    pairs = (
        work.groupby(["electrolyte", "_mw", "_dens"], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={"_mw": "solvent_mw_rounded", "_dens": "solvent_density_rounded"})
    )
    n_elec = int(work["electrolyte"].nunique())
    n_pairs = int(len(pairs))
    return {
        "n_rows": int(len(work)),
        "n_missing_mw": int(work["solvent_mw"].isna().sum()),
        "n_missing_density": int(work["solvent_density"].isna().sum()),
        "n_unknown_electrolyte": int((work["electrolyte"] == "UNKNOWN").sum()),
        "n_electrolytes": n_elec,
        "n_unique_mw_density_pairs": n_pairs,
        "one_to_one_with_electrolyte": bool(n_elec == n_pairs and n_pairs > 0),
        "include_solvent_properties": INCLUDE_SOLVENT_PROPERTIES,
        "electrolyte_onehot_in_X": False,
        "prospective": True,
        "derived_from_target": False,
        "units": {"solvent_mw": "g/mol (CMI Exact MW of solvent)", "solvent_density": "g cm^-3"},
        "pairs": pairs.to_dict(orient="records"),
        "warning": (
            "In this corpus solvent_mw and solvent_density are a perfect 1:1 encoding of "
            "electrolyte identity (six discrete media). They are kept as continuous medium "
            "features together with T and I. Electrolyte strings are for matching, grouping, "
            "reporting, and diagnostics only — there is no electrolyte one-hot block."
        )
        if n_elec == n_pairs
        else "",
    }


def _ie_not_from_target(df: pd.DataFrame) -> dict:
    mc = df[df["family"] == "metal_complex"].copy()
    mc["_ox"] = pd.to_numeric(mc["metal_oxidation_state"], errors="coerce").round(1)
    g = mc.groupby(["metal_el", "_ox"], dropna=False).agg(
        n=("logK", "size"),
        n_ie=("IE", "nunique"),
        n_val=("metal_valence_e", "nunique"),
        n_logK=("logK", "nunique"),
    )
    return {
        "source": "CMI feature CSV columns IE and metal_valence_e (not Exp LgK1 / logK)",
        "convention": METAL_PROPS_CONVENTION,
        "n_species": int(len(g)),
        "n_species_ie_not_unique": int((g["n_ie"] > 1).sum()),
        "n_species_valence_not_unique": int((g["n_val"] > 1).sum()),
        "oxidation_state_is_separate_feature": True,
        "note": (
            "IE and valence are unique given (element, oxidation_state) in this corpus, "
            "so they cannot be a within-species function of the logK target. Oxidation state "
            "is the explanatory input; row-level IE/valence are derived CMI properties."
        ),
    }


def _save_fallback_csv(rows: list[dict], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["row_index", "element", "oxidation_state", "column", "fallback"]).to_csv(
            path, index=False
        )


def _load_hplus_decision() -> str:
    for path in (
        REMAKE_ROOT / "results" / "canonical" / "hplus_decision.json",
        REMAKE_ROOT / "results" / "stage1" / "hplus_decision.json",
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["decision"]
    print("WARNING: hplus_decision.json missing; defaulting to train_with_H")
    return "train_with_H"


def _write_rdkit_folds(
    dataset: str,
    df: pd.DataFrame,
    folds: dict,
    X: np.ndarray,
    y: np.ndarray,
    out_root: Path,
) -> None:
    for fold_name, split in folds.items():
        tr = np.asarray(split["train_idx"], dtype=int)
        va = np.asarray(split["val_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)
        Xtr, Xva, Xte, prep = preprocess_fold(X[tr], X[va], X[te])
        manifest = fold_manifest(dataset, fold_name, prep, len(tr), len(va), len(te))
        save_fold_features(out_root / fold_name, Xtr, Xva, Xte, y[tr], y[va], y[te], manifest)
        print(
            f"  {dataset} {fold_name}: dim={manifest['feature_dimension']} "
            f"ntr={len(tr)} nva={len(va)} nte={len(te)}"
        )


def main() -> None:
    audit_dir = REMAKE_ROOT / "data" / "processed" / "audits"
    feat_dir = REMAKE_ROOT / "data" / "processed" / "features"
    res1 = REMAKE_ROOT / "results" / "stage1"
    audit_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    res1.mkdir(parents=True, exist_ok=True)

    print("Building M4 from CMI raw CSVs (external validation held out)...")
    modeling, external, parse_audit = build_m4_table()
    quarantined = parse_audit.loc[parse_audit["quarantine"]].copy() if "quarantine" in parse_audit.columns else parse_audit.iloc[0:0]
    print(
        f"  modeling={len(modeling)} external={len(external)} "
        f"quarantine={len(quarantined)} ligands={modeling['ligand_smiles'].nunique()}"
    )
    if len(quarantined):
        print("ERROR: quarantined rows present; not writing modeling tables.")
        quarantined.to_csv(audit_dir / "ligand_parse_quarantine.csv", index=False)
        parse_audit.to_csv(audit_dir / "ligand_parse_audit.csv", index=False)
        raise SystemExit(f"Quarantined {len(quarantined)} rows. Inspect ligand_parse_quarantine.csv")

    parse_summary = _ligand_parse_summary(parse_audit)
    (audit_dir / "ligand_parse_summary.json").write_text(_dump(parse_summary), encoding="utf-8")
    parse_audit.to_csv(audit_dir / "ligand_parse_audit.csv", index=False)

    solvent = _solvent_audit(pd.concat([modeling, external], ignore_index=True))
    (audit_dir / "solvent_audit.json").write_text(_dump(solvent), encoding="utf-8")
    print(
        f"  solvent 1:1 with electrolyte={solvent['one_to_one_with_electrolyte']} "
        f"include_solvent={INCLUDE_SOLVENT_PROPERTIES}"
    )

    print("Building metal property table from CMI columns (row-level IE/valence)...")
    metal_payload = build_metal_property_table(pd.concat([modeling, external], ignore_index=True))
    save_metal_property_reports(metal_payload, audit_dir)
    ie_check = _ie_not_from_target(pd.concat([modeling, external], ignore_index=True))
    (audit_dir / "metal_props_ie_target_independence.json").write_text(_dump(ie_check), encoding="utf-8")
    print(
        f"  n_elements={metal_payload['n_elements']} n_species={metal_payload['n_species']} "
        f"convention={metal_payload['convention']} zero_fill=False"
    )

    folds = make_project_folds(modeling, n_folds=5, seed=42)
    save_processed(modeling, folds, external=external, parse_audit=parse_audit)

    summary = summarize_replicates(modeling)
    (res1 / "replicate_summary.json").write_text(_dump(summary), encoding="utf-8")
    rows = []
    for fam in ("protonation", "metal_complex"):
        fam_keys = modeling[modeling["family"] == fam].drop_duplicates("experiment_key")
        fam_keys = fam_keys[fam_keys["n_replicates_in_key"] >= 2]
        for stratum in ("agree", "soft_disagreement", "hard_disagreement"):
            n = int((fam_keys["replicate_stratum"] == stratum).sum())
            rows.append({"family": fam, "stratum": stratum, "n_experiment_keys": n})
    pd.DataFrame(rows).to_csv(res1 / "replicate_strata_table.csv", index=False)

    hplus = _load_hplus_decision()
    if hplus == "train_without_H":
        ds1 = modeling[modeling["family"] != "protonation"].copy()
        train_policy = "Protonation (H+) records excluded from training and from DS1."
    else:
        ds1 = modeling.copy()
        train_policy = (
            "Protonation (H+) records kept in training and in DS1; "
            "test metrics remain metal_complex-only. CMI external is not in DS1."
        )
    ds1_path = REMAKE_ROOT / "data" / "processed" / "ds1.csv"
    ds1.to_csv(ds1_path, index=False)
    composition = {
        "dataset": "DS1",
        "hplus_decision": hplus,
        "train_policy": train_policy,
        "n_records": int(len(ds1)),
        "n_unique_ligands": int(ds1["ligand_smiles"].nunique()),
        "n_metals": int(ds1.loc[ds1["family"] == "metal_complex", "metal_el"].nunique()),
        "n_species_keys": int(ds1["species_key"].nunique()),
        "family_breakdown": ds1["family"].value_counts().to_dict(),
        "cmi_external_excluded": True,
        "n_external_records": int(len(external)),
        "n_external_ligands": int(external["ligand_smiles"].nunique()) if len(external) else 0,
        "test_restriction": "metal_complex only for Stage 1b evaluation",
        "include_solvent_properties": INCLUDE_SOLVENT_PROPERTIES,
        "electrolyte_in_X": False,
        "metal_props_convention": METAL_PROPS_CONVENTION,
    }
    (res1 / "ds1_composition.json").write_text(_dump(composition), encoding="utf-8")
    pd.DataFrame([composition]).to_csv(res1 / "ds1_composition_table.csv", index=False)

    print("Building DS2 species-level pairs...")
    ds2 = build_ds2(ds1)
    ds2_summary = summarize_ds2(ds2)
    medium_audit = last_medium_key_audit()
    if medium_audit:
        ds2_summary["medium_key_audit"] = medium_audit
        (audit_dir / "ds2_medium_key_audit.json").write_text(_dump(medium_audit), encoding="utf-8")
    save_ds2(ds2, ds2_summary)
    ds2_folds = make_ds2_folds(ds2)
    (REMAKE_ROOT / "data" / "splits" / "ds2_project_folds.json").write_text(
        json.dumps(ds2_folds, indent=2), encoding="utf-8"
    )
    print(_dump(ds2_summary))

    print("Auditing full RDKit 2D list and applying unsupervised drop rules...")
    smiles_all = pd.concat(
        [ds1["ligand_smiles"], external["ligand_smiles"], ds2["ligand_smiles"]],
        ignore_index=True,
    )
    desc_audit = audit_rdkit_descriptors(smiles_all, folds=None, dataset="DS1")
    save_descriptor_audit(desc_audit, audit_dir)
    selection = select_rdkit_descriptors(desc_audit)
    save_rdkit_selection(selection)
    print(
        f"  full={selection['full_n']} keep={selection['selected_n']} "
        f"drop={selection['n_dropped']} "
        f"(missing/near-constant/duplicate/extreme; no imputation)"
    )

    composition["rdkit_dimension"] = ds1_rdkit_dimension()
    composition["rdkit_n_selected"] = selection["selected_n"]
    (res1 / "ds1_composition.json").write_text(_dump(composition), encoding="utf-8")
    pd.DataFrame([composition]).to_csv(res1 / "ds1_composition_table.csv", index=False)

    ds1_schema = schema_dict("DS1")
    ds2_schema = schema_dict("DS2")
    ds1_schema["rdkit_version"] = rdkit_version()
    ds2_schema["rdkit_version"] = rdkit_version()
    (feat_dir / "ds1_rdkit_schema.json").write_text(_dump(ds1_schema), encoding="utf-8")
    (feat_dir / "ds2_rdkit_schema.json").write_text(_dump(ds2_schema), encoding="utf-8")
    (feat_dir / "rdkit_descriptor_names.json").write_text(
        _dump(
            {
                "pinned_rdkit_version": PINNED_RDKIT_VERSION,
                "runtime_rdkit_version": rdkit_version(),
                "n": len(ds1_schema["descriptor_names"]),
                "n_full": 210,
                "names": ds1_schema["descriptor_names"],
                "dropped": ds1_schema["rdkit_dropped_names"],
                "ds1_feature_hash": ds1_schema["feature_hash"],
                "ds2_feature_hash": ds2_schema["feature_hash"],
            }
        ),
        encoding="utf-8",
    )
    (res1 / "feature_schema_note.json").write_text(
        _dump(
            {
                "rdkit_version": rdkit_version(),
                "pinned_rdkit_version": PINNED_RDKIT_VERSION,
                "DS1": ds1_schema,
                "DS2": ds2_schema,
                "CPEMB_side": "emb300 || metal_props(9, row-level CMI) || DS1_CONTINUOUS_COLS (no electrolyte one-hot)",
                "electrolyte_in_X": False,
                "metal_props_convention": METAL_PROPS_CONVENTION,
            }
        ),
        encoding="utf-8",
    )

    print(f"Building DS1 RDKit matrix (dim={ds1_rdkit_dimension()})...")
    X1, cols1 = build_rdkit_matrix(ds1)
    layout1 = verify_feature_name_layout("DS1", cols1)
    (audit_dir / "ds1_feature_layout.json").write_text(_dump(layout1), encoding="utf-8")
    fb1 = last_fallback_report()
    _save_fallback_csv(fb1, audit_dir / "metal_property_fallback_ds1.csv")
    if list(cols1) != ds1_schema["feature_names"]:
        raise RuntimeError("DS1 feature names drifted from schema")
    y1 = ds1["logK"].to_numpy(dtype=float)
    # Folds index m4_processed/modeling; map to ds1 via record_id when H+ dropped.
    if len(ds1) == len(modeling) and (ds1["record_id"].values == modeling["record_id"].values).all():
        folds_ds1 = folds
    else:
        rid_to_i = {r: i for i, r in enumerate(ds1["record_id"])}
        folds_ds1 = {}
        for name, split in folds.items():
            def _map(idxs):
                rids = modeling.iloc[np.asarray(idxs, dtype=int)]["record_id"]
                return [rid_to_i[r] for r in rids if r in rid_to_i]
            folds_ds1[name] = {
                "train_idx": _map(split["train_idx"]),
                "val_idx": _map(split["val_idx"]),
                "test_idx": _map(split["test_idx"]),
            }
    _write_rdkit_folds("DS1", ds1, folds_ds1, X1, y1, feat_dir / "rdkit_ds1")

    print(f"Building DS2 RDKit matrix (dim={ds2_rdkit_dimension()})...")
    X2, cols2 = build_ds2_rdkit_matrix(ds2)
    layout2 = verify_feature_name_layout("DS2", cols2)
    (audit_dir / "ds2_feature_layout.json").write_text(_dump(layout2), encoding="utf-8")
    fb2 = last_fallback_report()
    _save_fallback_csv(fb2, audit_dir / "metal_property_fallback_ds2.csv")
    if list(cols2) != ds2_schema["feature_names"]:
        raise RuntimeError("DS2 feature names drifted from schema")
    y2 = ds2["selectivity_logK"].to_numpy(dtype=float)
    _write_rdkit_folds("DS2", ds2, ds2_folds, X2, y2, feat_dir / "rdkit_ds2")

    print("Confirming selected RDKit block has no missing/constant/duplicate columns...")
    d_sel = selection["selected_n"]
    X_rd = X1[:, :d_sel]
    if not np.isfinite(X_rd).all():
        raise RuntimeError("Selected RDKit block still has non-finite values")
    nuniq = [int(len(np.unique(np.round(X_rd[:, j], 12)))) for j in range(d_sel)]
    if min(nuniq) <= 1:
        raise RuntimeError("Selected RDKit block still has a near-constant column")
    print(
        f"  selected RDKit finite={True} min_nunique={min(nuniq)} "
        f"ds1_fallbacks={len(fb1)} ds2_fallbacks={len(fb2)}"
    )

    print("Writing Chemprop fold CSVs (no imputation)...")
    for fold_name, split in folds.items():
        write_chemprop_fold_csvs(
            modeling,
            split["train_idx"],
            split["val_idx"],
            split["test_idx"],
            feat_dir / "chemprop_ds1" / fold_name,
        )
    for fold_name, split in ds2_folds.items():
        write_chemprop_ds2_fold_csvs(
            ds2,
            split["train_idx"],
            split["val_idx"],
            split["test_idx"],
            feat_dir / "chemprop_ds2" / fold_name,
        )

    cmi = json.loads((REMAKE_ROOT / "data" / "splits" / "cmi_official_split.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(cmi["train_idx"], dtype=int)
    test_idx = np.asarray(cmi["test_idx"], dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_rel, va_rel = next(
        gss.split(train_idx, groups=modeling.iloc[train_idx]["ligand_smiles"].astype(str).values)
    )
    # External is a separate table; write Chemprop files with train medians from CMI train.
    ext_work = feat_dir / "chemprop_cmisplit"
    write_chemprop_fold_csvs(
        modeling,
        train_idx[tr_rel],
        train_idx[va_rel],
        test_idx,
        ext_work,
    )
    from uq_remake.data.build_m4 import chemprop_train_medians, write_chemprop_csvs

    med = chemprop_train_medians(modeling, train_idx[tr_rel])
    write_chemprop_csvs(
        external,
        np.arange(len(external)),
        ext_work / "ext.csv",
        ext_work / "ext_features.csv",
        train_feature_medians=med,
    )

    findings = {
        "rdkit_version": rdkit_version(),
        "pinned_rdkit_version": PINNED_RDKIT_VERSION,
        "ds1_rdkit_dimension": ds1_rdkit_dimension(),
        "ds2_rdkit_dimension": ds2_rdkit_dimension(),
        "ds1_feature_hash": ds1_schema["feature_hash"],
        "ds2_feature_hash": ds2_schema["feature_hash"],
        "include_solvent_properties": INCLUDE_SOLVENT_PROPERTIES,
        "imputation": False,
        "rdkit_n_full": selection["full_n"],
        "rdkit_n_selected": selection["selected_n"],
        "rdkit_n_dropped": selection["n_dropped"],
        "rdkit_dropped_names": [r["name"] for r in selection["dropped"]],
        "electrolyte_onehot_in_X": False,
        "solvent_one_to_one_electrolyte": solvent["one_to_one_with_electrolyte"],
        "metal_props_convention": METAL_PROPS_CONVENTION,
        "metal_prop_fallback_n_ds1": len(fb1),
        "metal_prop_fallback_n_ds2": len(fb2),
        "medium_key_sufficient": None if not medium_audit else medium_audit.get("sufficient"),
        "rdkit_near_constant": desc_audit["n_near_constant"],
        "rdkit_duplicate_pairs": desc_audit["n_duplicate_pairs"],
        "cmi_external_excluded_from_folds": True,
        "n_modeling": int(len(modeling)),
        "n_external": int(len(external)),
        "n_ds1": int(len(ds1)),
        "n_ds2": int(len(ds2)),
        "n_metal_elements": metal_payload["n_elements"],
        "n_metal_species": metal_payload["n_species"],
        "ligand_parse": parse_summary,
        "ie_not_from_target": ie_check,
        "layout_ds1": layout1,
        "layout_ds2": layout2,
        "questions_for_user": [],
        "decisions_applied": [
            "Keep solvent MW/density + T + I; electrolyte string for matching/grouping/reporting only; no electrolyte one-hot.",
            "RDKit 2023.09.6 full list is 210 names; missing, near-constant, duplicate, and extreme descriptors are dropped (no imputation).",
            "Row-level CMI IE and metal_valence_e in tabular features (same as Chemprop); oxidation state remains a separate feature.",
        ],
    }
    (audit_dir / "prepare_data_findings.json").write_text(_dump(findings), encoding="utf-8")
    print(_dump(findings))
    print("Wrote processed tables, folds, audits, and fold-local feature files. No models trained.")


def _allclose_float32(a, b, *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return False
    return bool(np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True))


def verify_existing() -> int:
    """Confirm frozen tables/folds and that current builders reproduce saved features."""
    issues: list[str] = []
    feat_dir = REMAKE_ROOT / "data" / "processed" / "features"
    ds1_path = REMAKE_ROOT / "data" / "processed" / "ds1.csv"
    ds2_path = REMAKE_ROOT / "data" / "processed" / "ds2.csv"
    folds_ds1_path = REMAKE_ROOT / "data" / "splits" / "project_folds.json"
    folds_ds2_path = REMAKE_ROOT / "data" / "splits" / "ds2_project_folds.json"
    for path in (ds1_path, ds2_path, folds_ds1_path, folds_ds2_path, feat_dir / "ds1_rdkit_schema.json", feat_dir / "ds2_rdkit_schema.json"):
        if not path.exists():
            issues.append(f"missing:{path.relative_to(REMAKE_ROOT).as_posix()}")
    if issues:
        print("VERIFY FAIL")
        print("\n".join(issues))
        return 1

    ds1 = pd.read_csv(ds1_path)
    ds2 = pd.read_csv(ds2_path)
    folds_ds1 = json.loads(folds_ds1_path.read_text(encoding="utf-8"))
    folds_ds2 = json.loads(folds_ds2_path.read_text(encoding="utf-8"))
    saved_ds1 = json.loads((feat_dir / "ds1_rdkit_schema.json").read_text(encoding="utf-8"))
    saved_ds2 = json.loads((feat_dir / "ds2_rdkit_schema.json").read_text(encoding="utf-8"))
    live_ds1 = schema_dict("DS1")
    live_ds2 = schema_dict("DS2")
    if live_ds1["feature_hash"] != saved_ds1.get("feature_hash"):
        issues.append(f"ds1_feature_hash mismatch live={live_ds1['feature_hash']} saved={saved_ds1.get('feature_hash')}")
    if live_ds2["feature_hash"] != saved_ds2.get("feature_hash"):
        issues.append(f"ds2_feature_hash mismatch live={live_ds2['feature_hash']} saved={saved_ds2.get('feature_hash')}")
    if live_ds1["feature_names"] != saved_ds1.get("feature_names"):
        issues.append("ds1_feature_names drifted from saved schema")
    if live_ds2["feature_names"] != saved_ds2.get("feature_names"):
        issues.append("ds2_feature_names drifted from saved schema")
    if rdkit_version() not in {PINNED_RDKIT_VERSION, "2023.9.6"}:
        issues.append(f"rdkit_version {rdkit_version()} is not pinned {PINNED_RDKIT_VERSION}")

    verify_feature_name_layout("DS1", live_ds1["feature_names"])
    verify_feature_name_layout("DS2", live_ds2["feature_names"])

    from uq_remake.data.feature_schema import CHEMPROP_DS1_FEATURE_COLUMNS
    from uq_remake.data.build_ds2 import CHEMPROP_DS2_FEATURE_COLUMNS

    print("Rebuilding DS1 RDKit matrix in memory (no writes)...", flush=True)
    X1, cols1 = build_rdkit_matrix(ds1)
    if list(cols1) != live_ds1["feature_names"]:
        issues.append("ds1 rebuilt column names drifted")
    y1 = ds1["logK"].to_numpy(dtype=float)
    for fold_name, split in folds_ds1.items():
        tr = np.asarray(split["train_idx"], dtype=int)
        va = np.asarray(split["val_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)
        if max(tr.max(), va.max(), te.max()) >= len(ds1):
            issues.append(f"DS1 {fold_name} index out of range")
            continue
        Xtr, Xva, Xte, prep = preprocess_fold(X1[tr], X1[va], X1[te])
        arrays, manifest = load_fold_features(feat_dir / "rdkit_ds1" / fold_name)
        if not _allclose_float32(Xtr, arrays["X_train"]) or not _allclose_float32(Xva, arrays["X_val"]) or not _allclose_float32(Xte, arrays["X_test"]):
            issues.append(f"DS1 {fold_name} RDKit arrays differ from rebuilt features")
        if not _allclose_float32(y1[te], arrays["y_test"]):
            issues.append(f"DS1 {fold_name} y_test differs")
        if manifest.get("feature_hash") != live_ds1["feature_hash"]:
            issues.append(f"DS1 {fold_name} manifest feature_hash drifted")
        if not _allclose_float32(prep["scaler_mean"], manifest.get("scaler_mean", []), atol=1e-5):
            issues.append(f"DS1 {fold_name} scaler_mean differs")
        chem = feat_dir / "chemprop_ds1" / fold_name / "test_features.csv"
        if not chem.exists():
            issues.append(f"missing:{chem.relative_to(REMAKE_ROOT).as_posix()}")
        else:
            cols = list(pd.read_csv(chem, nrows=0).columns)
            if cols != list(CHEMPROP_DS1_FEATURE_COLUMNS):
                issues.append(f"DS1 {fold_name} Chemprop feature columns drifted n={len(cols)}")
        print(f"  DS1 {fold_name} ok_so_far={not any(fold_name in i and i.startswith('DS1') for i in issues)}", flush=True)

    print("Rebuilding DS2 RDKit matrix in memory (no writes)...", flush=True)
    X2, cols2 = build_ds2_rdkit_matrix(ds2)
    if list(cols2) != live_ds2["feature_names"]:
        issues.append("ds2 rebuilt column names drifted")
    y2 = ds2["selectivity_logK"].to_numpy(dtype=float)
    for fold_name, split in folds_ds2.items():
        tr = np.asarray(split["train_idx"], dtype=int)
        va = np.asarray(split["val_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)
        if max(tr.max(), va.max(), te.max()) >= len(ds2):
            issues.append(f"DS2 {fold_name} index out of range")
            continue
        Xtr, Xva, Xte, prep = preprocess_fold(X2[tr], X2[va], X2[te])
        arrays, manifest = load_fold_features(feat_dir / "rdkit_ds2" / fold_name)
        if not _allclose_float32(Xtr, arrays["X_train"]) or not _allclose_float32(Xva, arrays["X_val"]) or not _allclose_float32(Xte, arrays["X_test"]):
            issues.append(f"DS2 {fold_name} RDKit arrays differ from rebuilt features")
        if not _allclose_float32(y2[te], arrays["y_test"]):
            issues.append(f"DS2 {fold_name} y_test differs")
        if manifest.get("feature_hash") != live_ds2["feature_hash"]:
            issues.append(f"DS2 {fold_name} manifest feature_hash drifted")
        chem = feat_dir / "chemprop_ds2" / fold_name / "test_features.csv"
        if not chem.exists():
            issues.append(f"missing:{chem.relative_to(REMAKE_ROOT).as_posix()}")
        else:
            cols = list(pd.read_csv(chem, nrows=0).columns)
            if cols != list(CHEMPROP_DS2_FEATURE_COLUMNS):
                issues.append(f"DS2 {fold_name} Chemprop feature columns drifted n={len(cols)}")
        print(f"  DS2 {fold_name} ok_so_far={not any(fold_name in i and i.startswith('DS2') for i in issues)}", flush=True)

    if issues:
        print("VERIFY FAIL")
        print("\n".join(issues))
        return 1
    print(
        json.dumps(
            {
                "status": "pass",
                "n_ds1": int(len(ds1)),
                "n_ds2": int(len(ds2)),
                "ds1_feature_hash": live_ds1["feature_hash"],
                "ds2_feature_hash": live_ds2["feature_hash"],
                "rdkit_version": rdkit_version(),
            },
            indent=2,
        )
    )
    return 0


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build or verify M4/DS1/DS2 tables, folds, and fold-local features")
    p.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild feature matrices in memory and compare to saved fold files; do not write",
    )
    args = p.parse_args(argv)
    if args.verify:
        raise SystemExit(verify_existing())
    main()


if __name__ == "__main__":
    cli()
