"""Build DS2: pairwise selectivity_logK for matched ligand+medium metal pairs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from uq_remake import REMAKE_ROOT
from uq_remake.data.build_m4 import REE_ELEMENTS
from uq_remake.data.feature_schema import (
    INCLUDE_SOLVENT_PROPERTIES,
    assert_runtime_schema,
    ds1_rdkit_column_names,
    verify_feature_name_layout,
)
from uq_remake.data.metal_props import METAL_PROP_COLS, metal_props_matrix
from uq_remake.data.rdkit_features import build_rdkit_matrix
from uq_remake.data.species import canonical_element_pair, canonical_species_pair, species_key

MEDIUM_KEY_MUST_AGREE = (
    "solvent_mw",
    "solvent_density",
    "ligand_charge",
    "temperature_C",
    "ionic_strength_M",
    "electrolyte",
)
# Expected to vary across metals in the same medium (not part of matching).
MEDIUM_KEY_MAY_DISAGREE = (
    "complex_charge",
    "metal_oxidation_state",
    "IE",
    "metal_valence_e",
    "metal_el",
)

_LAST_MEDIUM_KEY_AUDIT: dict | None = None


def last_medium_key_audit() -> dict | None:
    return _LAST_MEDIUM_KEY_AUDIT


DS1_SPECIES_KEY_COLS = [
    "ligand_smiles",
    "temperature_C",
    "ionic_strength_M",
    "electrolyte",
    "metal_el",
    "metal_oxidation_state",
]

CHEMPROP_DS2_FEATURE_COLUMNS = (
    [f"{c}_x" for c in METAL_PROP_COLS]
    + [f"{c}_y" for c in METAL_PROP_COLS]
    + [
        "metal_oxidation_state_x",
        "metal_oxidation_state_y",
        "complex_charge_x",
        "complex_charge_y",
        "ligand_charge",
        "Temp",
        "Concentration of Medium",
    ]
    + (["solvent_mw", "solvent_density"] if INCLUDE_SOLVENT_PROPERTIES else [])
)


def assert_unique(df: pd.DataFrame, cols: list[str], context: str) -> None:
    if df is None or len(df) == 0:
        raise ValueError(f"{context}: empty table")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context}: missing uniqueness columns {missing}")
    dup = df.duplicated(cols, keep=False)
    if dup.any():
        sample = df.loc[dup, cols].head(20)
        raise ValueError(
            f"{context}: {int(dup.sum())} duplicate rows on {cols}. "
            "Identify the source rather than silently aggregating again.\n"
            f"{sample.to_string(index=False)}"
        )


def make_ds2_folds(ds2: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> dict:
    """Ligand-disjoint GroupKFold splits for DS2 (shared by tabular + graph Stage 3)."""
    ligands = ds2["ligand_smiles"].astype(str).values
    gkf = GroupKFold(n_splits=n_folds)
    folds = {}
    idx = np.arange(len(ds2))
    for i, (trva, te) in enumerate(gkf.split(idx, groups=ligands)):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + i)
        tr, va = next(gss.split(trva, groups=ds2.iloc[trva]["ligand_smiles"].astype(str).values))
        folds[f"fold{i}"] = {
            "train_idx": trva[tr].tolist(),
            "val_idx": trva[va].tolist(),
            "test_idx": te.tolist(),
        }
    return folds


def _copy_metal_props(frame: pd.DataFrame, src_suffix: str, dest_unsuffixed: bool = True) -> None:
    missing = [c for c in METAL_PROP_COLS if f"{c}{src_suffix}" not in frame.columns]
    if missing:
        raise KeyError(
            f"DS2 is missing row-level metal_props{src_suffix} columns {missing}. "
            "Rebuild ds2.csv; do not substitute modal-element values."
        )
    if dest_unsuffixed:
        for c in METAL_PROP_COLS:
            frame[c] = frame[f"{c}{src_suffix}"]


def build_ds2_rdkit_matrix(ds2: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Ligand RDKit + row-level metal_props(x) + DS1 continuous (x) + metal_props(y) + ox_y + q_y.

    No metal one-hot. No electrolyte one-hot. Each intended variable appears once.
    Solvent MW/density live in the DS1 continuous block when enabled.
    """
    assert_runtime_schema("DS2")
    tmp = ds2.copy()
    tmp["metal_el"] = tmp["metal_x"].astype(str)
    tmp["metal_oxidation_state"] = tmp["metal_oxidation_state_x"]
    tmp["complex_charge"] = tmp["complex_charge_x"]
    _copy_metal_props(tmp, "_x", dest_unsuffixed=True)

    X_ligand_x, cols_ligand_x = build_rdkit_matrix(
        tmp,
        ligand_col="ligand_smiles",
        metal_el_col="metal_el",
    )
    expected_base = ds1_rdkit_column_names()
    if cols_ligand_x != expected_base:
        raise RuntimeError("DS2 base feature names drifted from DS1 schema")

    X_y = metal_props_matrix(
        tmp,
        metal_el_col="metal_y",
        ox_col="metal_oxidation_state_y",
        source_cols={c: f"{c}_y" for c in METAL_PROP_COLS},
        reset_report=False,
    )
    extra_cols = ["metal_oxidation_state_y", "complex_charge_y"]
    missing = [c for c in extra_cols if c not in tmp.columns]
    if missing:
        raise KeyError(f"build_ds2_rdkit_matrix: required columns absent {missing}")
    X_extra = tmp[extra_cols].to_numpy(dtype=float)
    X = np.hstack([X_ligand_x, X_y, X_extra])
    cols = cols_ligand_x + [f"{c}_y" for c in METAL_PROP_COLS] + extra_cols
    if X.shape[1] != len(cols):
        raise RuntimeError(f"DS2 width {X.shape[1]} != names {len(cols)}")
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"build_ds2_rdkit_matrix: {n} non-finite values after selection. Imputation is disabled."
        )
    verify_feature_name_layout("DS2", cols)
    return X, cols


def audit_medium_key(agg: pd.DataFrame) -> dict:
    """Rows sharing (ligand, T, I, electrolyte) must agree on solvent and ligand charge."""
    work = agg.copy()
    work["medium_key"] = (
        work["ligand_smiles"].astype(str)
        + "|"
        + work["temperature_C"].map(lambda x: f"{float(x):.2f}")
        + "|"
        + work["ionic_strength_M"].map(lambda x: f"{float(x):.4f}")
        + "|"
        + work["electrolyte"].astype(str)
    )
    disagreements: dict[str, int] = {}
    samples: dict[str, list] = {}
    for col in MEDIUM_KEY_MUST_AGREE:
        if col not in work.columns:
            raise KeyError(f"medium-key audit missing {col}")
        n_bad = 0
        sample = []
        for key, g in work.groupby("medium_key"):
            nun = int(g[col].nunique(dropna=False))
            if nun > 1:
                n_bad += 1
                if len(sample) < 5:
                    sample.append(
                        {
                            "medium_key": key,
                            "nunique": nun,
                            "values": [str(v) for v in g[col].unique()[:6]],
                        }
                    )
        disagreements[col] = n_bad
        samples[col] = sample
    may = {}
    for col in MEDIUM_KEY_MAY_DISAGREE:
        if col in work.columns:
            may[col] = int(
                sum(int(g[col].nunique(dropna=False) > 1) for _, g in work.groupby("medium_key"))
            )
    n_keys = int(work["medium_key"].nunique())
    must_ok = all(v == 0 for v in disagreements.values())
    if not must_ok:
        raise ValueError(
            "DS2 medium key is not sufficient: rows sharing "
            "(ligand, T, I, electrolyte) disagree on solvent MW/density or ligand charge. "
            f"disagreements={disagreements} samples={samples}"
        )
    return {
        "n_medium_keys": n_keys,
        "key": ["ligand_smiles", "temperature_C", "ionic_strength_M", "electrolyte"],
        "must_agree": list(MEDIUM_KEY_MUST_AGREE),
        "n_keys_disagreeing": disagreements,
        "sufficient": True,
        "may_disagree_n_keys": may,
        "note": (
            "solvent_mw, solvent_density, and ligand_charge are a 1:1 consequence of "
            "(ligand, T, I, electrolyte) in this corpus, so they are not added to the "
            "matching key. complex_charge, oxidation state, and IE/valence vary by metal "
            "and are not matching fields."
        ),
    }


def _aggregate_metal_complex(ds1: pd.DataFrame) -> pd.DataFrame:
    mc = ds1[ds1["family"] == "metal_complex"].copy()
    mc = mc[mc["metal_el"].astype(str).str.upper() != "H"].copy()
    if mc.empty:
        raise ValueError("No metal_complex rows left after excluding H for DS2")
    extra_first = [
        c
        for c in (
            "record_id",
            "ligand_charge",
            "complex_charge",
            "complex_smiles",
            "solvent_mw",
            "solvent_density",
            "is_ree",
            *METAL_PROP_COLS,
        )
        if c in mc.columns
    ]
    missing_props = [c for c in METAL_PROP_COLS if c not in mc.columns]
    if missing_props:
        raise KeyError(f"DS1 metal_complex missing row-level metal_props {missing_props}")
    agg = mc.groupby(DS1_SPECIES_KEY_COLS, as_index=False).agg(
        logK=("logK", "median"),
        **{c: (c, "first") for c in extra_first},
        n_replicates=("logK", "size"),
    )
    assert_unique(agg, DS1_SPECIES_KEY_COLS, "DS1 metal_complex after replicate aggregation")
    return agg


def build_ds2(ds1: pd.DataFrame) -> pd.DataFrame:
    """
    selectivity_logK(X,Y,L) = logK(X,L) - logK(Y,L)
    Matched on ligand_smiles + temperature_C + ionic_strength_M + electrolyte.
    Metal–metal pairs only: protonation / H⁺ rows never enter DS2.

    Electrolyte identity is used for experimental matching and grouping but is
    excluded from the tabular predictor.
    """
    global _LAST_MEDIUM_KEY_AUDIT
    agg = _aggregate_metal_complex(ds1)
    medium_audit = audit_medium_key(agg)
    _LAST_MEDIUM_KEY_AUDIT = medium_audit
    agg["medium_key"] = (
        agg["ligand_smiles"].astype(str)
        + "|"
        + agg["temperature_C"].map(lambda x: f"{float(x):.2f}")
        + "|"
        + agg["ionic_strength_M"].map(lambda x: f"{float(x):.4f}")
        + "|"
        + agg["electrolyte"].astype(str)
    )
    rows = []
    for medium_key, g in agg.groupby("medium_key"):
        n_species = g.apply(
            lambda r: species_key(r["metal_el"], r["metal_oxidation_state"]), axis=1
        ).nunique()
        if n_species < 2:
            continue
        metals = g.to_dict("records")
        for i in range(len(metals)):
            for j in range(i + 1, len(metals)):
                a, b = metals[i], metals[j]
                if (a["metal_el"], a["metal_oxidation_state"]) > (
                    b["metal_el"],
                    b["metal_oxidation_state"],
                ):
                    a, b = b, a
                sel = float(a["logK"]) - float(b["logK"])
                msp = canonical_species_pair(
                    a["metal_el"], a["metal_oxidation_state"],
                    b["metal_el"], b["metal_oxidation_state"],
                )
                rec = {
                    "pair_id": f"{a['record_id']}__{b['record_id']}",
                    "ligand_smiles": a["ligand_smiles"],
                    "metal_x": a["metal_el"],
                    "metal_y": b["metal_el"],
                    "metal_oxidation_state_x": a["metal_oxidation_state"],
                    "metal_oxidation_state_y": b["metal_oxidation_state"],
                    "logK_x": a["logK"],
                    "logK_y": b["logK"],
                    "selectivity_logK": sel,
                    "temperature_C": a["temperature_C"],
                    "ionic_strength_M": a["ionic_strength_M"],
                    "electrolyte": a["electrolyte"],
                    "solvent_mw": a.get("solvent_mw", np.nan),
                    "solvent_density": a.get("solvent_density", np.nan),
                    "ligand_charge": a["ligand_charge"],
                    "complex_charge_x": a["complex_charge"],
                    "complex_charge_y": b["complex_charge"],
                    "complex_smiles_x": a["complex_smiles"],
                    "complex_smiles_y": b["complex_smiles"],
                    "record_id_x": a["record_id"],
                    "record_id_y": b["record_id"],
                    "is_ree_pair": (a["metal_el"] in REE_ELEMENTS) and (b["metal_el"] in REE_ELEMENTS),
                    "metal_species_pair": msp,
                    "metal_element_pair": canonical_element_pair(a["metal_el"], b["metal_el"]),
                    "metal_pair": canonical_element_pair(a["metal_el"], b["metal_el"]),
                    "medium_key": medium_key,
                    "metal_props_convention": "row_level_cmi",
                }
                for c in METAL_PROP_COLS:
                    rec[f"{c}_x"] = a[c]
                    rec[f"{c}_y"] = b[c]
                rows.append(rec)
    ds2 = pd.DataFrame(rows)
    ds2 = assert_metal_metal_pairs(ds2)
    assert_unique(
        ds2,
        ["ligand_smiles", "medium_key", "metal_species_pair"],
        "DS2 after pair generation",
    )
    return ds2


def assert_metal_metal_pairs(ds2: pd.DataFrame) -> pd.DataFrame:
    """Drop / reject any metal–H pairs so Stage 3–5 metrics stay metal–metal only."""
    if ds2 is None or len(ds2) == 0:
        return ds2
    bad = pd.Series(False, index=ds2.index)
    if "metal_x" in ds2.columns and "metal_y" in ds2.columns:
        hx = ds2["metal_x"].astype(str).str.upper() == "H"
        hy = ds2["metal_y"].astype(str).str.upper() == "H"
        bad = bad | hx | hy
    for col in ("metal_species_pair", "metal_element_pair", "metal_pair"):
        if col in ds2.columns:
            mp = ds2[col].astype(str).str.upper()
            bad = bad | mp.str.contains(r"(?:^|[:\-])H(?:$|[:\-])", regex=True)
    n_bad = int(bad.sum())
    if n_bad:
        ds2 = ds2.loc[~bad].reset_index(drop=True)
    if len(ds2) == 0:
        raise ValueError("DS2 empty after removing metal–H pairs")
    return ds2


def summarize_ds2(ds2: pd.DataFrame) -> dict:
    group_col = "metal_species_pair" if "metal_species_pair" in ds2.columns else "metal_pair"
    pair_counts = ds2.groupby(group_col)["ligand_smiles"].nunique()
    n_h = int(
        ((ds2["metal_x"].astype(str).str.upper() == "H") | (ds2["metal_y"].astype(str).str.upper() == "H")).sum()
    )
    return {
        "n_pairs_rows": int(len(ds2)),
        "n_unique_ligands": int(ds2["ligand_smiles"].nunique()),
        "n_metal_species_pairs": int(ds2["metal_species_pair"].nunique()) if "metal_species_pair" in ds2.columns else None,
        "n_metal_element_pairs": int(ds2["metal_element_pair"].nunique()) if "metal_element_pair" in ds2.columns else None,
        "n_metal_pairs": int(ds2[group_col].nunique()),
        "fraction_ree_ree_pairs": float(ds2["is_ree_pair"].mean()) if len(ds2) else None,
        "median_candidate_pool_per_metal_pair": float(pair_counts.median()) if len(pair_counts) else None,
        "mean_pairs_per_ligand": float(ds2.groupby("ligand_smiles").size().mean()) if len(ds2) else None,
        "pair_scope": "metal_metal_only",
        "grouping_column": group_col,
        "n_metal_H_pairs": n_h,
        "electrolyte_in_X": False,
        "metal_props_convention": "row_level_cmi",
        "medium_key": ["ligand_smiles", "temperature_C", "ionic_strength_M", "electrolyte"],
        "note": (
            "Electrolyte identity is used for experimental matching and grouping "
            "but is excluded from the tabular predictor. Grouping uses metal_species_pair. "
            "IE and metal_valence_e are row-level CMI values (same convention as Chemprop)."
        ),
    }


def save_ds2(ds2: pd.DataFrame, summary: dict) -> None:
    out = REMAKE_ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    ds2.to_csv(out / "ds2.csv", index=False)
    payload = json.dumps(summary, indent=2)
    canon = REMAKE_ROOT / "results" / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    (canon / "ds2_composition.json").write_text(payload, encoding="utf-8")
    (out / "ds2_composition.json").write_text(payload, encoding="utf-8")


def chemprop_ds2_feature_frame(ds2: pd.DataFrame) -> pd.DataFrame:
    """Additional features for DS2 graph models: metal_props(x‖y) + charges + medium.

    Graph SMILES is the shared ligand. Metal identity lives in the additional-feature
    block. Electrolyte string is not included.
    """
    cols_x = [f"{c}_x" for c in METAL_PROP_COLS]
    cols_y = [f"{c}_y" for c in METAL_PROP_COLS]
    missing = [c for c in cols_x + cols_y if c not in ds2.columns]
    if missing:
        raise KeyError(
            f"chemprop_ds2_feature_frame missing row-level metal_props {missing}. "
            "Rebuild ds2.csv rather than substituting modal-element values."
        )
    mx = metal_props_matrix(
        ds2,
        metal_el_col="metal_x",
        ox_col="metal_oxidation_state_x",
        source_cols={c: f"{c}_x" for c in METAL_PROP_COLS},
    )
    my = metal_props_matrix(
        ds2,
        metal_el_col="metal_y",
        ox_col="metal_oxidation_state_y",
        source_cols={c: f"{c}_y" for c in METAL_PROP_COLS},
    )
    feats = pd.concat(
        [
            pd.DataFrame(mx, columns=cols_x, index=ds2.index),
            pd.DataFrame(my, columns=cols_y, index=ds2.index),
        ],
        axis=1,
    )
    feats["metal_oxidation_state_x"] = pd.to_numeric(ds2["metal_oxidation_state_x"], errors="coerce")
    feats["metal_oxidation_state_y"] = pd.to_numeric(ds2["metal_oxidation_state_y"], errors="coerce")
    feats["complex_charge_x"] = pd.to_numeric(ds2["complex_charge_x"], errors="coerce")
    feats["complex_charge_y"] = pd.to_numeric(ds2["complex_charge_y"], errors="coerce")
    feats["ligand_charge"] = pd.to_numeric(ds2["ligand_charge"], errors="coerce")
    feats["Temp"] = pd.to_numeric(ds2["temperature_C"], errors="coerce")
    feats["Concentration of Medium"] = pd.to_numeric(ds2["ionic_strength_M"], errors="coerce")
    if INCLUDE_SOLVENT_PROPERTIES:
        feats["solvent_mw"] = pd.to_numeric(ds2["solvent_mw"], errors="coerce")
        feats["solvent_density"] = pd.to_numeric(ds2["solvent_density"], errors="coerce")
    feats = feats[CHEMPROP_DS2_FEATURE_COLUMNS]
    return feats


def write_chemprop_ds2_csvs(
    ds2: pd.DataFrame,
    idx,
    out_data: Path,
    out_features: Path,
    *,
    train_feature_medians: pd.Series,
) -> None:
    sub = ds2.iloc[list(idx)].copy()
    data = pd.DataFrame(
        {
            "smiles": sub["ligand_smiles"].astype(str).values,
            "logK": sub["selectivity_logK"].to_numpy(dtype=float),
        }
    )
    feats = chemprop_ds2_feature_frame(sub)
    feats = feats.apply(pd.to_numeric, errors="coerce")
    if feats.isna().any().any():
        bad = [c for c in feats.columns if feats[c].isna().any()]
        n = int(feats.isna().any(axis=1).sum())
        raise ValueError(
            f"DS2 Chemprop: {n} rows have missing additional features {bad}. "
            "Imputation is disabled."
        )
    if list(feats.columns) != list(CHEMPROP_DS2_FEATURE_COLUMNS):
        raise RuntimeError("DS2 Chemprop feature column order drifted from schema")
    out_data.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_data, index=False)
    feats.to_csv(out_features, index=False)


def write_chemprop_ds2_fold_csvs(
    ds2: pd.DataFrame,
    train_idx,
    val_idx,
    test_idx,
    work_dir: Path,
) -> pd.Series:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    train_feats = chemprop_ds2_feature_frame(ds2.iloc[list(train_idx)])
    med = train_feats.apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    med = med.reindex(CHEMPROP_DS2_FEATURE_COLUMNS)
    for name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        write_chemprop_ds2_csvs(
            ds2,
            idx,
            work_dir / f"{name}.csv",
            work_dir / f"{name}_features.csv",
            train_feature_medians=med,
        )
    schema = {
        "columns": list(CHEMPROP_DS2_FEATURE_COLUMNS),
        "n_features": len(CHEMPROP_DS2_FEATURE_COLUMNS),
        "imputation": False,
        "train_feature_medians": {k: (None if pd.isna(v) else float(v)) for k, v in med.items()},
        "smiles_col": "ligand_smiles",
        "electrolyte_in_X": False,
        "metal_props_convention": "row_level_cmi",
    }
    (work_dir / "chemprop_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return med
