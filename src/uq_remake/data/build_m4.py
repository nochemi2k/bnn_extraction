"""Build processed M4 table from CMI official CSVs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from uq_remake import REMAKE_ROOT
from uq_remake.data.feature_schema import CHEMPROP_DS1_FEATURE_COLUMNS
from uq_remake.data.ligand_parse import extract_ligand_and_metal

REE_ELEMENTS = {
    "Sc", "Y",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}

# Exact MW / density → electrolyte name (matching / reporting only; not a model feature)
ELECTROLYTE_TABLE = {
    (round(100.952, 3), round(2.10, 2)): "KNO3",
    (round(121.938, 3), round(2.00, 2)): "NaClO4",
    (round(73.933, 3), round(1.98, 2)): "KCl",
    (round(84.978, 3), round(2.30, 2)): "NaNO3",
    (round(57.959, 3), round(2.17, 2)): "NaCl",
    (round(109.066, 3), round(1.17, 2)): "Me4NCl",
}

VALIDATION_FEATURE_RENAME = {
    "Temp/C": "Temp",
    "Ionic strength /M": "Concentration of Medium",
    "Exact MW of electrolyte (g/mol)": "Exact MW of solvent (g/mol)",
}


def decode_electrolyte(mw: float, density: float) -> str:
    key = (round(float(mw), 3), round(float(density), 2))
    if key in ELECTROLYTE_TABLE:
        return ELECTROLYTE_TABLE[key]
    best, best_d = "UNKNOWN", float("inf")
    for (m, d), name in ELECTROLYTE_TABLE.items():
        dist = abs(m - float(mw)) + abs(d - float(density))
        if dist < best_d:
            best, best_d = name, dist
    return best if best_d < 0.05 else "UNKNOWN"


def _harmonize_feature_columns(feat: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in feat.columns:
        cl = str(c).replace("\ufeff", "")
        if cl in VALIDATION_FEATURE_RENAME:
            rename[c] = VALIDATION_FEATURE_RENAME[cl]
        elif cl.startswith("Density of electrolyte"):
            rename[c] = "Density of solvent g/cm-3"
        elif cl != c:
            rename[c] = cl
    return feat.rename(columns=rename) if rename else feat


def _load_split(raw_dir: Path, split_name: str) -> pd.DataFrame:
    if split_name == "validation":
        inp = pd.read_csv(raw_dir / "validation_input.csv")
        feat = pd.read_csv(raw_dir / "validation_features.csv", encoding="utf-8-sig")
        feat = _harmonize_feature_columns(feat)
        smiles_col = "SMILES" if "SMILES" in inp.columns else inp.columns[0]
        y_col = "Experimental Lg K1" if "Experimental Lg K1" in inp.columns else inp.columns[1]
        inp = inp.rename(columns={smiles_col: "complex_smiles", y_col: "logK"})
    else:
        inp = pd.read_csv(raw_dir / f"{split_name}_input.csv")
        feat = pd.read_csv(raw_dir / f"{split_name}_features.csv")
        feat = _harmonize_feature_columns(feat)
        inp = inp.rename(
            columns={
                "cmplx_smiles_neutral_std": "complex_smiles",
                "Exp LgK1": "logK",
            }
        )
    if len(inp) != len(feat):
        raise ValueError(f"{split_name}: input/features length mismatch {len(inp)} vs {len(feat)}")
    out = pd.concat([inp.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    out["cmi_source"] = {"train": "train", "test": "test", "validation": "ext"}[split_name]
    return out


def _atomic_number_of(el: str) -> float:
    if not el:
        return float("nan")
    try:
        from rdkit import Chem

        return float(Chem.GetPeriodicTable().GetAtomicNumber(str(el)))
    except Exception:
        return float("nan")


def build_m4_table(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (modeling_df, external_df, parse_audit).

    CMI validation_input.csv is an external set: it is not concatenated into
    the modeling corpus and never enters project folds.
    """
    raw_dir = raw_dir or (REMAKE_ROOT / "data" / "raw_m4")
    frames = [_load_split(raw_dir, s) for s in ("train", "test", "validation")]
    df = pd.concat(frames, ignore_index=True)

    parsed = df["complex_smiles"].map(extract_ligand_and_metal)
    parsed_df = pd.DataFrame(list(parsed))
    df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    df["is_ree"] = df["metal_el"].isin(REE_ELEMENTS)

    df["temperature_C"] = pd.to_numeric(df["Temp"], errors="coerce")
    df["ionic_strength_M"] = pd.to_numeric(df["Concentration of Medium"], errors="coerce")
    df["ligand_charge"] = pd.to_numeric(df["Total_ligand charge"], errors="coerce")
    df["complex_charge"] = pd.to_numeric(df["Cmplx_charge"], errors="coerce")
    df["solvent_mw"] = pd.to_numeric(df["Exact MW of solvent (g/mol)"], errors="coerce")
    df["solvent_density"] = pd.to_numeric(df["Density of solvent g/cm-3"], errors="coerce")

    feat_ox = pd.to_numeric(df["Metal_charge"], errors="coerce")
    df["metal_oxidation_state"] = pd.to_numeric(df["metal_oxidation_state"], errors="coerce").fillna(feat_ox)

    df["electrolyte"] = [
        decode_electrolyte(mw, dens) for mw, dens in zip(df["solvent_mw"], df["solvent_density"])
    ]
    df["family"] = np.where(df["is_proton"], "protonation", "metal_complex")

    df["parsed_atomic_number"] = df["metal_el"].map(_atomic_number_of)
    cmi_z = pd.to_numeric(df["Atomic_number"], errors="coerce")
    df["metal_label_disagreement"] = (
        df["parsed_atomic_number"].notna()
        & cmi_z.notna()
        & (df["parsed_atomic_number"] != cmi_z)
    )

    quarantine_mask = (
        df["parse_status"].ne("ok")
        | (df["ligand_smiles"].astype(str).str.len() == 0)
        | df["metal_el"].astype(str).eq("")
        | df["metal_label_disagreement"].fillna(False)
        | df["solvent_mw"].isna()
        | df["solvent_density"].isna()
        | df["temperature_C"].isna()
        | df["ionic_strength_M"].isna()
        | df["electrolyte"].eq("UNKNOWN")
    )
    df["quarantine"] = quarantine_mask

    df["experiment_key"] = (
        df["ligand_smiles"].astype(str)
        + "|"
        + df["metal_el"].astype(str)
        + "|"
        + df["metal_oxidation_state"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "nan")
        + "|"
        + df["temperature_C"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "nan")
        + "|"
        + df["ionic_strength_M"].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "nan")
        + "|"
        + df["electrolyte"].astype(str)
    )
    df["species_key"] = (
        df["ligand_smiles"].astype(str)
        + "|"
        + df["metal_el"].astype(str)
        + "|"
        + df["metal_oxidation_state"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "nan")
    )

    grp = df.groupby("experiment_key")["logK"]
    spread = grp.transform(lambda s: float(s.max() - s.min()) if len(s) else 0.0)
    df["replicate_spread"] = spread.astype(float)
    df["n_replicates_in_key"] = df.groupby("experiment_key")["logK"].transform("size")

    cfg = json.loads((REMAKE_ROOT / "configs" / "shared_dmpnn.json").read_text(encoding="utf-8"))
    agree_max = float(cfg["replicate_agree_max"])
    soft_max = float(cfg["replicate_soft_max"])

    def stratum(v: float, n: int) -> str:
        if n < 2:
            return "singleton"
        if v <= agree_max:
            return "agree"
        if v <= soft_max:
            return "soft_disagreement"
        return "hard_disagreement"

    df["replicate_stratum"] = [
        stratum(float(s), int(n)) for s, n in zip(df["replicate_spread"], df["n_replicates_in_key"])
    ]

    parse_audit = df[
        [
            "cmi_source",
            "complex_smiles",
            "ligand_smiles",
            "metal_el",
            "metal_oxidation_state",
            "Atomic_number",
            "parsed_atomic_number",
            "family",
            "parse_status",
            "n_dot_parts",
            "n_metal_tokens",
            "n_ligand_fragments",
            "n_organic_fragments",
            "n_discarded_organic_fragments",
            "n_failed_parts",
            "failed_parts",
            "metal_tokens",
            "metal_label_disagreement",
            "quarantine",
        ]
    ].copy()

    quarantined = df.loc[df["quarantine"]].copy()
    keep = df.loc[~df["quarantine"]].copy()
    if len(quarantined):
        # Do not let invalid rows become ligand_smiles == "".
        pass

    modeling = keep.loc[keep["cmi_source"] != "ext"].reset_index(drop=True)
    external = keep.loc[keep["cmi_source"] == "ext"].reset_index(drop=True)
    modeling["record_id"] = [f"M4_{i:06d}" for i in range(len(modeling))]
    external["record_id"] = [f"M4EXT_{i:06d}" for i in range(len(external))]
    return modeling, external, parse_audit


def make_project_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction_of_trainligands: float = 0.15,
) -> dict:
    """Ligand-disjoint GroupKFold on the modeling corpus only (no CMI external)."""
    if "cmi_source" in df.columns and (df["cmi_source"] == "ext").any():
        raise ValueError("make_project_folds received CMI external rows; exclude them first.")
    empty = df["ligand_smiles"].astype(str).eq("")
    if empty.any():
        raise ValueError(f"make_project_folds: {int(empty.sum())} empty ligand_smiles")
    ligands = df["ligand_smiles"].astype(str).values
    gkf = GroupKFold(n_splits=n_folds)
    folds = {}
    indices = np.arange(len(df))
    for fold_i, (trainval_idx, test_idx) in enumerate(gkf.split(indices, groups=ligands)):
        tv_ligands = df.iloc[trainval_idx]["ligand_smiles"].astype(str).values
        gss = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction_of_trainligands, random_state=seed + fold_i
        )
        tr_rel, va_rel = next(gss.split(trainval_idx, groups=tv_ligands))
        train_idx = trainval_idx[tr_rel]
        val_idx = trainval_idx[va_rel]
        folds[f"fold{fold_i}"] = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        tr_l = set(df.iloc[train_idx]["ligand_smiles"])
        va_l = set(df.iloc[val_idx]["ligand_smiles"])
        te_l = set(df.iloc[test_idx]["ligand_smiles"])
        assert tr_l.isdisjoint(va_l) and tr_l.isdisjoint(te_l) and va_l.isdisjoint(te_l)
        for name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
            if "cmi_source" in df.columns and (df.iloc[idx]["cmi_source"] == "ext").any():
                raise AssertionError(f"{name} fold{fold_i} contains CMI external rows")
    return folds


def chemprop_feature_columns() -> list[str]:
    return list(CHEMPROP_DS1_FEATURE_COLUMNS)


def chemprop_train_medians(df: pd.DataFrame, train_idx: Iterable[int]) -> pd.Series:
    cols = chemprop_feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Chemprop schema columns missing: {missing}")
    feats = df.iloc[list(train_idx)][cols].apply(pd.to_numeric, errors="coerce")
    med = feats.median(numeric_only=True)
    return med.reindex(cols)


def write_chemprop_csvs(
    df: pd.DataFrame,
    idx: Iterable[int],
    out_data: Path,
    out_features: Path,
    *,
    train_feature_medians: pd.Series,
    smiles_col: str = "complex_smiles",
    y_col: str = "logK",
) -> None:
    """Write Chemprop CSVs using *training* medians for imputation (never val/test medians)."""
    sub = df.iloc[list(idx)].copy()
    cols = chemprop_feature_columns()
    missing = [c for c in cols if c not in sub.columns]
    if missing:
        raise KeyError(f"Chemprop schema columns missing: {missing}")
    data = pd.DataFrame({"smiles": sub[smiles_col].values, "logK": sub[y_col].values})
    feats = sub[cols].apply(pd.to_numeric, errors="coerce")
    if feats.isna().any().any():
        bad = [c for c in cols if feats[c].isna().any()]
        n = int(feats.isna().any(axis=1).sum())
        raise ValueError(
            f"Chemprop DS1: {n} rows have missing additional features {bad}. "
            "Imputation is disabled."
        )
    if list(feats.columns) != cols:
        raise RuntimeError("Chemprop feature column order drifted from schema")
    out_data.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_data, index=False)
    feats.to_csv(out_features, index=False)


def write_chemprop_fold_csvs(
    df: pd.DataFrame,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    test_idx: Iterable[int],
    work_dir: Path,
    extra: dict[str, Iterable[int]] | None = None,
    smiles_col: str = "complex_smiles",
    y_col: str = "logK",
) -> pd.Series:
    """Write train/val/test Chemprop files; imputation medians from train only."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    med = chemprop_train_medians(df, train_idx)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    if extra:
        splits.update(extra)
    for name, idx in splits.items():
        write_chemprop_csvs(
            df,
            idx,
            work_dir / f"{name}.csv",
            work_dir / f"{name}_features.csv",
            train_feature_medians=med,
            smiles_col=smiles_col,
            y_col=y_col,
        )
    schema = {
        "columns": chemprop_feature_columns(),
        "n_features": len(chemprop_feature_columns()),
        "imputation": False,
        "train_feature_medians": {k: (None if pd.isna(v) else float(v)) for k, v in med.items()},
        "smiles_col": smiles_col,
        "metal_props_convention": "row_level_cmi",
        "note": (
            "Electrolyte identity is not a Chemprop additional feature. "
            "Solvent MW/density are the CMI continuous medium fields. "
            "IE and metal_valence_e are row-level CMI feature columns."
        ),
    }
    (work_dir / "chemprop_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return med


def save_processed(
    modeling: pd.DataFrame,
    folds: dict,
    external: pd.DataFrame | None = None,
    parse_audit: pd.DataFrame | None = None,
    quarantined: pd.DataFrame | None = None,
) -> None:
    out_dir = REMAKE_ROOT / "data" / "processed"
    split_dir = REMAKE_ROOT / "data" / "splits"
    audit_dir = out_dir / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(out_dir / "m4_processed.csv", index=False)
    try:
        modeling.to_parquet(out_dir / "m4_processed.parquet", index=False)
    except Exception:
        pass
    (split_dir / "project_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    cmi = {
        "train_idx": modeling.index[modeling["cmi_source"] == "train"].tolist(),
        "test_idx": modeling.index[modeling["cmi_source"] == "test"].tolist(),
        "ext_file": "data/processed/m4_external.csv",
        "ext_in_project_folds": False,
        "note": "CMI validation is an external set and is not in project_folds or DS1/DS2.",
    }
    (split_dir / "cmi_official_split.json").write_text(json.dumps(cmi, indent=2), encoding="utf-8")
    if external is not None:
        external.to_csv(out_dir / "m4_external.csv", index=False)
    if parse_audit is not None:
        parse_audit.to_csv(audit_dir / "ligand_parse_audit.csv", index=False)
    if quarantined is not None and len(quarantined):
        quarantined.to_csv(audit_dir / "ligand_parse_quarantine.csv", index=False)
    else:
        pd.DataFrame().to_csv(audit_dir / "ligand_parse_quarantine.csv", index=False)
