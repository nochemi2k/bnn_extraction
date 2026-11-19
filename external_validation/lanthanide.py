"""Prepare and train the fixed lanthanide log-separation-factor model.

Historical training/validation rows from Liu et al. are paired only within exact
ligand, publication, and condition cells. The target is logD_x - logD_y for the
canonical increasing-atomic-number orientation, reported as a log separation
factor. The prospective sheet, acquisition experiments, and plotting are outside
this reproduction package.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import FOLDS, REPOSITORY_ROOT, write_json

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # before sklearn on Windows  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402

from uq_remake.data.rdkit_features import preprocess_fold  # noqa: E402
from uq_remake.models.uq_tabular import aggregate_members, predict_hetero_mc, train_hetero_mlp  # noqa: E402
import uq_remake.models.uq_tabular as uq_mod  # noqa: E402
from uq_remake.training.protocol import load_active_grid_cfg, load_shared_cfg, model_seed, seed_everything  # noqa: E402

ECFP_COLS = [f"ECFP-{i}" for i in range(1, 2049)]
METAL_COL = "Atomic Number_metal "
TARGET = "log_D"
CONDITION_COLS = [
    "ligand c.c./mM", "volume ratio of solvent a", "volume ratio of solvent b",
    "Molar mass(a)", "density(a)", "boiling point(a)", "melting point(a)",
    "Dipole moment(a)", "Solubility in water(a)", "log P(a)",
    "Molar mass(b)", "density(b)", "boiling point(b)", "melting point(b)",
    "Dipole moment(b)", "Solubility in water(b)", "log P(b)",
    "aicd dipole", "acid c.c./M", "temperature", "Ln c.c./mM",
]
DILUENT_COLS = CONDITION_COLS[1:19]
ELEMENTS = {
    57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu",
    64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
}


def ligand_hashes(frame: pd.DataFrame) -> pd.Series:
    """Use ECFP and RDKit fields because ECFP-only collisions occur."""
    bits = frame[ECFP_COLS].to_numpy()
    if not np.isin(bits, [0, 1]).all():
        raise ValueError("ECFP block contains non-binary values")
    rdkit = frame.iloc[:, 2048:2256].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if rdkit.shape[1] != 208 or not np.isfinite(rdkit).all():
        raise ValueError("Invalid released RDKit ligand block")
    hashes = []
    for bit_row, descriptor_row in zip(bits, rdkit):
        digest = hashlib.sha256(np.packbits(bit_row.astype(np.uint8)).tobytes())
        digest.update(np.ascontiguousarray(descriptor_row).tobytes())
        hashes.append(digest.hexdigest()[:16])
    return pd.Series(hashes, index=frame.index)


def load_historical(source_file: Path) -> pd.DataFrame:
    frames = []
    for sheet in ("training set", "validation set"):
        part = pd.read_excel(source_file, sheet_name=sheet)
        part["source_partition"] = sheet
        part["source_row"] = np.arange(2, len(part) + 2)
        part["source_sheet"] = sheet
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    required = ECFP_COLS + CONDITION_COLS + [METAL_COL, TARGET, "reference"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df["ligand_id"] = ligand_hashes(df)
    df["metal_z"] = pd.to_numeric(df[METAL_COL], errors="raise").astype(int)
    unknown = sorted(set(df["metal_z"]) - set(ELEMENTS))
    if unknown:
        raise ValueError(f"Unexpected lanthanide atomic numbers: {unknown}")
    df["metal"] = df["metal_z"].map(ELEMENTS)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    df["source_record_id"] = df["source_sheet"] + ":" + df["source_row"].astype(str)
    return df


def pair_historical(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["ligand_id", "reference"] + CONDITION_COLS
    for condition_number, (_, group) in enumerate(df.groupby(keys, dropna=False, sort=False)):
        metals = group.groupby(["metal_z", "metal"], as_index=False).agg(
            observed_log_d=(TARGET, "median"), n_source_rows=(TARGET, "size"),
            source_record_ids=("source_record_id", lambda values: ";".join(values.astype(str))),
        )
        if len(metals) < 2:
            continue
        condition = group.iloc[0]
        for left, right in itertools.combinations(metals.itertuples(index=False), 2):
            x, y = (left, right) if left.metal_z < right.metal_z else (right, left)
            rows.append({
                "pair_id": f"LNPAIR-{len(rows):06d}", "condition_id": f"LNCOND-{condition_number:05d}",
                "ligand_id": condition["ligand_id"], "reference": condition["reference"],
                "metal_x": x.metal, "metal_y": y.metal, "metal_z_x": x.metal_z, "metal_z_y": y.metal_z,
                "metal_pair": f"{x.metal}_{y.metal}", "log_d_x": x.observed_log_d, "log_d_y": y.observed_log_d,
                "y_true": x.observed_log_d - y.observed_log_d,
                "source_record_ids_x": x.source_record_ids, "source_record_ids_y": y.source_record_ids,
                **{column: condition[column] for column in CONDITION_COLS},
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No matched-condition pairs were formed")
    return pairs


def feature_blocks(source: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    rdkit_cols = source.columns[2048:2256].tolist()
    metal_cols = source.columns[2277:2291].tolist()
    if len(rdkit_cols) != 208 or len(CONDITION_COLS) != 21 or len(metal_cols) != 14:
        raise ValueError("Workbook feature block widths changed")
    if metal_cols[0] != METAL_COL or source.columns[2291] != TARGET:
        raise ValueError("Workbook feature layout changed")
    return rdkit_cols, metal_cols, ECFP_COLS + rdkit_cols + CONDITION_COLS


def pair_feature_matrix(pairs: pd.DataFrame, source: pd.DataFrame, source_feature_cols: list[str], metal_cols: list[str]) -> np.ndarray:
    features = source[source_feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    metals = source[metal_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    combined = np.hstack([features, metals])
    lookup = {record_id: i for i, record_id in enumerate(source["source_record_id"].astype(str))}
    if len(lookup) != len(source):
        raise ValueError("Duplicate source_record_id")
    rows = []
    base_dim = len(source_feature_cols)
    for pair in pairs.itertuples(index=False):
        vectors = []
        for ids_text in (pair.source_record_ids_x, pair.source_record_ids_y):
            ids = str(ids_text).split(";")
            indices = [lookup[record_id] for record_id in ids]
            reference = combined[indices[0]]
            for index in indices[1:]:
                if not np.array_equal(reference, combined[index]):
                    raise ValueError(f"Feature mismatch within exact replicate group {ids_text}")
            vectors.append(reference)
        x, y = vectors
        rows.append(np.concatenate([x[:base_dim], x[base_dim:], y[base_dim:]]))
    matrix = np.vstack(rows).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite released features; imputation is disabled")
    return matrix


def tanimoto_nn_distances(raw_x: np.ndarray, pairs: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train_ligands = sorted(pairs.iloc[train_idx]["ligand_id"].unique())
    test_ligands = sorted(pairs.iloc[test_idx]["ligand_id"].unique())
    first_index = pairs.reset_index(drop=True).groupby("ligand_id").first()["row_index"]
    bits = {ligand: np.asarray(raw_x[int(first_index.loc[ligand]), :2048], dtype=np.int16) for ligand in set(train_ligands + test_ligands)}
    train_matrix = np.stack([bits[ligand] for ligand in train_ligands])
    train_sums = train_matrix.sum(axis=1)
    distances = {}
    for ligand in test_ligands:
        query = bits[ligand]
        intersection = train_matrix @ query
        union = train_sums + query.sum() - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        distances[ligand] = 1.0 - float(similarity.max())
    return pairs.iloc[test_idx]["ligand_id"].map(distances).to_numpy(dtype=float)


def mahalanobis_centroid_distances(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    # Exclude ECFP; retain joint ligand-RDKit, condition, and both-metal descriptors.
    train = np.asarray(train[:, 2048:], dtype=np.float64)
    test = np.asarray(test[:, 2048:], dtype=np.float64)
    estimator = LedoitWolf().fit(train)
    center = train.mean(axis=0)
    delta = test - center
    transformed = delta @ estimator.precision_
    squared = np.einsum("ij,ij->i", transformed, delta, optimize=True)
    return np.sqrt(np.maximum(squared, 0.0))


def prepare(source_file: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    historical = load_historical(source_file)
    pairs = pair_historical(historical)
    _, metal_cols, source_feature_cols = feature_blocks(historical)
    X = pair_feature_matrix(pairs, historical, source_feature_cols, metal_cols)
    feature_names = source_feature_cols + [f"{c}_x" for c in metal_cols] + [f"{c}_y" for c in metal_cols]
    if X.shape != (len(pairs), 2305) or X.shape[1] != len(feature_names):
        raise ValueError("Pair feature width mismatch")
    pairs.to_csv(work_dir / "pairs.csv.gz", index=False, compression="gzip")
    write_json(work_dir / "feature_names.json", {"feature_names": feature_names, "n_features": len(feature_names)})

    groups = pairs["ligand_id"].astype(str).to_numpy()
    dummy = np.zeros((len(pairs), 1), dtype=np.float32)
    y = pairs["y_true"].to_numpy(dtype=np.float32)
    pairs_with_index = pairs.reset_index(drop=True).copy()
    pairs_with_index["row_index"] = np.arange(len(pairs))
    splits = {}
    audit = []
    all_test: list[int] = []
    for fold_number, (trainval_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(dummy, groups=groups)):
        fold = f"fold{fold_number}"
        relative = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + fold_number)
        train_relative, val_relative = next(relative.split(dummy[trainval_idx], groups=groups[trainval_idx]))
        train_idx, val_idx = trainval_idx[train_relative], trainval_idx[val_relative]
        for left, right in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)):
            if set(groups[left]) & set(groups[right]):
                raise AssertionError(f"{fold}: ligand overlap")
        Xtr, Xva, Xte, prep = preprocess_fold(X[train_idx], X[val_idx], X[test_idx])
        fold_dir = work_dir / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "arrays.npz", X_train=Xtr.astype(np.float32), X_val=Xva.astype(np.float32),
            X_test=Xte.astype(np.float32), y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
        )
        splits[fold] = {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "test_idx": test_idx.tolist()}
        all_test.extend(test_idx.tolist())
        scores = pairs.iloc[test_idx][["pair_id"]].reset_index(drop=True)
        scores["tanimoto_nn_distance"] = tanimoto_nn_distances(X, pairs_with_index, train_idx, test_idx)
        scores["mahalanobis_feature_centroid"] = mahalanobis_centroid_distances(Xtr, Xte)
        scores.to_csv(fold_dir / "test_scores.csv", index=False)
        write_json(fold_dir / "manifest.json", {
            "fold": fold, "feature_names": feature_names, "feature_dimension": len(feature_names),
            "preprocessing": "train-only StandardScaler; no imputation; no PCA",
            "scaler_mean": prep["scaler_mean"], "scaler_scale": prep["scaler_scale"],
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "n_train_ligands": len(set(groups[train_idx])), "n_val_ligands": len(set(groups[val_idx])),
            "n_test_ligands": len(set(groups[test_idx])),
            "metal_orientation": "increasing atomic number; log separation factor = logD_x - logD_y",
            "tanimoto": "1 minus maximum ligand-ECFP Tanimoto similarity to any outer-training ligand",
            "mahalanobis": "training-centroid Ledoit-Wolf distance in standardized joint ligand-RDKit + 21 condition + both-metal descriptor block; ECFP excluded",
        })
        audit.append({
            "fold_id": fold, "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "n_train_ligands": len(set(groups[train_idx])), "n_val_ligands": len(set(groups[val_idx])),
            "n_test_ligands": len(set(groups[test_idx])),
        })
        print(f"[prepared] {fold} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    if sorted(all_test) != list(range(len(pairs))):
        raise AssertionError("Outer test folds do not cover every pair exactly once")
    write_json(work_dir / "splits.json", splits)
    pd.DataFrame(audit).to_csv(work_dir / "split_audit.csv", index=False)
    write_json(work_dir / "preparation_audit.json", {
        "n_historical_pairs": len(pairs), "n_historical_ligands_with_pairs": pairs["ligand_id"].nunique(),
        "feature_dimension": X.shape[1],
        "features": "released 2048 ECFP + 208 RDKit + 21 conditions + 14 metal descriptors per oriented metal",
        "excluded_from_features": ["log_D", "reference", "source", "prospective information"],
        "target": "log separation factor = logD_x - logD_y, increasing atomic-number orientation",
        "split": "five ligand-full-feature-disjoint outer folds; 20% outer-training ligands for validation",
        "matching": "exact ligand ECFP+RDKit block + publication source + all 21 condition fields; median exact metal replicates",
    })


def signature_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite condition signature")
    return pd.Series([tuple(row) for row in values], index=frame.index, dtype=object)


def context_support(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    train, test = train.copy(), test.copy()
    train["signature"] = signature_series(train, columns)
    test["signature"] = signature_series(test, columns)
    unique_conditions = train.drop_duplicates("condition_id")
    ligand_support = train.groupby("signature")["ligand_id"].nunique()
    cell_support = unique_conditions.groupby("signature")["condition_id"].nunique()
    output = test[["pair_id"]].copy()
    output[f"{prefix}_seen_in_training"] = test["signature"].isin(set(train["signature"])).to_numpy()
    output[f"{prefix}_training_ligands"] = test["signature"].map(ligand_support).fillna(0).astype(int).to_numpy()
    output[f"{prefix}_training_condition_cells"] = test["signature"].map(cell_support).fillna(0).astype(int).to_numpy()
    return output


def support_bin(series: pd.Series) -> pd.Categorical:
    return pd.cut(series.astype(float), bins=[-0.5, 0.5, 4.5, 19.5, 99.5, np.inf], labels=["0", "1-4", "5-19", "20-99", ">=100"], ordered=True)


def add_analysis_fields(base: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    base = base.merge(scores, on="pair_id", how="left", validate="one_to_one")
    global_mean = float(train["y_true"].mean())
    pair_means = train.groupby("metal_pair")["y_true"].mean()
    base["dummy_pred"] = base["metal_pair"].map(pair_means).fillna(global_mean)
    base["metal_pair_seen_in_training"] = base["metal_pair"].isin(set(train["metal_pair"]))
    support = train.groupby("metal_pair")["ligand_id"].nunique()
    base["metal_pair_training_ligands"] = base["metal_pair"].map(support).fillna(0).astype(int)
    base["metal_pair_training_support_bin"] = support_bin(base["metal_pair_training_ligands"])
    condition = context_support(train, test, CONDITION_COLS, "condition")
    diluent = context_support(train, test, DILUENT_COLS, "diluent")
    base = base.merge(condition.merge(diluent, on="pair_id", validate="one_to_one"), on="pair_id", how="left", validate="one_to_one")
    base["condition_familiarity"] = np.where(base["condition_seen_in_training"], "seen_exactly", "unseen_exactly")
    base["diluent_familiarity"] = np.where(base["diluent_seen_in_training"], "seen_exactly", "unseen_exactly")
    base["diluent_training_support_bin"] = support_bin(base["diluent_training_ligands"])
    return base


def load_member(path: Path, expected: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    if frame["pair_id"].astype(str).tolist() != expected["pair_id"].astype(str).tolist():
        raise ValueError(f"{path}: pair alignment failure")
    return {
        "mu": frame["y_pred"].to_numpy(dtype=float),
        "var_aleatoric": frame["var_aleatoric"].to_numpy(dtype=float),
        "var_epistemic_within": frame["var_epistemic_within"].to_numpy(dtype=float),
    }


def train(work_dir: Path, output_dir: Path, retrain: bool = False) -> None:
    from weight_io import save_weight_checkpoint

    cfg, grid = load_shared_cfg(), load_active_grid_cfg()
    uq_mod.LOG_VAR_CLAMP = tuple(grid["log_var_clamp"])
    pairs = pd.read_csv(work_dir / "pairs.csv.gz")
    splits = json.loads((work_dir / "splits.json").read_text(encoding="utf-8"))
    feature_names = json.loads((work_dir / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_json(output_dir / "training_spec.json", {
        "model": "fixed five-member heteroscedastic MLP ensemble with MC dropout",
        "members_per_fold": 5, "folds": FOLDS, "mc_test_passes": grid["mc_dropout_passes"],
        "mc_validation_passes": grid["mc_validation_passes"], "device": device,
        "weights_saved": True,
        "weight_format": "weight_io v1; per-member CPU state_dict with feature schema and fold scaler",
        "hyperparameter_search": False,
    })
    metadata = ["pair_id", "condition_id", "ligand_id", "metal_pair", "y_true"]
    for fold in FOLDS:
        fold_number = int(fold.removeprefix("fold"))
        arrays = np.load(work_dir / fold / "arrays.npz")
        manifest = json.loads((work_dir / fold / "manifest.json").read_text(encoding="utf-8"))
        train_idx = np.asarray(splits[fold]["train_idx"], dtype=int)
        test_idx = np.asarray(splits[fold]["test_idx"], dtype=int)
        train_pairs, test_pairs = pairs.iloc[train_idx].copy(), pairs.iloc[test_idx].copy()
        base = test_pairs[metadata].reset_index(drop=True)
        base.insert(0, "cohort_index", test_idx)
        base.insert(0, "fold_id", fold)
        base = add_analysis_fields(base, train_pairs, test_pairs, pd.read_csv(work_dir / fold / "test_scores.csv"))
        if not np.allclose(arrays["y_test"], base["y_true"].to_numpy(dtype=np.float32), rtol=0, atol=0):
            raise ValueError(f"{fold}: target alignment failure")
        member_outputs = []
        for member in range(5):
            member_path = output_dir / "predictions" / fold / f"member_{member}.csv"
            meta_path = output_dir / "predictions" / fold / f"member_{member}.json"
            weight_path = output_dir / "weights" / fold / f"member_{member}.pt"
            if member_path.exists() and weight_path.exists() and not retrain:
                member_outputs.append(load_member(member_path, base))
                print(f"[reused] {fold} member={member}", flush=True)
                continue
            seed = model_seed(fold_number, member, grid["model_offsets"]["HETERO"], grid["base_seed"], dataset="DS2")
            seed_everything(seed)
            started = time.perf_counter()
            model, meta = train_hetero_mlp(
                arrays["X_train"], arrays["y_train"], arrays["X_val"], arrays["y_val"], cfg,
                device=device, seed=seed, n_mc_val=int(grid["mc_validation_passes"]),
            )
            save_weight_checkpoint(
                weight_path, model=model, fold=fold, member=member, seed=seed,
                target="log separation factor = logD_x - logD_y", feature_names=feature_names,
                hidden_dims=list(cfg["mlp_hidden_dims"]), log_var_clamp=grid["log_var_clamp"],
                scaler_mean=manifest["scaler_mean"], scaler_scale=manifest["scaler_scale"],
                training_meta=meta,
            )
            uq = predict_hetero_mc(model, arrays["X_test"], n_mc=int(grid["mc_dropout_passes"]), device=device, seed=seed + 11)
            member_path.parent.mkdir(parents=True, exist_ok=True)
            frame = base[["pair_id", "y_true"]].copy()
            frame["y_pred"] = uq["mu"]
            for key in ("var_aleatoric", "var_epistemic_within", "var_total", "sigma_total"):
                frame[key] = uq[key]
            frame.to_csv(member_path, index=False)
            write_json(meta_path, {"fold": fold, "member": member, "seed": seed, "train_seconds_wall": time.perf_counter() - started, "training_meta": meta})
            member_outputs.append(uq)
            print(f"[trained+saved] {fold} member={member} seconds={time.perf_counter() - started:.1f}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        aggregated = aggregate_members(member_outputs)
        ensemble = base.copy()
        ensemble["y_pred"] = aggregated["mu"]
        for key in ("var_aleatoric", "var_epistemic_within", "var_epistemic_between", "var_epistemic_total", "var_total", "sigma_aleatoric", "sigma_epistemic_total", "sigma_total"):
            ensemble[key] = aggregated[key]
        if not np.isfinite(ensemble[["y_pred", "sigma_total", "dummy_pred"]]).all().all():
            raise ValueError(f"{fold}: non-finite ensemble or dummy output")
        ensemble.to_csv(output_dir / "predictions" / fold / "ensemble.csv", index=False)
        print(f"[aggregated] {fold}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "all"])
    parser.add_argument("--source-file", type=Path, default=REPOSITORY_ROOT / "data/external/lanthanide_logd/PMC9241157_supplementary/au2c00122_si_002.xlsx")
    parser.add_argument("--work-dir", type=Path, default=REPOSITORY_ROOT / "external_validation_work/lanthanide")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results/external_validation/lanthanide")
    parser.add_argument("--retrain", action="store_true", help="retrain even when predictions and checkpoints already exist")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args.source_file, args.work_dir)
    if args.stage in ("train", "all"):
        train(args.work_dir, args.output_dir, retrain=args.retrain)


if __name__ == "__main__":
    main()
