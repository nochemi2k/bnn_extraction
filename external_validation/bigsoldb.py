"""Prepare and train the fixed BigSolDB v2.2 external-validation model.

The protocol is deliberately fixed: five solute-disjoint outer folds, a grouped
validation split, five heteroscedastic MLP members, and 30 MC-dropout passes.
No hyperparameter search, plotting, acquisition analysis, or manuscript code is
included. Existing member predictions are reused only when their checkpoints
exist and their row identities align.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import FOLDS, REPOSITORY_ROOT, write_json

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # before sklearn/RDKit on Windows  # noqa: E402
from rdkit import Chem, DataStructs, rdBase  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402

from uq_remake.data.feature_schema import PINNED_RDKIT_VERSIONS, rdkit_selected_descriptor_names  # noqa: E402
from uq_remake.data.rdkit_features import preprocess_fold, rdkit_descriptor_vector  # noqa: E402
from uq_remake.models.uq_tabular import aggregate_members, predict_hetero_mc, train_hetero_mlp  # noqa: E402
import uq_remake.models.uq_tabular as uq_mod  # noqa: E402
from uq_remake.training.protocol import load_active_grid_cfg, load_shared_cfg, model_seed, seed_everything  # noqa: E402

TARGET = "LogS(mol/L)"
GROUP_KEYS = ["solute_smiles", "solvent_smiles", "Temperature_K", "Source", "solid_form_label"]
METADATA_COLUMNS = [
    "Solvent", "Compound_Name", "CAS", "PubChem_CID", "FDA_Approved",
    "solubility_type", "method", "equilibration_h", "settling_h", "agitation_type",
    "agitation_rpm", "temperature_control", "pressure_kPa", "replicates",
    "solute_purity", "purity_method", "solute_supplier", "solid_phase_control",
    "phase_change_observed", "solid_form", "melting_point_K", "melting_point_method",
    "decomposition_temperature_K", "fusion_enthalpy_kJ_mol", "fusion_enthalpy_source",
]


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    return Chem.MolToSmiles(mol, canonical=True)


def descriptor_cache(smiles_values: list[str], role: str) -> tuple[dict, dict, list[dict]]:
    vectors: dict[str, np.ndarray] = {}
    canonical: dict[str, str] = {}
    invalid: list[dict] = []
    unique_values = sorted(set(smiles_values))
    for number, smiles in enumerate(unique_values, start=1):
        try:
            canonical[smiles] = canonical_smiles(smiles)
            vectors[smiles] = rdkit_descriptor_vector(smiles)
        except Exception as exc:  # invalid rows are logged and excluded, never imputed
            invalid.append({"role": role, "smiles": smiles, "error": str(exc)})
        if number % 250 == 0:
            print(f"[descriptors] {role} {number}/{len(unique_values)}", flush=True)
    return vectors, canonical, invalid


def load_and_filter(source_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_csv(source_dir / "BigSolDBv2.2.csv")
    methods = pd.read_csv(source_dir / "BigSolDBv2.2_methods.csv")
    forms = pd.read_csv(source_dir / "BigSolDBv2.2_solid_forms.csv")
    raw["source_row_id"] = np.arange(2, len(raw) + 2)
    for frame, column in ((raw, "Source"), (methods, "doi"), (forms, "doi")):
        frame[column] = frame[column].astype(str).str.strip().str.lower()
    if methods["doi"].duplicated().any():
        raise ValueError("Non-unique DOI in methods table")
    form_keys = ["doi", "SMILES_Solute", "solid_form_label"]
    if forms.duplicated(form_keys).any():
        raise ValueError("Non-unique source/solute/form key in solid-form table")
    df = raw.merge(methods, how="left", left_on="Source", right_on="doi", validate="many_to_one")
    df = df.merge(
        forms, how="left", left_on=["Source", "SMILES_Solute", "solid_form_label"],
        right_on=form_keys, validate="many_to_one", suffixes=("", "_form"),
    )
    if len(df) != len(raw) or df["method"].isna().any() or df["phase_change_observed"].isna().any():
        raise ValueError("Incomplete protocol join")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["Temperature_K"] = pd.to_numeric(df["Temperature_K"], errors="coerce")
    counts = {"raw_rows": int(len(df))}
    df = df.loc[np.isfinite(df[TARGET]) & np.isfinite(df["Temperature_K"])].copy()
    counts["after_finite_target_temperature"] = int(len(df))
    known_non_equilibrium = ~df["solubility_type"].fillna("unknown").isin(["equilibrium", "unknown"])
    df = df.loc[~known_non_equilibrium].copy()
    counts["after_non_equilibrium_exclusion"] = int(len(df))
    acceptable_phase = {"none", "not_checked", "unknown", "not_reported"}
    observed_phase_change = ~df["phase_change_observed"].fillna("not_checked").isin(acceptable_phase)
    df = df.loc[~observed_phase_change].copy()
    counts["after_observed_phase_change_exclusion"] = int(len(df))

    solute_vectors, solute_canonical, invalid_solutes = descriptor_cache(df["SMILES_Solute"].astype(str).tolist(), "solute")
    solvent_vectors, solvent_canonical, invalid_solvents = descriptor_cache(df["SMILES_Solvent"].astype(str).tolist(), "solvent")
    invalid = pd.DataFrame(invalid_solutes + invalid_solvents, columns=["role", "smiles", "error"])
    valid_solute = df["SMILES_Solute"].isin(solute_vectors)
    valid_solvent = df["SMILES_Solvent"].isin(solvent_vectors)
    counts["excluded_invalid_solute_rows"] = int((~valid_solute).sum())
    counts["excluded_invalid_solvent_rows"] = int((~valid_solvent).sum())
    df = df.loc[valid_solute & valid_solvent].copy()
    counts["after_valid_descriptors"] = int(len(df))
    if df.empty:
        raise ValueError("No valid BigSolDB rows remain")

    for role, vectors, mapping in (("solute", solute_vectors, solute_canonical), ("solvent", solvent_vectors, solvent_canonical)):
        canonical_vectors: dict[str, np.ndarray] = {}
        for original, vector in vectors.items():
            canonical_vectors.setdefault(mapping[original], vector)
    df["solute_smiles"] = df["SMILES_Solute"].map(solute_canonical)
    df["solvent_smiles"] = df["SMILES_Solvent"].map(solvent_canonical)

    aggregations = {TARGET: ("y_true", "median"), "source_row_id": ("n_source_rows", "size")}
    for column in METADATA_COLUMNS:
        if column in df.columns:
            aggregations[column] = (column, "first")
    grouped = df.groupby(GROUP_KEYS, dropna=False, sort=False).agg(**{
        output: (source, function) for source, (output, function) in aggregations.items()
    }).reset_index()
    source_ids = df.groupby(GROUP_KEYS, dropna=False, sort=False)["source_row_id"].agg(
        lambda values: ";".join(str(int(value)) for value in values)
    ).rename("source_row_ids").reset_index()
    cohort = grouped.merge(source_ids, on=GROUP_KEYS, how="left", validate="one_to_one")
    cohort.insert(0, "record_id", [f"BIG-{i:07d}" for i in range(len(cohort))])
    counts["after_exact_replicate_median"] = int(len(cohort))
    counts["exact_replicate_rows_collapsed"] = int(counts["after_valid_descriptors"] - len(cohort))
    counts["solutes"] = int(cohort["solute_smiles"].nunique())
    counts["solvents"] = int(cohort["solvent_smiles"].nunique())
    counts["sources"] = int(cohort["Source"].nunique())
    if not np.isfinite(cohort["y_true"]).all():
        raise ValueError("Non-finite aggregated target")
    return cohort, counts, invalid


def build_features(cohort: pd.DataFrame) -> tuple[dict, list[str], np.ndarray]:
    names = rdkit_selected_descriptor_names()
    solutes = sorted(cohort["solute_smiles"].unique())
    solvents = sorted(cohort["solvent_smiles"].unique())
    solute_matrix = np.vstack([rdkit_descriptor_vector(s) for s in solutes]).astype(np.float32)
    solvent_matrix = np.vstack([rdkit_descriptor_vector(s) for s in solvents]).astype(np.float32)
    cache = {
        "solute_smiles": np.asarray(solutes), "solvent_smiles": np.asarray(solvents),
        "solute_descriptors": solute_matrix, "solvent_descriptors": solvent_matrix,
    }
    solute_index = {s: i for i, s in enumerate(solutes)}
    solvent_index = {s: i for i, s in enumerate(solvents)}
    X = np.hstack([
        solute_matrix[cohort["solute_smiles"].map(solute_index).to_numpy(dtype=int)],
        solvent_matrix[cohort["solvent_smiles"].map(solvent_index).to_numpy(dtype=int)],
        cohort[["Temperature_K"]].to_numpy(dtype=np.float32),
    ])
    feature_names = [f"solute__{n}" for n in names] + [f"solvent__{n}" for n in names] + ["temperature_K"]
    if X.shape != (len(cohort), len(feature_names)) or not np.isfinite(X).all():
        raise ValueError("Invalid full feature matrix")
    return cache, feature_names, X


def morgan_distances(cohort: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_solutes = sorted(cohort.iloc[train_idx]["solute_smiles"].unique())
    test_solutes = sorted(cohort.iloc[test_idx]["solute_smiles"].unique())
    train_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_solutes]
    distances = {}
    for smiles in test_solutes:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        distances[smiles] = 1.0 - float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return cohort.iloc[test_idx]["solute_smiles"].map(distances).to_numpy(dtype=float)


def mahalanobis_distances(X_train: np.ndarray, X_test: np.ndarray, descriptor_dim: int) -> np.ndarray:
    # Temperature is deliberately excluded from this joint descriptor distance.
    train = np.asarray(X_train[:, :descriptor_dim], dtype=np.float64)
    test = np.asarray(X_test[:, :descriptor_dim], dtype=np.float64)
    estimator = LedoitWolf().fit(train)
    center = train.mean(axis=0)
    output = np.empty(len(test), dtype=float)
    for start in range(0, len(test), 4000):
        delta = test[start:start + 4000] - center
        transformed = delta @ estimator.precision_
        squared = np.einsum("ij,ij->i", transformed, delta, optimize=True)
        output[start:start + len(delta)] = np.sqrt(np.maximum(squared, 0.0))
    return output


def prepare(source_dir: Path, work_dir: Path) -> None:
    if rdBase.rdkitVersion not in PINNED_RDKIT_VERSIONS:
        raise RuntimeError(f"RDKit {rdBase.rdkitVersion} does not match the pinned versions")
    work_dir.mkdir(parents=True, exist_ok=True)
    cohort, filter_counts, invalid = load_and_filter(source_dir)
    cache, feature_names, X = build_features(cohort)
    cohort.to_csv(work_dir / "cohort.csv", index=False)
    invalid.to_csv(work_dir / "excluded_invalid_structures.csv", index=False)
    np.savez_compressed(work_dir / "descriptor_cache.npz", **cache)
    write_json(work_dir / "feature_names.json", {"feature_names": feature_names, "n_features": len(feature_names)})

    groups = cohort["solute_smiles"].astype(str).to_numpy()
    dummy = np.zeros((len(cohort), 1), dtype=np.float32)
    splits: dict[str, dict] = {}
    split_audit = []
    for fold_number, (trainval_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(dummy, groups=groups)):
        fold = f"fold{fold_number}"
        relative = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + fold_number)
        train_relative, val_relative = next(relative.split(dummy[trainval_idx], groups=groups[trainval_idx]))
        train_idx, val_idx = trainval_idx[train_relative], trainval_idx[val_relative]
        for left, right in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)):
            if set(groups[left]) & set(groups[right]):
                raise AssertionError(f"{fold}: solute overlap detected")
        Xtr, Xva, Xte, prep = preprocess_fold(X[train_idx], X[val_idx], X[test_idx])
        y = cohort["y_true"].to_numpy(dtype=np.float32)
        fold_dir = work_dir / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "arrays.npz", X_train=Xtr.astype(np.float32), X_val=Xva.astype(np.float32),
            X_test=Xte.astype(np.float32), y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
        )
        splits[fold] = {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "test_idx": test_idx.tolist()}
        scores = cohort.iloc[test_idx][["record_id", "solute_smiles", "solvent_smiles"]].reset_index(drop=True)
        scores["tanimoto_nn_distance"] = morgan_distances(cohort, train_idx, test_idx)
        print(f"[scores] {fold} Mahalanobis distance", flush=True)
        scores["mahalanobis_feature_centroid"] = mahalanobis_distances(Xtr, Xte, len(feature_names) - 1)
        support_map = cohort.iloc[train_idx].groupby("solvent_smiles")["solute_smiles"].nunique()
        scores["training_solute_support"] = cohort.iloc[test_idx]["solvent_smiles"].map(support_map).fillna(0).to_numpy(dtype=float)
        if not np.isfinite(scores[["tanimoto_nn_distance", "mahalanobis_feature_centroid", "training_solute_support"]]).all().all():
            raise ValueError(f"{fold}: non-finite retention scores")
        scores.to_csv(fold_dir / "test_scores.csv", index=False)
        write_json(fold_dir / "manifest.json", {
            "fold": fold, "rdkit_version": rdBase.rdkitVersion,
            "feature_names": feature_names, "feature_dimension": len(feature_names),
            "preprocessing": "train-only StandardScaler; no imputation; no PCA",
            "scaler_mean": prep["scaler_mean"], "scaler_scale": prep["scaler_scale"],
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "n_train_solutes": int(len(set(groups[train_idx]))),
            "n_val_solutes": int(len(set(groups[val_idx]))), "n_test_solutes": int(len(set(groups[test_idx]))),
            "mahalanobis": "LedoitWolf train-row joint solute+solvent standardized descriptor space; temperature excluded",
            "tanimoto": "Morgan radius 2, 2048 bits; nearest training-solute distance",
            "support": "number of distinct training solutes measured in the row solvent",
        })
        split_audit.append({
            "fold_id": fold, "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "n_train_solutes": len(set(groups[train_idx])), "n_val_solutes": len(set(groups[val_idx])),
            "n_test_solutes": len(set(groups[test_idx])),
        })
        print(f"[prepared] {fold} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    write_json(work_dir / "splits.json", splits)
    pd.DataFrame(split_audit).to_csv(work_dir / "split_audit.csv", index=False)
    write_json(work_dir / "preparation_audit.json", {
        "source_files": ["BigSolDBv2.2.csv", "BigSolDBv2.2_methods.csv", "BigSolDBv2.2_solid_forms.csv"],
        "rdkit_version": rdBase.rdkitVersion, "target": TARGET, "filter_counts": filter_counts,
        "invalid_structures_excluded": invalid.to_dict(orient="records"),
        "replicate_rule": "median LogS only within canonical solute + canonical solvent + temperature + source + solid_form_label",
        "protocol_rule": "exclude known synthetic/dynamic solubility and observed phase changes; retain unknown/not_checked optional metadata",
        "split_rule": "five canonical-solute-disjoint GroupKFold outer folds; 20% of outer-training solutes for validation",
        "feature_rule": "fixed 187-name RDKit selection for both solute and solvent plus temperature; train-only scaling",
    })


def support_bin(series: pd.Series) -> pd.Categorical:
    return pd.cut(series.astype(float), bins=[-0.5, 0.5, 4.5, 19.5, 99.5, np.inf], labels=["0", "1-4", "5-19", "20-99", ">=100"], ordered=True)


def member_output(path: Path, expected: pd.DataFrame) -> dict[str, np.ndarray]:
    saved = pd.read_csv(path)
    if saved["record_id"].astype(str).tolist() != expected["record_id"].astype(str).tolist():
        raise ValueError(f"{path}: prediction alignment failure")
    return {key: saved[key].to_numpy(dtype=float) for key in ("y_pred", "var_aleatoric", "var_epistemic_within")} | {"mu": saved["y_pred"].to_numpy(dtype=float)}


def train(work_dir: Path, output_dir: Path, retrain: bool = False) -> None:
    from weight_io import save_weight_checkpoint

    cfg, grid = load_shared_cfg(), load_active_grid_cfg()
    uq_mod.LOG_VAR_CLAMP = tuple(grid["log_var_clamp"])
    cohort = pd.read_csv(work_dir / "cohort.csv", low_memory=False)
    splits = json.loads((work_dir / "splits.json").read_text(encoding="utf-8"))
    feature_names = json.loads((work_dir / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_json(output_dir / "training_spec.json", {
        "model": "fixed five-member heteroscedastic MLP ensemble with MC dropout",
        "folds": FOLDS, "members_per_fold": 5, "mc_validation_passes": grid["mc_validation_passes"],
        "mc_test_passes": grid["mc_dropout_passes"], "device": device, "weights_saved": True,
        "weight_format": "weight_io v1; per-member CPU state_dict with feature schema and fold scaler",
        "hyperparameter_search": False,
    })
    for fold in FOLDS:
        fold_number = int(fold.removeprefix("fold"))
        arrays = np.load(work_dir / fold / "arrays.npz")
        manifest = json.loads((work_dir / fold / "manifest.json").read_text(encoding="utf-8"))
        train_idx = np.asarray(splits[fold]["train_idx"], dtype=int)
        test_idx = np.asarray(splits[fold]["test_idx"], dtype=int)
        train_cohort = cohort.iloc[train_idx]
        base = cohort.iloc[test_idx][["record_id", "solute_smiles", "solvent_smiles", "y_true"]].reset_index(drop=True)
        base.insert(0, "cohort_index", test_idx)
        base.insert(0, "fold_id", fold)
        scores = pd.read_csv(work_dir / fold / "test_scores.csv")
        base = base.merge(scores, on=["record_id", "solute_smiles", "solvent_smiles"], how="left", validate="one_to_one")
        global_mean = float(train_cohort["y_true"].mean())
        solvent_means = train_cohort.groupby("solvent_smiles")["y_true"].mean()
        base["dummy_pred"] = base["solvent_smiles"].map(solvent_means).fillna(global_mean)
        base["solvent_seen_in_training"] = base["solvent_smiles"].isin(set(train_cohort["solvent_smiles"]))
        base["training_solute_support_bin"] = support_bin(base["training_solute_support"])
        if not np.allclose(arrays["y_test"], base["y_true"].to_numpy(dtype=np.float32), rtol=0, atol=0):
            raise ValueError(f"{fold}: target alignment failure")
        member_outputs = []
        for member in range(5):
            member_path = output_dir / "predictions" / fold / f"member_{member}.csv"
            meta_path = output_dir / "predictions" / fold / f"member_{member}.json"
            weight_path = output_dir / "weights" / fold / f"member_{member}.pt"
            if member_path.exists() and weight_path.exists() and not retrain:
                member_outputs.append(member_output(member_path, base))
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
                weight_path, model=model, fold=fold, member=member, seed=seed, target=TARGET,
                feature_names=feature_names, hidden_dims=list(cfg["mlp_hidden_dims"]),
                log_var_clamp=grid["log_var_clamp"], scaler_mean=manifest["scaler_mean"],
                scaler_scale=manifest["scaler_scale"], training_meta=meta,
            )
            uq = predict_hetero_mc(model, arrays["X_test"], n_mc=int(grid["mc_dropout_passes"]), device=device, seed=seed + 11)
            member_path.parent.mkdir(parents=True, exist_ok=True)
            frame = base[["record_id", "y_true"]].copy()
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
    parser.add_argument("--source-dir", type=Path, default=REPOSITORY_ROOT / "data/external/bigsoldb_v2_2")
    parser.add_argument("--work-dir", type=Path, default=REPOSITORY_ROOT / "external_validation_work/bigsoldb")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results/external_validation/bigsoldb")
    parser.add_argument("--retrain", action="store_true", help="retrain even when predictions and checkpoints already exist")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args.source_dir, args.work_dir)
    if args.stage in ("train", "all"):
        train(args.work_dir, args.output_dir, retrain=args.retrain)


if __name__ == "__main__":
    main()
