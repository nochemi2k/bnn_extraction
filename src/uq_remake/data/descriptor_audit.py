"""Audit and unsupervised selection of the 210 RDKit 2D descriptors.

Selection is a fixed preprocessing rule (not tuned on test performance / y):
  drop any descriptor with missing values;
  drop near-constant columns;
  drop later members of duplicate pairs (keep RDKit list order);
  drop columns with |x| > 1e6 (would otherwise need clipping/imputation).
Nothing is imputed. The keep/drop lists are saved and hashed with the schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import ks_2samp

from uq_remake.data.feature_schema import (
    FEATURE_ABS_MAX,
    RDKIT_FULL_DESCRIPTOR_NAMES,
    SELECTION_PATH,
    set_rdkit_selection,
)

_FULL_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(list(RDKIT_FULL_DESCRIPTOR_NAMES))

NEAR_CONSTANT_CV = 1e-6
NEAR_CONSTANT_NUNIQUE = 1
DUPLICATE_CORR = 0.999999


def rdkit_descriptor_vector_full(smiles: str) -> np.ndarray:
    """Full 210-d RDKit vector for audit/selection. Non-finite → NaN; no clipping."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.full(len(RDKIT_FULL_DESCRIPTOR_NAMES), np.nan, dtype=float)
    vals = np.asarray(_FULL_CALC.CalcDescriptors(mol), dtype=float)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def raw_rdkit_matrix(smiles: pd.Series) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    rows = []
    for s in smiles.astype(str):
        if s not in cache:
            cache[s] = rdkit_descriptor_vector_full(s)
        rows.append(cache[s])
    return np.vstack(rows)


def _finite_col(X: np.ndarray, j: int) -> np.ndarray:
    v = X[:, j]
    return v[np.isfinite(v)]


def audit_rdkit_descriptors(
    smiles: pd.Series,
    folds: dict | None = None,
    *,
    dataset: str = "DS1",
) -> dict[str, Any]:
    names = list(RDKIT_FULL_DESCRIPTOR_NAMES)
    X = raw_rdkit_matrix(smiles)
    n, d = X.shape
    if d != len(names):
        raise RuntimeError(f"Raw RDKit width {d} != {len(names)}")

    missing_frac = np.mean(~np.isfinite(X), axis=0)
    extreme = np.isfinite(X) & (np.abs(X) > FEATURE_ABS_MAX)
    extreme_frac = np.mean(extreme, axis=0)

    near_constant = []
    for j, name in enumerate(names):
        v = _finite_col(X, j)
        nuniq = int(len(np.unique(np.round(v, 12)))) if len(v) else 0
        if len(v) < 2:
            near_constant.append(
                {
                    "name": name,
                    "reason": "too_few_finite",
                    "n_finite": int(len(v)),
                    "nunique": nuniq,
                }
            )
            continue
        mu = float(np.mean(v))
        sd = float(np.std(v))
        cv = sd / (abs(mu) + 1e-12)
        if nuniq <= NEAR_CONSTANT_NUNIQUE or cv < NEAR_CONSTANT_CV:
            near_constant.append(
                {
                    "name": name,
                    "reason": "nunique" if nuniq <= NEAR_CONSTANT_NUNIQUE else "low_cv",
                    "n_finite": int(len(v)),
                    "nunique": nuniq,
                    "mean": mu,
                    "std": sd,
                    "cv": cv,
                }
            )

    duplicates = []
    for i in range(d):
        vi = X[:, i]
        for j in range(i + 1, d):
            both = np.isfinite(vi) & np.isfinite(X[:, j])
            if int(both.sum()) < 8:
                continue
            a, b = vi[both], X[both, j]
            if np.allclose(a, b, rtol=0.0, atol=0.0):
                duplicates.append(
                    {
                        "a": names[i],
                        "b": names[j],
                        "kind": "identical_finite",
                        "n_overlap": int(both.sum()),
                    }
                )
                continue
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(corr) and abs(corr) >= DUPLICATE_CORR:
                duplicates.append(
                    {
                        "a": names[i],
                        "b": names[j],
                        "kind": "pearson",
                        "corr": corr,
                        "n_overlap": int(both.sum()),
                    }
                )

    per_col = []
    for j, name in enumerate(names):
        v = _finite_col(X, j)
        per_col.append(
            {
                "name": name,
                "missing_frac": float(missing_frac[j]),
                "n_missing": int((~np.isfinite(X[:, j])).sum()),
                "extreme_frac": float(extreme_frac[j]),
                "n_extreme_abs_gt_1e6": int(extreme[:, j].sum()),
                "n_finite": int(len(v)),
                "nunique_finite": int(len(np.unique(np.round(v, 12)))) if len(v) else 0,
                "min": float(np.min(v)) if len(v) else None,
                "max": float(np.max(v)) if len(v) else None,
                "mean": float(np.mean(v)) if len(v) else None,
                "std": float(np.std(v)) if len(v) else None,
            }
        )

    shift_rows = []
    if folds:
        for fold_name, split in folds.items():
            tr = np.asarray(split["train_idx"], dtype=int)
            te = np.asarray(split["test_idx"], dtype=int)
            for j, name in enumerate(names):
                a = _finite_col(X[tr], j)
                b = _finite_col(X[te], j)
                rec = {
                    "fold": fold_name,
                    "name": name,
                    "n_train_finite": int(len(a)),
                    "n_test_finite": int(len(b)),
                    "train_mean": float(np.mean(a)) if len(a) else None,
                    "test_mean": float(np.mean(b)) if len(b) else None,
                }
                if len(a) >= 8 and len(b) >= 8:
                    stat = ks_2samp(a, b, alternative="two-sided", mode="auto")
                    rec["ks_stat"] = float(stat.statistic)
                    rec["ks_pvalue"] = float(stat.pvalue)
                else:
                    rec["ks_stat"] = None
                    rec["ks_pvalue"] = None
                shift_rows.append(rec)

    return {
        "dataset": dataset,
        "n_rows": int(n),
        "n_descriptors": int(d),
        "descriptor_names": names,
        "fixed_preprocessing_rule": (
            "Drop missing, near-constant, later duplicate, and |x|>1e6 columns. "
            "No imputation. Not tuned on test performance."
        ),
        "do_not_drop_by_test_performance": True,
        "n_near_constant": len(near_constant),
        "near_constant": near_constant,
        "n_duplicate_pairs": len(duplicates),
        "duplicate_pairs": duplicates,
        "n_any_missing": int((missing_frac > 0).sum()),
        "n_any_extreme": int((extreme_frac > 0).sum()),
        "columns": per_col,
        "train_test_shifts": shift_rows,
        "top_missing": sorted(per_col, key=lambda r: -r["missing_frac"])[:15],
        "top_extreme": sorted(per_col, key=lambda r: -r["extreme_frac"])[:15],
        "top_ks": sorted(
            [r for r in shift_rows if r.get("ks_stat") is not None],
            key=lambda r: -float(r["ks_stat"]),
        )[:20],
    }


def save_descriptor_audit(audit: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = audit["dataset"].lower()
    (out_dir / f"rdkit_descriptor_audit_{ds}.json").write_text(
        json.dumps(
            {k: v for k, v in audit.items() if k not in {"columns", "train_test_shifts"}},
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(audit["columns"]).to_csv(
        out_dir / f"rdkit_descriptor_audit_{ds}_columns.csv", index=False
    )
    if audit["train_test_shifts"]:
        pd.DataFrame(audit["train_test_shifts"]).to_csv(
            out_dir / f"rdkit_descriptor_audit_{ds}_fold_shifts.csv", index=False
        )
    if audit["near_constant"]:
        pd.DataFrame(audit["near_constant"]).to_csv(
            out_dir / f"rdkit_descriptor_audit_{ds}_near_constant.csv", index=False
        )
    else:
        pd.DataFrame(columns=["name", "reason"]).to_csv(
            out_dir / f"rdkit_descriptor_audit_{ds}_near_constant.csv", index=False
        )
    if audit["duplicate_pairs"]:
        pd.DataFrame(audit["duplicate_pairs"]).to_csv(
            out_dir / f"rdkit_descriptor_audit_{ds}_duplicates.csv", index=False
        )
    else:
        pd.DataFrame(columns=["a", "b", "kind"]).to_csv(
            out_dir / f"rdkit_descriptor_audit_{ds}_duplicates.csv", index=False
        )


def select_rdkit_descriptors(audit: dict[str, Any]) -> dict[str, Any]:
    """Fixed unsupervised drop list from a full-210 audit. Not tuned on y or test metrics."""
    names = list(audit["descriptor_names"])
    reasons: dict[str, list[str]] = {n: [] for n in names}
    for col in audit["columns"]:
        n = col["name"]
        if col["n_missing"] > 0:
            reasons[n].append("missing")
        if col["n_extreme_abs_gt_1e6"] > 0:
            reasons[n].append("extreme_abs_gt_1e6")
        if col["nunique_finite"] <= 1:
            reasons[n].append("near_constant")
    for rec in audit["near_constant"]:
        n = rec["name"]
        tag = "near_constant" if rec.get("reason") in {"nunique", "low_cv", None} else str(rec["reason"])
        if tag not in reasons[n]:
            reasons[n].append(tag)

    dropped = {n for n, r in reasons.items() if r}
    remaining = [n for n in names if n not in dropped]
    remaining_set = set(remaining)
    order = {n: i for i, n in enumerate(names)}
    for pair in audit["duplicate_pairs"]:
        a, b = pair["a"], pair["b"]
        if a in remaining_set and b in remaining_set:
            later = b if order[b] > order[a] else a
            kept = a if later == b else b
            reasons[later].append(f"duplicate_of_{kept}")
            remaining_set.discard(later)

    keep = [n for n in names if n in remaining_set]
    drop_records = [
        {"name": n, "reasons": reasons[n]}
        for n in names
        if reasons[n]
    ]
    return {
        "full_n": len(names),
        "full_names": names,
        "selected_n": len(keep),
        "selected_names": keep,
        "dropped": drop_records,
        "n_dropped": len(drop_records),
        "rules": {
            "drop_any_missing": True,
            "drop_near_constant": True,
            "drop_duplicate_later_in_rdkit_order": True,
            "drop_extreme_abs_gt": FEATURE_ABS_MAX,
            "imputation": False,
            "fit_on": "unique ligand SMILES in the research tables (not y, not test metrics)",
            "drop_descriptors_by_test_performance": False,
        },
    }


def save_rdkit_selection(selection: dict[str, Any], out_path: Path | None = None) -> Path:
    out_path = out_path or SELECTION_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **selection,
        "pinned_rdkit_version": "2023.09.6",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    set_rdkit_selection(selection["selected_names"], selection["dropped"])
    dropped_csv = out_path.parent.parent / "audits" / "rdkit_descriptor_dropped.csv"
    dropped_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in selection["dropped"]:
        rows.append({"name": rec["name"], "reasons": ";".join(rec["reasons"])})
    pd.DataFrame(rows).to_csv(dropped_csv, index=False)
    return out_path
