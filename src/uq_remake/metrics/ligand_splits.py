"""Ligand-grouped inner splits. Not a risk-control procedure.

NOT ON THE PRODUCTION PATH. `data/prepare_data.py`, `data/build_ds2.py`,
`data/build_m4.py` and `data/run_cmisplit.py` each build their inner ligand-disjoint split
inline with a raw `GroupShuffleSplit` rather than calling these helpers. The helpers encode
the grouping contract those four sites re-implement, and `tests/test_cv_metrics.py` and
`tests/test_downstream_cv.py` assert it here. Kept deliberately: the alternative is four
hand-rolled copies with no single place to test the invariant.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def nested_calibration_split(train_idx, groups, seed: int = 42, cal_frac: float = 0.2):
    """Split outer-train into inner-train / calibration. Groups are ligands."""
    train_idx = np.asarray(train_idx)
    groups = np.asarray(groups)
    if len(train_idx) != len(groups):
        raise ValueError("train_idx and groups length mismatch")
    gss = GroupShuffleSplit(n_splits=1, test_size=float(cal_frac), random_state=int(seed))
    inner_rel, cal_rel = next(gss.split(train_idx, groups=groups))
    return train_idx[inner_rel], train_idx[cal_rel]


def nested_three_way_ligand_split(
    train_idx,
    groups,
    seed: int = 42,
    cal_frac: float = 0.2,
    val_frac_of_remainder: float = 0.2,
):
    """Ligand-grouped inner-train / inner-val / inner-holdout split of outer-train."""
    train_idx = np.asarray(train_idx)
    groups = np.asarray(groups)
    rest, cal = nested_calibration_split(train_idx, groups, seed=int(seed), cal_frac=float(cal_frac))
    idx_to_group = {int(i): g for i, g in zip(train_idx, groups)}
    rest_groups = np.asarray([idx_to_group[int(i)] for i in rest])
    inner, val = nested_calibration_split(
        rest, rest_groups, seed=int(seed) + 1, cal_frac=float(val_frac_of_remainder)
    )
    return inner, val, cal
