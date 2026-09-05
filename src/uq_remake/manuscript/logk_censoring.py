"""Analysis task 4: reporting-censoring analysis for DS1, DS2, and the CMI external file.

The literature records a stability constant when a complex forms; non-binding
is not reported as a number, so the training distribution is censored toward
higher logK. Quantified here as a histogram, cumulative fractions <=0/1/2/3,
median, minimum -- pooled and split by periodic-table block (s / 3d / 4d-5d /
4f / 5f / p_block, from atomic number; see `manuscript_common.block_from_atomic_number`).

Reproduces, for DS2 endpoints pooled (logK_x, logK_y), from
`manuscripts/derived/analysis/chemical_sanity_checks.json`:
  frac <= 0   ~ 1.71%
  frac <= 1   ~ 5.52%
  frac <= 2   ~ 13.56%
  median      ~ 5.50
  minimum     ~ -2.65

Writes `manuscripts/derived/tables/logk_censoring.csv`.

    python -m uq_remake.manuscript.logk_censoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uq_remake.manuscript.manuscript_common import (
    CMI_EXTERNAL_PATH,
    DS1_PATH,
    DS2_PATH,
    DERIVED_TABLES,
    block_from_atomic_number,
    write_provenance,
)

OUT = DERIVED_TABLES / "logk_censoring.csv"

BIN_EDGES = [-np.inf, -1, 0, 1, 2, 3, 5, 10, 20, np.inf]
BIN_LABELS = [
    "(-inf,-1]",
    "(-1,0]",
    "(0,1]",
    "(1,2]",
    "(2,3]",
    "(3,5]",
    "(5,10]",
    "(10,20]",
    "(20,inf)",
]


def _nan() -> float:
    return float("nan")


def censoring_row(values: pd.Series, *, dataset: str, block: str, n_min: int = 5) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(v))
    row = {
        "dataset": dataset,
        "metal_block": block,
        "n": n,
        "frac_le_0": float((v <= 0).mean()) if n >= n_min else _nan(),
        "frac_le_1": float((v <= 1).mean()) if n >= n_min else _nan(),
        "frac_le_2": float((v <= 2).mean()) if n >= n_min else _nan(),
        "frac_le_3": float((v <= 3).mean()) if n >= n_min else _nan(),
        "median": float(v.median()) if n >= n_min else _nan(),
        "minimum": float(v.min()) if n >= n_min else _nan(),
    }
    if n >= n_min:
        counts = pd.cut(v, bins=BIN_EDGES, labels=BIN_LABELS, right=True, include_lowest=True).value_counts()
        for label in BIN_LABELS:
            row[f"hist_{label}"] = int(counts.get(label, 0))
    else:
        for label in BIN_LABELS:
            row[f"hist_{label}"] = 0 if n else 0
    return row


def dataset_rows(dataset_name: str, logk: pd.Series, atomic_number: pd.Series) -> list[dict]:
    rows = [censoring_row(logk, dataset=dataset_name, block="all")]
    blocks = atomic_number.map(block_from_atomic_number)
    for block, idx in blocks.groupby(blocks).groups.items():
        rows.append(censoring_row(logk.loc[idx], dataset=dataset_name, block=block))
    return rows


def main() -> dict:
    ds1 = pd.read_csv(DS1_PATH, usecols=["logK", "Atomic_number"])
    ds2 = pd.read_csv(DS2_PATH, usecols=["logK_x", "logK_y", "Atomic_number_x", "Atomic_number_y"])
    cmi = pd.read_csv(CMI_EXTERNAL_PATH, usecols=["logK", "Atomic_number"])

    rows: list[dict] = []
    rows.extend(dataset_rows("DS1", ds1["logK"], ds1["Atomic_number"]))
    rows.extend(dataset_rows("CMI_external", cmi["logK"], cmi["Atomic_number"]))

    ds2_logk = pd.concat([ds2["logK_x"], ds2["logK_y"]], ignore_index=True)
    ds2_atomic = pd.concat([ds2["Atomic_number_x"], ds2["Atomic_number_y"]], ignore_index=True)
    rows.extend(dataset_rows("DS2_endpoints_pooled", ds2_logk, ds2_atomic))

    out = pd.DataFrame(rows)
    out["source"] = ""
    out.loc[out.dataset == "DS1", "source"] = "data/processed/ds1.csv (logK, Atomic_number)"
    out.loc[out.dataset == "CMI_external", "source"] = "data/processed/m4_external.csv (logK, Atomic_number)"
    out.loc[out.dataset == "DS2_endpoints_pooled", "source"] = "data/processed/ds2.csv (logK_x, logK_y, Atomic_number_x, Atomic_number_y pooled)"

    DERIVED_TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    write_provenance(
        OUT,
        module="logk_censoring.main",
        inputs=[DS1_PATH, DS2_PATH, CMI_EXTERNAL_PATH],
        notes={
            "bin_edges": ["-inf", -1, 0, 1, 2, 3, 5, 10, 20, "inf"],
            "block_classifier": "atomic number: s, 3d, 4d-5d, 4f (La-Lu), 5f (Ac-Lr), p_block (else)",
        },
    )
    print(f"wrote {OUT} ({len(out)} rows)", flush=True)

    ds2_all = out[(out.dataset == "DS2_endpoints_pooled") & (out.metal_block == "all")].iloc[0]
    print(
        "DS2 endpoints pooled: "
        f"frac<=0={ds2_all['frac_le_0']:.4f} frac<=1={ds2_all['frac_le_1']:.4f} "
        f"frac<=2={ds2_all['frac_le_2']:.4f} median={ds2_all['median']:.2f} min={ds2_all['minimum']:.2f}",
        flush=True,
    )
    return {"table": out}


if __name__ == "__main__":
    main()
