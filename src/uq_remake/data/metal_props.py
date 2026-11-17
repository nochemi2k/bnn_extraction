"""Row-level CMI metal_props (9-d) for tabular models.

Convention: **row-level CMI M4 feature columns**.
  Invariant within element: Atomic_number, metal_density, polarizability,
  vdW radius, atomic weight, Pyykkö covalent radius, Sanderson EN.
  Oxidation-state dependent in CMI: IE, metal_valence_e.

Oxidation state is a separate continuous feature. Row-level IE/valence are
derived CMI chemical properties, not a second copy of the ox-state integer.
They come from the CMI *feature* tables (not from Exp LgK1 / logK).

Fallback if a row cell is missing (never zero-fill):
  1. CMI row-level column
  2. unique CMI value for (element, ox.1f) in this corpus
  3. for invariant columns only: unique CMI value for the element
If still unresolved, raise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from uq_remake import REMAKE_ROOT

METAL_PROP_COLS = [
    "Atomic_number",
    "metal_density",
    "metal_dipole_polarizability",
    "metal_vdw_radius",
    "metal_atomic_weight",
    "metal_covalent_radius_by_pyykko",
    "metal_Sanderson Electronegativity",
    "IE",
    "metal_valence_e",
]
METAL_PROP_DIM = len(METAL_PROP_COLS)
METAL_PROPS_CONVENTION = "row_level_cmi"

INVARIANT_PROP_COLS = [
    "Atomic_number",
    "metal_density",
    "metal_dipole_polarizability",
    "metal_vdw_radius",
    "metal_atomic_weight",
    "metal_covalent_radius_by_pyykko",
    "metal_Sanderson Electronegativity",
]
OX_DEPENDENT_PROP_COLS = ["IE", "metal_valence_e"]

METAL_PROP_UNITS = {
    "Atomic_number": "dimensionless (Z); CMI row-level",
    "metal_density": "g cm^-3 (CMI M4 table); element-invariant in this corpus",
    "metal_dipole_polarizability": "as provided in CMI M4 (not the 10^-24 cm^3 handbook convention); element-invariant",
    "metal_vdw_radius": "pm (CMI M4 table); element-invariant",
    "metal_atomic_weight": "g mol^-1 (CMI M4 table); element-invariant",
    "metal_covalent_radius_by_pyykko": "pm (Pyykko, CMI M4 table); element-invariant",
    "metal_Sanderson Electronegativity": "Sanderson scale as provided in CMI M4; element-invariant",
    "IE": "eV; CMI row-level (varies with oxidation state). Not computed from logK.",
    "metal_valence_e": "count; CMI row-level (varies with oxidation state). Not computed from logK.",
}

_TABLE_PATH = REMAKE_ROOT / "data" / "processed" / "metal_property_table.json"
_SPECIES: dict[str, np.ndarray] | None = None
_ELEMENT_INVARIANT: dict[str, np.ndarray] | None = None
_LAST_FALLBACK: list[dict[str, Any]] = []


def species_lookup_key(element: Any, oxidation: Any) -> str:
    return f"{str(element).strip()}:{float(oxidation):.1f}"


def _vec_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.array([float(row[c]) for c in METAL_PROP_COLS], dtype=np.float64)


def set_lookup_tables(
    species: dict[str, dict[str, Any]],
    by_element: dict[str, dict[str, Any]],
) -> None:
    global _SPECIES, _ELEMENT_INVARIANT
    spec_vecs = {}
    for k, rec in species.items():
        vec = _vec_from_row(rec)
        if not np.isfinite(vec).all():
            raise ValueError(f"Non-finite species metal properties for {k!r}: {vec}")
        spec_vecs[str(k)] = vec.astype(np.float64)
    inv = {}
    for el, rec in by_element.items():
        vec = np.array([float(rec[c]) for c in INVARIANT_PROP_COLS], dtype=np.float64)
        if not np.isfinite(vec).all():
            raise ValueError(f"Non-finite invariant metal properties for {el!r}")
        inv[str(el)] = vec
    _SPECIES = spec_vecs
    _ELEMENT_INVARIANT = inv


def _load_lookups() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    global _SPECIES, _ELEMENT_INVARIANT
    if _SPECIES is not None and _ELEMENT_INVARIANT is not None:
        return _SPECIES, _ELEMENT_INVARIANT
    if not _TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Metal property table missing: {_TABLE_PATH}. Run prepare_data.py first."
        )
    raw = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    set_lookup_tables(raw["by_species"], raw["by_element"])
    assert _SPECIES is not None and _ELEMENT_INVARIANT is not None
    return _SPECIES, _ELEMENT_INVARIANT


def last_fallback_report() -> list[dict[str, Any]]:
    return list(_LAST_FALLBACK)


def metal_props_matrix(
    df: pd.DataFrame,
    metal_el_col: str = "metal_el",
    ox_col: str = "metal_oxidation_state",
    source_cols: dict[str, str] | None = None,
    *,
    reset_report: bool = True,
) -> np.ndarray:
    """(n, 9) row-level CMI metal_props. Missing cells use documented fallbacks, never 0."""
    global _LAST_FALLBACK
    if reset_report:
        _LAST_FALLBACK = []
    if metal_el_col not in df.columns:
        raise KeyError(f"Need column {metal_el_col} for metal_props")
    src = source_cols or {c: c for c in METAL_PROP_COLS}
    missing_src = [src[c] for c in METAL_PROP_COLS if src[c] not in df.columns]
    n = len(df)
    X = np.full((n, METAL_PROP_DIM), np.nan, dtype=np.float64)
    if not missing_src:
        X = df[[src[c] for c in METAL_PROP_COLS]].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
    else:
        present = [c for c in METAL_PROP_COLS if src[c] in df.columns]
        if present:
            idx = [METAL_PROP_COLS.index(c) for c in present]
            X[:, idx] = df[[src[c] for c in present]].apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=np.float64
            )

    need = ~np.isfinite(X)
    if not need.any():
        return X.astype(np.float32)

    species_tab, elem_inv = _load_lookups()
    els = df[metal_el_col].astype(str).str.strip().to_numpy()
    if ox_col not in df.columns:
        raise KeyError(
            f"Missing {ox_col} required to fall back unresolved metal_props (never zero-fill)."
        )
    ox = pd.to_numeric(df[ox_col], errors="coerce").to_numpy(dtype=float)
    rows, cols = np.where(need)
    for i, j in zip(rows.tolist(), cols.tolist()):
        col = METAL_PROP_COLS[j]
        el = els[i]
        ox_i = ox[i]
        filled = False
        kind = ""
        if np.isfinite(ox_i):
            spec = species_tab.get(species_lookup_key(el, ox_i))
            if spec is not None and np.isfinite(spec[j]):
                X[i, j] = spec[j]
                filled = True
                kind = "species_(element,ox)"
        if not filled and col in INVARIANT_PROP_COLS:
            inv = elem_inv.get(el)
            if inv is not None:
                jj = INVARIANT_PROP_COLS.index(col)
                if np.isfinite(inv[jj]):
                    X[i, j] = inv[jj]
                    filled = True
                    kind = "element_invariant"
        if not filled:
            raise ValueError(
                f"Unresolved metal property {col} for element {el!r} ox={ox_i} "
                f"(row {i}). Do not zero-fill."
            )
        _LAST_FALLBACK.append(
            {
                "row_index": int(i),
                "element": el,
                "oxidation_state": None if not np.isfinite(ox_i) else float(ox_i),
                "column": col,
                "fallback": kind,
            }
        )
    return X.astype(np.float32)


def metal_element_props(el: str, oxidation_state: float | None = None) -> np.ndarray:
    """Species-level CMI vector. Requires oxidation_state (row-level convention)."""
    if oxidation_state is None or not np.isfinite(float(oxidation_state)):
        raise TypeError(
            "metal_element_props requires oxidation_state under row-level CMI convention. "
            "Use metal_props_matrix on a table with CMI columns."
        )
    species_tab, _ = _load_lookups()
    key = species_lookup_key(el, oxidation_state)
    if key not in species_tab:
        raise KeyError(
            f"Unresolved metal species {key}. Complete the CMI species table rather than zero-filling."
        )
    return species_tab[key].astype(np.float32)


def build_metal_property_table(df: pd.DataFrame) -> dict[str, Any]:
    """Audit CMI metal columns and build species + element lookup tables."""
    need = ["metal_el", "metal_oxidation_state", *METAL_PROP_COLS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot build metal property table; missing columns {missing}")

    work = df.copy()
    for c in METAL_PROP_COLS + ["metal_oxidation_state"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work["metal_el"] = work["metal_el"].astype(str).str.strip()
    work = work[work["metal_el"].ne("") & work["metal_el"].notna()]

    missing_elements: list[dict[str, Any]] = []
    by_element: dict[str, dict[str, Any]] = {}
    by_species: dict[str, dict[str, Any]] = {}
    ox_coverage: dict[str, Any] = {}
    pt = Chem.GetPeriodicTable()

    for el, g in work.groupby("metal_el", sort=True):
        rec: dict[str, Any] = {"element": el, "n_rows": int(len(g))}
        bad = False
        for col in INVARIANT_PROP_COLS:
            vals = g[col].dropna().unique()
            if len(vals) == 0:
                missing_elements.append({"element": el, "column": col, "reason": "all_missing"})
                bad = True
                continue
            if len(vals) > 1:
                missing_elements.append(
                    {
                        "element": el,
                        "column": col,
                        "reason": "not_invariant",
                        "nunique": int(len(vals)),
                        "values": [float(v) for v in vals[:8]],
                    }
                )
                bad = True
                continue
            rec[col] = float(vals[0])
        ox_counts = (
            g["metal_oxidation_state"].dropna().round(1).value_counts()
            if g["metal_oxidation_state"].notna().any()
            else pd.Series(dtype=int)
        )
        rec["oxidation_states"] = [float(x) for x in sorted(ox_counts.index.tolist())]
        rec["ie_nunique"] = int(g["IE"].nunique(dropna=True))
        rec["valence_nunique"] = int(g["metal_valence_e"].nunique(dropna=True))
        try:
            rec["rdkit_atomic_number"] = int(pt.GetAtomicNumber(str(el)))
            rec["rdkit_atomic_weight"] = float(pt.GetAtomicWeight(str(el)))
            rec["rdkit_rvdw_pm"] = float(pt.GetRvdw(str(el))) * 100.0
            rec["rdkit_rcov_pm"] = float(pt.GetRcovalent(str(el))) * 100.0
        except Exception as exc:
            rec["rdkit_lookup_error"] = str(exc)

        ox_ie: dict[str, Any] = {}
        ox_key = g["metal_oxidation_state"].round(1)
        for ox, gg in g.groupby(ox_key):
            if pd.isna(ox):
                continue
            spec_rec = {c: rec[c] for c in INVARIANT_PROP_COLS if c in rec}
            spec_bad = False
            for col in OX_DEPENDENT_PROP_COLS:
                nuniq = int(gg[col].nunique(dropna=True))
                if nuniq == 0:
                    missing_elements.append(
                        {
                            "element": el,
                            "oxidation_state": float(ox),
                            "column": col,
                            "reason": "all_missing",
                        }
                    )
                    spec_bad = True
                    continue
                if nuniq > 1:
                    missing_elements.append(
                        {
                            "element": el,
                            "oxidation_state": float(ox),
                            "column": col,
                            "reason": "not_unique_at_species",
                            "nunique": nuniq,
                        }
                    )
                    spec_bad = True
                    continue
                spec_rec[col] = float(gg[col].dropna().iloc[0])
            ox_ie[f"{float(ox):.1f}"] = {
                "n": int(len(gg)),
                "IE": spec_rec.get("IE"),
                "metal_valence_e": spec_rec.get("metal_valence_e"),
            }
            if not spec_bad and not bad:
                spec_rec.update(
                    {
                        "element": el,
                        "oxidation_state": float(ox),
                        "n_rows": int(len(gg)),
                    }
                )
                by_species[species_lookup_key(el, ox)] = spec_rec
        ox_coverage[str(el)] = ox_ie
        if not bad:
            by_element[str(el)] = rec

    if missing_elements:
        raise ValueError(
            "Metal property table incomplete; see missing list. "
            f"n_issues={len(missing_elements)} sample={missing_elements[:8]}"
        )

    els = sorted(by_element)
    X = np.vstack(
        [np.array([by_element[e][c] for c in INVARIANT_PROP_COLS], dtype=float) for e in els]
    )
    sd = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    Z = (X - X.mean(axis=0)) / sd
    nearest = []
    for i, el in enumerate(els):
        d = np.sqrt(((Z - Z[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        j = int(np.argmin(d))
        nearest.append(
            {"element": el, "nearest_element": els[j], "zscored_l2_invariant7": float(d[j])}
        )

    ranges = {}
    for col in METAL_PROP_COLS:
        if col in INVARIANT_PROP_COLS:
            vals = np.array([by_element[e][col] for e in els], dtype=float)
        else:
            vals = np.array([rec[col] for rec in by_species.values() if col in rec], dtype=float)
        ranges[col] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": int(len(vals)),
        }

    payload = {
        "by_element": by_element,
        "by_species": by_species,
        "n_elements": len(by_element),
        "n_species": len(by_species),
        "prop_cols": list(METAL_PROP_COLS),
        "invariant_cols": list(INVARIANT_PROP_COLS),
        "ox_dependent_cols": list(OX_DEPENDENT_PROP_COLS),
        "convention": METAL_PROPS_CONVENTION,
        "ie_valence_rule": (
            "row-level CMI IE and metal_valence_e; unique given (element, oxidation_state) "
            "in this corpus. Oxidation state remains a separate explanatory feature. "
            "Values come from CMI feature CSVs, not from the logK target."
        ),
        "target_leakage": False,
        "units": METAL_PROP_UNITS,
        "ranges": ranges,
        "nearest_neighbors": nearest,
        "oxidation_coverage": ox_coverage,
        "zero_fill": False,
        "fallback_order": [
            "row_level_cmi_column",
            "species_table_(element,ox)",
            "element_invariant_for_density_radii_weight_only",
        ],
    }
    set_lookup_tables(by_species, by_element)
    return payload


def save_metal_property_reports(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {
        "prop_cols": payload["prop_cols"],
        "convention": payload["convention"],
        "ie_valence_rule": payload["ie_valence_rule"],
        "target_leakage": False,
        "zero_fill": False,
        "fallback_order": payload["fallback_order"],
        "by_element": payload["by_element"],
        "by_species": payload["by_species"],
    }
    _TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TABLE_PATH.write_text(json.dumps(table, indent=2), encoding="utf-8")
    (out_dir / "metal_property_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (out_dir / "metal_property_units.json").write_text(
        json.dumps(payload["units"], indent=2), encoding="utf-8"
    )
    (out_dir / "metal_property_ranges.json").write_text(
        json.dumps(
            {
                "convention": payload["convention"],
                "ranges": payload["ranges"],
                "nearest_neighbors": payload["nearest_neighbors"],
                "n_elements": payload["n_elements"],
                "n_species": payload["n_species"],
                "ie_valence_rule": payload["ie_valence_rule"],
                "target_leakage": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(columns=["element", "column", "reason"]).to_csv(
        out_dir / "metal_property_missing_elements.csv", index=False
    )
    rows = []
    for el, rec in payload["by_element"].items():
        row = {
            k: rec.get(k)
            for k in ["element", "n_rows", *INVARIANT_PROP_COLS, "ie_nunique", "valence_nunique"]
        }
        rows.append(row)
    pd.DataFrame(rows).sort_values("Atomic_number").to_csv(
        out_dir / "metal_property_table.csv", index=False
    )
    ox_rows = []
    for el, oxmap in payload["oxidation_coverage"].items():
        for ox, stats in oxmap.items():
            ox_rows.append({"element": el, "oxidation_state": ox, **stats})
    pd.DataFrame(ox_rows).to_csv(out_dir / "metal_property_oxidation_coverage.csv", index=False)
    spec_rows = []
    for k, rec in payload["by_species"].items():
        spec_rows.append({"species_key": k, **{c: rec.get(c) for c in ["element", "oxidation_state", *METAL_PROP_COLS, "n_rows"]}})
    pd.DataFrame(spec_rows).to_csv(out_dir / "metal_property_species_table.csv", index=False)
