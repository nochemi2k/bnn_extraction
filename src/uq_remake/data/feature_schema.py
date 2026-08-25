"""Fixed continuous feature schema for tabular (RDKit / CPEMB) models.

Electrolyte *identity* is never a model feature. It may be used for DS2 medium
matching, grouping, split diagnostics, and reporting only.

Solvent MW and density are continuous medium variables (plus T and I). They are a
1:1 encoding of the six CMI electrolytes in this corpus; see the solvent audit
written by prepare_data. Do not add an electrolyte one-hot block as well.

Dimensions are *schema-generated* from the selected RDKit 2D block (a subset of
the pinned 210-name RDKit 2023.09.6 list) plus metal_props and medium. Model code
must read widths from the fold manifest / this module, never hard-code them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rdkit import rdBase
from rdkit.Chem import Descriptors

from uq_remake import REMAKE_ROOT
from uq_remake.data.metal_props import METAL_PROP_COLS, METAL_PROPS_CONVENTION

# Pin: RDKit 2023.09.6 (pip package 2023.9.6 reports rdBase.rdkitVersion "2023.09.6").
PINNED_RDKIT_VERSION = "2023.09.6"
PINNED_RDKIT_VERSIONS = frozenset({"2023.09.6", "2023.9.6"})
EXPECTED_RDKIT_N_FULL = 210

# Continuous solvent properties are specified experimental conditions (available
# at inference if the medium is known). In this corpus they uniquely identify
# electrolyte; they are *not* one-hot electrolyte columns.
INCLUDE_SOLVENT_PROPERTIES = True

# |x| > this on the *full* 210-d list → drop that descriptor (no clip, no impute).
FEATURE_ABS_MAX = 1e6

RDKIT_FULL_DESCRIPTOR_NAMES: list[str] = [name for name, _ in Descriptors._descList]
SELECTION_PATH = REMAKE_ROOT / "data" / "processed" / "features" / "rdkit_descriptor_selection.json"
_SELECTED: list[str] | None = None
_DROPPED: list[dict] | None = None

DS1_CONTINUOUS_COLS: list[str] = [
    "metal_oxidation_state",
    "ligand_charge",
    "complex_charge",
    "temperature_C",
    "ionic_strength_M",
]
DS2_CONTINUOUS_COLS: list[str] = [
    "metal_oxidation_state_x",
    "metal_oxidation_state_y",
    "ligand_charge",
    "complex_charge_x",
    "complex_charge_y",
    "temperature_C",
    "ionic_strength_M",
]
SOLVENT_COLS: list[str] = ["solvent_mw", "solvent_density"]

if INCLUDE_SOLVENT_PROPERTIES:
    DS1_CONTINUOUS_COLS = DS1_CONTINUOUS_COLS + SOLVENT_COLS
    DS2_CONTINUOUS_COLS = DS2_CONTINUOUS_COLS + SOLVENT_COLS

# Chemprop DS1 additional features: CMI M4 column names, continuous only.
CHEMPROP_DS1_FEATURE_COLUMNS: list[str] = [
    "Atomic_number",
    "metal_density",
    "metal_dipole_polarizability",
    "metal_vdw_radius",
    "metal_atomic_weight",
    "metal_covalent_radius_by_pyykko",
    "metal_Sanderson Electronegativity",
    "IE",
    "Total_ligand charge",
    "Metal_charge",
    "Cmplx_charge",
    "metal_valence_e",
    "Temp",
    "Concentration of Medium",
    "Exact MW of solvent (g/mol)",
    "Density of solvent g/cm-3",
]


def set_rdkit_selection(keep: list[str], dropped: list[dict] | None = None) -> None:
    """Install the unsupervised keep-list (RDKit order). Used by prepare_data and loaders."""
    global _SELECTED, _DROPPED
    full = list(RDKIT_FULL_DESCRIPTOR_NAMES)
    full_i = {n: i for i, n in enumerate(full)}
    unknown = [n for n in keep if n not in full_i]
    if unknown:
        raise ValueError(f"Keep list contains names not in RDKit 2023.09.6 _descList: {unknown[:8]}")
    ordered = sorted(keep, key=lambda n: full_i[n])
    if list(keep) != ordered:
        raise ValueError("Keep list must preserve RDKit _descList order")
    _SELECTED = list(keep)
    _DROPPED = list(dropped or [])


def load_rdkit_selection(path: Path | None = None) -> list[str]:
    path = path or SELECTION_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    set_rdkit_selection(raw["selected_names"], raw.get("dropped"))
    return rdkit_selected_descriptor_names()


def rdkit_selected_descriptor_names() -> list[str]:
    global _SELECTED
    if _SELECTED is None:
        if not SELECTION_PATH.exists():
            raise FileNotFoundError(
                f"RDKit descriptor selection missing: {SELECTION_PATH}. "
                "Run prepare_data.py before building or training RDKit matrices."
            )
        load_rdkit_selection()
    assert _SELECTED is not None
    return list(_SELECTED)


def rdkit_dropped_records() -> list[dict]:
    if _SELECTED is None and SELECTION_PATH.exists():
        load_rdkit_selection()
    return list(_DROPPED or [])


def rdkit_descriptor_names() -> list[str]:
    return rdkit_selected_descriptor_names()


def ds1_rdkit_column_names() -> list[str]:
    selected = rdkit_selected_descriptor_names()
    return (
        [f"rdkit_{n}" for n in selected]
        + list(METAL_PROP_COLS)
        + list(DS1_CONTINUOUS_COLS)
    )


def ds2_rdkit_column_names() -> list[str]:
    """Names aligned with build_ds2_rdkit_matrix (ligand+x block, then y props, then extras).

    Oxidation state of *x*, complex charge of *x*, ligand charge, and solvent
    variables appear in the DS1-style x-block (unsuffixed) once. y contributes
    metal_props_y + ox_y + complex_charge_y only.
    """
    extra = ["metal_oxidation_state_y", "complex_charge_y"]
    return (
        ds1_rdkit_column_names()
        + [f"{c}_y" for c in METAL_PROP_COLS]
        + extra
    )


def ds1_rdkit_dimension() -> int:
    return len(ds1_rdkit_column_names())


def ds2_rdkit_dimension() -> int:
    return len(ds2_rdkit_column_names())


def preprocessing_rules() -> dict:
    return {
        "rdkit_full_n": EXPECTED_RDKIT_N_FULL,
        "rdkit_drop_missing": True,
        "rdkit_drop_near_constant": True,
        "rdkit_drop_duplicate_later_in_rdkit_order": True,
        "rdkit_drop_extreme_abs_gt": FEATURE_ABS_MAX,
        "imputation": False,
        "scaling": "StandardScaler_on_train",
        "drop_descriptors_by_test_performance": False,
        "electrolyte_onehot": False,
        "metal_onehot": False,
        "metal_props_convention": METAL_PROPS_CONVENTION,
        "ie_valence": "row_level_cmi_not_modal",
        "missing_metal_props": "never_zero_fill; species_(element,ox) then element_invariant",
    }


def _hash_payload(dataset: str) -> dict:
    d = dataset.upper()
    if d == "DS1":
        names = ds1_rdkit_column_names()
        cont = list(DS1_CONTINUOUS_COLS)
    elif d == "DS2":
        names = ds2_rdkit_column_names()
        cont = list(DS2_CONTINUOUS_COLS)
    else:
        raise ValueError(d)
    selected = rdkit_selected_descriptor_names()
    return {
        "dataset": d,
        "pinned_rdkit_version": PINNED_RDKIT_VERSION,
        "rdkit_n_full": EXPECTED_RDKIT_N_FULL,
        "rdkit_n_descriptors": len(selected),
        "descriptor_names": list(selected),
        "rdkit_full_descriptor_names": list(RDKIT_FULL_DESCRIPTOR_NAMES),
        "rdkit_dropped_names": [r["name"] for r in rdkit_dropped_records()],
        "metal_prop_cols": list(METAL_PROP_COLS),
        "metal_props_convention": METAL_PROPS_CONVENTION,
        "continuous_feature_names": cont,
        "feature_names": names,
        "include_solvent_properties": INCLUDE_SOLVENT_PROPERTIES,
        "electrolyte_in_X": False,
        "metal_onehot_in_X": False,
        "preprocessing": preprocessing_rules(),
    }


def feature_schema_hash(dataset: str) -> str:
    blob = json.dumps(_hash_payload(dataset), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def schema_dict(dataset: str) -> dict:
    d = dataset.upper()
    payload = _hash_payload(d)
    return {
        **payload,
        "feature_dimension": len(payload["feature_names"]),
        "feature_hash": feature_schema_hash(d),
        "runtime_rdkit_version": str(rdBase.rdkitVersion),
        "note": (
            "Electrolyte identity is used for experimental matching and grouping "
            "but is excluded from the tabular predictor. Continuous solvent MW/density "
            "plus T and I are the medium features. metal_props IE/valence are row-level "
            "CMI values (oxidation-state-dependent), not modal-element values. "
            "Oxidation state remains a separate explanatory feature. "
            "RDKit 2D columns with missing values, near-zero variance, or duplicate "
            "information are dropped; nothing is imputed."
        ),
    }


def verify_feature_name_layout(dataset: str, names: list[str]) -> dict:
    """Fail if ox/charge/ligand charge/solvent are duplicated or one-hots leaked in."""
    d = dataset.upper()
    expected = ds1_rdkit_column_names() if d == "DS1" else ds2_rdkit_column_names()
    if list(names) != expected:
        raise RuntimeError(f"{d} feature names drifted from schema")
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"{d} duplicate feature names: {dup}")

    def _count(pred) -> int:
        return sum(1 for n in names if pred(n))

    lower = [n.lower() for n in names]
    if any(n.startswith("electrolyte_") or n == "electrolyte" for n in lower):
        raise RuntimeError(f"{d}: electrolyte column leaked into X")
    if any("onehot" in n or n.startswith("metal_x_") or n == "unknown_metal" for n in lower):
        raise RuntimeError(f"{d}: metal/electrolyte one-hot leaked into X")

    checks = {
        "metal_oxidation_state": _count(lambda n: n == "metal_oxidation_state"),
        "metal_oxidation_state_y": _count(lambda n: n == "metal_oxidation_state_y"),
        "complex_charge": _count(lambda n: n == "complex_charge"),
        "complex_charge_y": _count(lambda n: n == "complex_charge_y"),
        "ligand_charge": _count(lambda n: n == "ligand_charge"),
        "solvent_mw": _count(lambda n: n == "solvent_mw"),
        "solvent_density": _count(lambda n: n == "solvent_density"),
        "IE": _count(lambda n: n == "IE"),
        "IE_y": _count(lambda n: n == "IE_y"),
        "metal_valence_e": _count(lambda n: n == "metal_valence_e"),
        "metal_valence_e_y": _count(lambda n: n == "metal_valence_e_y"),
    }
    expect = {
        "DS1": {
            "metal_oxidation_state": 1,
            "metal_oxidation_state_y": 0,
            "complex_charge": 1,
            "complex_charge_y": 0,
            "ligand_charge": 1,
            "solvent_mw": 1 if INCLUDE_SOLVENT_PROPERTIES else 0,
            "solvent_density": 1 if INCLUDE_SOLVENT_PROPERTIES else 0,
            "IE": 1,
            "IE_y": 0,
            "metal_valence_e": 1,
            "metal_valence_e_y": 0,
        },
        "DS2": {
            "metal_oxidation_state": 1,  # x-block, unsuffixed
            "metal_oxidation_state_y": 1,
            "complex_charge": 1,
            "complex_charge_y": 1,
            "ligand_charge": 1,
            "solvent_mw": 1 if INCLUDE_SOLVENT_PROPERTIES else 0,
            "solvent_density": 1 if INCLUDE_SOLVENT_PROPERTIES else 0,
            "IE": 1,
            "IE_y": 1,
            "metal_valence_e": 1,
            "metal_valence_e_y": 1,
        },
    }[d]
    for key, want in expect.items():
        got = checks[key]
        if got != want:
            raise RuntimeError(f"{d} layout: {key} appears {got} times, expected {want}")
    return {"dataset": d, "ok": True, "counts": checks, "n": len(names)}


def default_schema_path(dataset: str) -> Path:
    return REMAKE_ROOT / "data" / "processed" / "features" / f"{dataset.lower()}_rdkit_schema.json"


def assert_runtime_schema(dataset: str | None = None, saved_path: Path | None = None) -> str:
    """Fail closed if RDKit version or descriptor schema does not match the pin."""
    ver = str(rdBase.rdkitVersion)
    if ver not in PINNED_RDKIT_VERSIONS:
        raise RuntimeError(
            f"RDKit {ver} != pinned {PINNED_RDKIT_VERSION}. "
            "Install rdkit==2023.9.6 and set RDBASE to the venv rdkit package."
        )
    n_full = len(RDKIT_FULL_DESCRIPTOR_NAMES)
    if n_full != EXPECTED_RDKIT_N_FULL:
        raise RuntimeError(
            f"RDKit 2D full descriptor count {n_full} != {EXPECTED_RDKIT_N_FULL}. "
            "The environment is not the pinned RDKit 2023.09.6 list."
        )
    if dataset is None:
        return feature_schema_hash("DS1")
    hash_now = feature_schema_hash(dataset)
    selected = rdkit_selected_descriptor_names()
    path = default_schema_path(dataset) if saved_path is None else saved_path
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("descriptor_names") != list(selected):
            raise RuntimeError(
                f"Saved descriptor names/order at {path} do not match runtime selection. "
                "Refusing inference on a different RDKit descriptor schema."
            )
        expected_names = (
            ds1_rdkit_column_names() if dataset.upper() == "DS1" else ds2_rdkit_column_names()
        )
        if saved.get("feature_names") != expected_names:
            raise RuntimeError(f"Saved feature_names at {path} do not match runtime schema.")
        if saved.get("feature_hash") and saved["feature_hash"] != hash_now:
            raise RuntimeError(
                f"Feature hash mismatch: saved {saved['feature_hash']} vs runtime {hash_now}. "
                "Refusing inference on a different descriptor schema."
            )
    return hash_now


def assert_manifest_matches_runtime(manifest: dict) -> None:
    dataset = str(manifest.get("dataset", "DS1"))
    hash_now = assert_runtime_schema(dataset)
    if manifest.get("descriptor_names") != rdkit_selected_descriptor_names():
        raise RuntimeError("Fold manifest descriptor_names do not match runtime schema.")
    expected_names = (
        ds1_rdkit_column_names() if dataset.upper() == "DS1" else ds2_rdkit_column_names()
    )
    if manifest.get("feature_names") != expected_names:
        raise RuntimeError("Fold manifest feature_names do not match runtime schema.")
    if int(manifest["feature_dimension"]) != len(expected_names):
        raise RuntimeError(
            f"Fold manifest feature_dimension {manifest['feature_dimension']} "
            f"!= schema {len(expected_names)}"
        )
    saved_hash = manifest.get("feature_hash")
    if saved_hash and saved_hash != hash_now:
        raise RuntimeError(
            f"Fold manifest feature_hash {saved_hash} != runtime {hash_now}. "
            "Refusing inference on a different descriptor schema."
        )
