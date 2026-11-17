"""Parse CMI complex SMILES into ligand + metal centre.

Convention (validated on this corpus): the ligand is the largest organic
fragment by heavy-atom count. If no organic fragment exists (halide, borate,
…), the largest remaining non-metal fragment is used.

Proton tokens are exact ``[H+]`` / ``[H]`` / ``H+`` only. Tokens such as
``[Hg+2]``, ``[Ho+3]``, ``[Hf+4]`` are metals, not protons.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from rdkit import Chem

ORGANIC_ATOMS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Se", "B", "Si", "H"}

# Ligand ions / non-metal centres — never treated as the complexing metal.
LIGAND_ION_ELEMENTS = {"F", "Cl", "Br", "I", "O", "S", "N", "P", "C", "Se", "Te", "H", "B", "Si"}

METAL_TOKEN_RE = re.compile(r"^\[([A-Z][a-z]?)([0-9]*[\+\-]|[\+\-][0-9]*)\]$")
PROTON_TOKENS = {"[H+]", "[H]", "H+", "[H+1]", "[H1+]"}
_PROTON_RE = re.compile(r"^(?:\[H\+\]|\[H\]|H\+|\[H\+1\]|\[H1\+\])$")


def is_proton_token(token: str) -> bool:
    p = str(token).strip()
    return p in PROTON_TOKENS or bool(_PROTON_RE.match(p))


def _is_organic_fragment(mol: Chem.Mol) -> bool:
    return all(atom.GetSymbol() in ORGANIC_ATOMS for atom in mol.GetAtoms())


def _parse_oxidation(charge_str: str) -> float:
    if charge_str.endswith("+") or charge_str.endswith("-"):
        sign = 1 if charge_str.endswith("+") else -1
        num = charge_str[:-1]
        return float(num) * sign if num else float(sign)
    if charge_str.startswith("+") or charge_str.startswith("-"):
        return float(charge_str)
    return float(charge_str) if charge_str else float("nan")


def extract_ligand_and_metal(complex_smiles: str) -> dict[str, Any]:
    """Parse CMI ``ligand.metal`` SMILES. Never returns an empty ligand silently."""
    parts = [p.strip() for p in str(complex_smiles).split(".") if p.strip()]
    metal_tokens: list[str] = []
    proton_tokens: list[str] = []
    ligand_mols: list[Chem.Mol] = []
    failed_parts: list[str] = []

    for p in parts:
        if is_proton_token(p):
            proton_tokens.append(p)
            continue
        m = METAL_TOKEN_RE.match(p)
        if m:
            el = m.group(1)
            if el not in LIGAND_ION_ELEMENTS:
                metal_tokens.append(p)
                continue
            # halide / ligand ion — treat as a ligand fragment
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            failed_parts.append(p)
            continue
        ligand_mols.append(mol)

    organic = [m for m in ligand_mols if _is_organic_fragment(m)]
    pool = organic if organic else ligand_mols
    n_discarded_organic = max(len(organic) - 1, 0) if organic else 0

    ligand_smiles = ""
    n_heavy_best = 0
    if pool:
        best = max(pool, key=lambda mol: mol.GetNumHeavyAtoms())
        ligand_smiles = Chem.MolToSmiles(best, canonical=True)
        n_heavy_best = int(best.GetNumHeavyAtoms())

    unique_metals = []
    for tok in metal_tokens:
        mm = METAL_TOKEN_RE.match(tok)
        if mm:
            unique_metals.append(mm.group(1))
    unique_metals = list(dict.fromkeys(unique_metals))

    is_proton = bool(proton_tokens) and not unique_metals
    metal_token = ""
    metal_el = ""
    metal_ox = np.nan

    if unique_metals:
        # Prefer the metal centre over a co-occurring proton token.
        metal_token = metal_tokens[0]
        mm = METAL_TOKEN_RE.match(metal_token)
        metal_el = mm.group(1) if mm else ""
        metal_ox = _parse_oxidation(mm.group(2)) if mm else np.nan
        is_proton = False
    elif proton_tokens:
        metal_token = "[H+]"
        metal_el = "H"
        metal_ox = 1.0
        is_proton = True

    parse_status = "ok"
    if failed_parts and not ligand_smiles:
        parse_status = "unparseable_ligand"
    elif not ligand_smiles:
        parse_status = "empty_ligand"
    elif not metal_el:
        parse_status = "missing_metal"
    elif len(unique_metals) > 1:
        parse_status = "multi_metal"

    return {
        "ligand_smiles": ligand_smiles,
        "metal": metal_token,
        "metal_el": metal_el,
        "metal_oxidation_state": metal_ox,
        "is_proton": bool(is_proton),
        "n_dot_parts": len(parts),
        "n_metal_tokens": len(metal_tokens),
        "n_proton_tokens": len(proton_tokens),
        "n_ligand_fragments": len(ligand_mols),
        "n_organic_fragments": len(organic),
        "n_discarded_organic_fragments": n_discarded_organic,
        "n_failed_parts": len(failed_parts),
        "n_heavy_atoms_ligand": n_heavy_best,
        "failed_parts": "|".join(failed_parts),
        "metal_tokens": "|".join(metal_tokens),
        "parse_status": parse_status,
    }
