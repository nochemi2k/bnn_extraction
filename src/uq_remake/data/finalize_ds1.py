"""
Finalize DS1 definition artefacts from the H+ decision and replicate summary.
Requires: replicate_summary, cmisplit results, hplus_decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uq_remake import REMAKE_ROOT


def main() -> None:
    df = pd.read_csv(REMAKE_ROOT / "data" / "processed" / "m4_processed.csv")
    hplus_path = REMAKE_ROOT / "results" / "canonical" / "hplus_decision.json"
    if not hplus_path.exists():
        hplus_path = REMAKE_ROOT / "results" / "stage1" / "hplus_decision.json"
    decision = json.loads(hplus_path.read_text(encoding="utf-8"))
    hplus = decision["decision"]

    if hplus == "train_without_H":
        ds1 = df[df["family"] != "protonation"].copy()
        train_policy = "Protonation (H+) records excluded from training and from DS1."
    else:
        ds1 = df.copy()
        train_policy = "Protonation (H+) records kept in training and in DS1; test metrics remain metal_complex-only."

    ds1_path = REMAKE_ROOT / "data" / "processed" / "ds1.csv"
    ds1.to_csv(ds1_path, index=False)

    composition = {
        "dataset": "DS1",
        "hplus_decision": hplus,
        "train_policy": train_policy,
        "n_records": int(len(ds1)),
        "n_unique_ligands": int(ds1["ligand_smiles"].nunique()),
        "n_metals": int(ds1.loc[ds1["family"] == "metal_complex", "metal_el"].nunique()),
        "n_species_keys": int(ds1["species_key"].nunique()),
        "family_breakdown": ds1["family"].value_counts().to_dict(),
        "test_restriction": "metal_complex only for Stage 1b evaluation",
        "test_restriction_justification": (
            "Protonation-only constants answer a different chemical question (ligand pKa-like "
            "protonation) than metal–ligand stability; mixing them into the test metric "
            "inflates apparent accuracy without validating metal-complex prediction. "
            "H+ may still appear in training per the Chemprop-only decision above."
        ),
    }
    payload = json.dumps(composition, indent=2)
    (REMAKE_ROOT / "results" / "stage1").mkdir(parents=True, exist_ok=True)
    (REMAKE_ROOT / "results" / "stage1" / "ds1_composition.json").write_text(payload, encoding="utf-8")
    canon = REMAKE_ROOT / "results" / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    (canon / "ds1_composition.json").write_text(payload, encoding="utf-8")
    pd.DataFrame([composition]).to_csv(REMAKE_ROOT / "results" / "stage1" / "ds1_composition_table.csv", index=False)

    # Interpretation stub filled from available numbers
    rep = json.loads((REMAKE_ROOT / "results" / "stage1" / "replicate_summary.json").read_text(encoding="utf-8"))
    s1a = {}
    s1a_path = REMAKE_ROOT / "results" / "stage1" / "stage1a_cmisplit_results.json"
    if s1a_path.exists():
        s1a = json.loads(s1a_path.read_text(encoding="utf-8"))

    interpretation = f"""
## DS1 definition notes (not manuscript prose)

Noise floor: multi-replicate experiment keys have median replicate_spread={rep.get('replicate_spread_median_multi'):.3f} \
and hard_disagreement on {rep.get('hard_disagreement_key_fraction', float('nan')):.1%} of those keys \
(protonation hard rate={rep['hard_disagreement_rate_by_family'].get('protonation')}, \
metal_complex hard rate={rep['hard_disagreement_rate_by_family'].get('metal_complex')}). \
The empirical aleatoric_noise_floor (median within-key σ on agree+soft keys) is {rep.get('aleatoric_noise_floor'):.3f}. \
Absolute-logK variance is large, so a naive R² ceiling from agree+soft noise alone is high; nevertheless, \
hard conflicts and ligand-disjoint evaluation mean that headline R²≈0.95 is not a trustworthy decision metric \
without UQ — point predictions ignore input-dependent aleatoric structure.

H+ decision: {hplus} (Chemprop-only). Applied uniformly to DS1/DS2 and to Chemprop training.

Embeddings: project_fold Chemprop embeddings under embeddings/project_fold/; cmisplit embeddings under embeddings/cmisplit/.
""".strip()
    canon = REMAKE_ROOT / "results" / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    (canon / "DS1_DEFINITION_NOTES.md").write_text(interpretation + "\n", encoding="utf-8")

    print(json.dumps(composition, indent=2))
    print("Wrote DS1 to", ds1_path)


if __name__ == "__main__":
    main()
