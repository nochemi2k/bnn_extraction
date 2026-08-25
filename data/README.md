# Data

Ready-to-use tables:

- `processed/ds1.csv` — DS1 metal-complex log *K* records
- `processed/ds2.csv` — DS2 matched metal–metal selectivity rows
- `processed/metal_property_table.json` — element/oxidation-state properties used in the joint RDKit vector
- `processed/features/` — fold-local RDKit matrices, Chemprop CSVs, and schema/manifest JSON
- `processed/audits/prepare_data_findings.json` — data-preparation findings cited by the SI

Splits:

- `splits/project_folds.json` — DS1 five ligand-disjoint folds
- `splits/ds2_project_folds.json` — DS2 five ligand-disjoint folds
- `splits/cmi_official_split.json` — historical CMI official split (SI S2.2)

`raw_m4/` holds the CMI train/validation/test CSVs used by optional `python scripts/prepare_data.py`. The supported release path loads the processed tables above and does not require rebuilding them.
