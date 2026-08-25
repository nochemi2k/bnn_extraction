# Checkpoints

Unpacked study weights live in `active/<MODEL_ID>/fold*/member_*/`.

- Torch models: `model.pt`
- XGBoost: `model.json`
- Chemprop: `fold_0/model_0/model.pt` plus Chemprop workdir metadata
- Exact GPR (`DS1_RDKIT_GPR`, `DS2_RDKIT_GPR`): `infeasible.json` only (computationally infeasible; sparse GPR is the trained GP family)

If `active/` is missing after clone, unpack `artifacts/checkpoints.7z` at the **repository root** (`scripts/unpack_artifacts.ps1 -Active`). Fold-matched Table 2 / S13 comparison weights unpack from `artifacts/checkpoints_ad_hoc.7z` into `ad_hoc_comparison/`. Do not extract a `checkpoints/...` archive into `checkpoints/` a second time.

Published CMI M4 `best_model` weights are downloaded into `cmi_published/` by `run_cmi_reproduction.py`. The architecture-matched retrain writes `M4_GRAPH_DMPNN_POINT_cmiarch/` (gitignored); Table 1 numbers are stored in `results/stage1/cmi_reproduction_results.json`.
