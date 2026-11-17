# Packed artifacts (Git LFS)

Large working trees are **not** committed unpacked. They are 7z archives under
`artifacts/`, tracked with Git LFS (`artifacts/*.7z` in `.gitattributes`). The unpacked
directories are gitignored.

## Which archive to fetch

### Weights and predictions

| Archive | Unpack target | Required for |
|---------|---------------|--------------|
| `checkpoints.7z` | `checkpoints/active/` | Re-evaluation or re-inference from the trained active grid |
| `cv_predictions.7z` | `results/cv_predictions/` | Skipping fold re-inference (`evaluate.py --from-predictions`) |
| `checkpoints_ad_hoc.7z` | `checkpoints/ad_hoc_comparison/` | Fold-matched DS1 weights for the composed-versus-direct comparison |
| `ad_hoc_foldmatched_results.7z` | `results/downstream_cv/ad_hoc_selectivity_foldmatched/` | Composed-model metrics without retraining |
| `ad_hoc_foldmatched_features.7z` | `data/processed/features/ad_hoc_foldmatched/` | Resuming fold-matched Chemprop training |

### Derived data and row-level results

| Archive | Unpack target | Required for |
|---------|---------------|--------------|
| `data_processed.7z` | `data/processed/` tables | Training or evaluation without re-running `prepare_data` |
| `data_features.7z` | `data/processed/features/{rdkit,chemprop}_*` | Training and re-inference |
| `ad_retention_scores.7z` | `results/downstream_cv/ad_retention/` row scores | Re-deriving retention curves |
| `downstream_rowlevel.7z` | `results/downstream_cv/` row-level dumps | Re-deriving downstream summaries from row level |

**What stays in git:** `data/raw_m4/` (the raw input), `data/splits/*.json` (the fold
definitions), every summary CSV/JSON in `results/`, and all code. Everything else is
either regenerable from the raw tables or packed above.

## Unpack (after clone)

```powershell
git lfs install
git lfs pull
powershell -File scripts/unpack_artifacts.ps1
```

Selective fetch (saves bandwidth):

```powershell
git lfs pull -I artifacts/checkpoints.7z
git lfs pull -I artifacts/data_features.7z
```

Every archive stores **paths from the repository root** and extracts with `-o<repo-root>`.
Do not extract into `checkpoints/` a second time or you will nest
`checkpoints/checkpoints/active`. The script refuses to extract a file under 1 KB, which
is how an unfetched LFS pointer presents itself.

## Gitignore contract

Unpacked trees stay ignored: `checkpoints/active/`, `checkpoints/ad_hoc_comparison/`,
`results/cv_predictions/`, `results/downstream_cv/ad_hoc_selectivity_foldmatched/`,
`data/processed/features/ad_hoc_foldmatched/`, plus the derived data trees listed above.
The `artifacts/*.7z` files are the LFS pointers you commit.
