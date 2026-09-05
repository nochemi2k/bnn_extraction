# Packed artifacts (Git LFS)

Large working trees are **not** committed unpacked. They are 7z archives under `artifacts/`, tracked with Git LFS (`artifacts/*.7z` in `.gitattributes`). Unpacked directories are gitignored.

Do **not** `git add .`. 7-Zip can write `*.7z.tmp` beside the destination; `scripts/pack_artifacts.ps1` writes temp files under `_archive_tmp/` instead. GitHub rejects git blobs over 100 MB. A 1.7 GB `checkpoints.7z.tmp` in history will block `git push` even if the current tree deleted it.

## Which archive to fetch

### Weights and predictions

| Archive | Unpack target | Required for | Skip if |
|---------|---------------|--------------|---------|
| `checkpoints.7z` | `checkpoints/active/` | Reload active-grid weights, `validate_release`, re-inference | You only need stored `results/` metrics |
| `cv_predictions.7z` | `results/cv_predictions/` | Skip fold re-inference (`evaluate_cv --from-predictions`) | You will re-infer or you already have `results/cv_metrics/` |
| `checkpoints_ad_hoc.7z` | `checkpoints/ad_hoc_comparison/` | fold-matched DS1 weights for the composed-versus-direct comparison | You will retrain with `run_ad_hoc_foldmatched.py` |
| `ad_hoc_foldmatched_results.7z` | `results/downstream_cv/ad_hoc_selectivity_foldmatched/` | composed-model metrics and DS1 predictions, without retraining | You will re-run `--stage eval` from local weights |
| `ad_hoc_foldmatched_features.7z` | `data/processed/features/ad_hoc_foldmatched/` | Resume fold-matched Chemprop / SI training | You will re-run `--stage prepare` |

### Derived data and row-level results

These four were previously committed unpacked, about **640 MB of CSV and NPZ in plain git**. They are all rebuildable from `data/raw_m4/` plus the packed weights, and none of them is a manuscript source file: every number the paper displays comes from the small summary CSVs that remain in git.

| Archive | Unpack target | Required for | Skip if |
|---------|---------------|--------------|---------|
| `data_processed.7z` | `data/processed/ds1.csv`, `ds2.csv`, `m4_processed.csv`, `audits/ligand_parse_audit.csv` (~76 MB raw) | Any training, evaluation or feature rebuild | You will re-run `prepare_data` from `data/raw_m4/` |
| `data_features.7z` | `data/processed/features/{rdkit_ds1,rdkit_ds2,chemprop_ds1,chemprop_ds2,chemprop_cmisplit}` (~185 MB raw) | Training, re-inference, and the frozen-bundle figure generator, which recomputes the median-Mahalanobis score from the fold-local matrices | You will re-run `prepare_data` |
| `ad_retention_scores.7z` | `results/downstream_cv/ad_retention/fold_scores/`, `fold_retention.csv` (~207 MB raw) | the retention curves and the score-error screen (main-text Fig. 2; SI S6-S7) | You will re-run `run_downstream_cv --rescore` |
| `downstream_rowlevel.7z` | `results/downstream_cv/ad/ad_fold_scores.csv`, `ad_hoc_selectivity/ds1_heldout_preds/`, both `pair_acquisition_fold_metrics.csv` (~170 MB raw) | Re-deriving pair-acquisition and ad hoc summaries from row level | You only need the summary CSVs already in git |

**What stays in git:** `data/raw_m4/` (the actual raw input), `data/splits/*.json` (the fold definitions — load-bearing and only ~10 MB), every `results/**/summary` and fold-level CSV the manuscript cites, `manuscripts/natcompsci/` and `manuscripts/derived/` in full, `manuscripts/final/` in full, and all code.

**Not packed (excluded on purpose):**

- `checkpoints.7z.tmp` and any incomplete 7z sidecar
- CMI architecture retrain dumps (`checkpoints/M4_GRAPH_DMPNN_POINT_cmiarch/`, `data/processed/features/chemprop_cmi_official/`)
- logs, TensorBoard `events.out.tfevents*`, `chemprop_raw_preds.csv`, `__pycache__`

Each archive is a separate LFS object, so an 800 MB active-grid pack is not mixed with the fold-matched comparison weights or the derived tables.

## Unpack (clone)

```powershell
git lfs install
git lfs pull
powershell -File scripts/unpack_artifacts.ps1
```

Default unpack: active-grid weights, fold-matched ad hoc weights/results, and all four derived trees — i.e. everything needed to run the pipeline or rebuild the paper.

```powershell
powershell -File scripts/unpack_artifacts.ps1 -Rebuild   # tables + figures only, no weights
powershell -File scripts/unpack_artifacts.ps1 -Weights   # weights and predictions only
powershell -File scripts/unpack_artifacts.ps1 -Bulk      # the four derived trees only
powershell -File scripts/unpack_artifacts.ps1 -All
```

Every archive stores **paths from the repository root**. The script always extracts with `-o<repo-root>`. Do not extract into `checkpoints/` a second time or you will nest `checkpoints/checkpoints/active`. The script refuses to extract a file under 1 KB, which is how an unfetched LFS pointer presents itself.

Selective LFS (saves bandwidth):

```powershell
git lfs pull -I artifacts/checkpoints.7z
git lfs pull -I artifacts/ad_retention_scores.7z
git lfs pull -I artifacts/data_features.7z
```

## Re-pack

Run only when training / eval jobs are finished writing:

```powershell
powershell -File scripts/pack_artifacts.ps1            # everything
powershell -File scripts/pack_artifacts.ps1 -Bulk      # just the four derived trees
powershell -File scripts/pack_artifacts.ps1 -Weights   # just checkpoints / predictions

git add artifacts/*.7z artifacts/README.md .gitattributes .gitignore
git lfs push origin HEAD
git push
```

### First time: removing the originals from the index

The four derived trees were tracked before. Adding them to `.gitignore` does not untrack them — you must remove them from the index once, without deleting them from disk:

```powershell
powershell -File scripts/pack_artifacts.ps1 -Bulk

git rm -r --cached data/processed/features/rdkit_ds1 data/processed/features/rdkit_ds2 `
                   data/processed/features/chemprop_ds1 data/processed/features/chemprop_ds2 `
                   data/processed/features/chemprop_cmisplit `
                   results/downstream_cv/ad_retention/fold_scores `
                   results/downstream_cv/ad_hoc_selectivity/ds1_heldout_preds
git rm --cached data/processed/ds1.csv data/processed/ds2.csv data/processed/m4_processed.csv `
                data/processed/audits/ligand_parse_audit.csv `
                results/downstream_cv/ad_retention/fold_retention.csv `
                results/downstream_cv/ad/ad_fold_scores.csv `
                results/downstream_cv/pair_acquisition/pair_acquisition_fold_metrics.csv `
                results/downstream_cv/pair_acquisition_tau_2_0p5/pair_acquisition_fold_metrics.csv

git status                      # confirm only deletions-from-index and the new .7z files
git add artifacts/*.7z .gitignore artifacts/README.md scripts/pack_artifacts.ps1 scripts/unpack_artifacts.ps1
git commit -m "Move derived data and row-level results into Git LFS archives"
git lfs push origin HEAD
git push
```

This shrinks the working tree that future clones download, but it does **not** shrink history — the old blobs stay in past commits. If you need the history slimmed too, that is a `git filter-repo` rewrite and a force-push, which is a separate decision.

## Gitignore contract

Unpacked trees stay ignored: `checkpoints/active/`, `checkpoints/ad_hoc_comparison/`, `results/cv_predictions/`, `results/downstream_cv/ad_hoc_selectivity_foldmatched/`, `data/processed/features/ad_hoc_foldmatched/`, plus the four derived trees listed above. The `artifacts/*.7z` files are the LFS pointers you commit.
