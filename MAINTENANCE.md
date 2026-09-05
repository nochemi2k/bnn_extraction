# Repository maintenance notes

**The Nature Computational Science submission was pushed to `main` on 5 September 2026.**
These notes were written to get that push done; they are kept because three of the findings
are standing conditions rather than one-off steps, and because the reproduction paths in
them are still the ones to follow.

## Still live

**Git LFS filters are not configured in this clone.** `git config --get filter.lfs.clean`
returns nothing, locally and globally, so the smudge and clean filters never run. The
symptom is that all nine `artifacts/*.7z` show as modified in `git status` when nothing has
changed: HEAD holds a 132-byte pointer, the working tree holds the real archive, and without
the clean filter git compares the two directly. **Do not `git add` those archives in this
state** — you would commit an 840 MB blob and GitHub would reject the push at its 100 MB
limit. Fix it once with:

```bat
git lfs install
git checkout -- artifacts
git status --short
```

The last line should then be silent for `artifacts/`.

**The LFS quota question is unresolved.** The nine archives total 1.4 GiB against a free
allowance of 1 GiB storage and 1 GiB bandwidth a month, and `checkpoints.7z` alone is
801 MiB. The recommendation below still stands.

**Local disk.** `.git` was about 66 GB, of which 62 GB is the LFS object cache and 1.43 GiB
is collectable garbage. `git lfs prune` and `git gc --prune=now` reclaim nearly all of it and
touch nothing on the remote.

---

## Committing the 5 September documentation pass

The pass that followed the submission push moved the derived tables out of the JCIM-era
folder, renamed two modules, and rewrote the repository documentation. 51 files modified,
3 deleted, 4 new.

**Install the LFS filters first.** Until this is done the nine `artifacts/*.7z` show as
modified when nothing has changed, and a blanket `git add -A` would stage an 840 MB blob
that GitHub rejects.

```bat
git lfs install
git status --short | findstr artifacts
```

If that prints only `M artifacts/README.md`, the filters are working. If the `.7z` lines are
still there, leave them alone — the staging below never touches them.

**Stage by path, not with a blanket `-A`.**

```bat
git add -A manuscripts src
git add README.md REPO_COMPOSITION.md CITATION.cff MAINTENANCE.md artifacts/README.md
git add -A PUSH_INSTRUCTIONS.md
```

The third line stages the removal of the retired file. Then confirm no archive slipped in —
this must print nothing:

```bat
git diff --cached --name-only | findstr ".7z"
```

**Commit and push.**

```bat
git commit -m "Point the repository at the Nature Computational Science submission" -m "Moves the derived tables from manuscripts/jacs/tables to manuscripts/derived/tables, so a fresh clone can rebuild every figure and run every validator. Renames jacs_common.py to manuscript_common.py and validate_jacs_analyses.py to validate_core_analyses.py and repoints the 23 modules that referenced them. Rewrites README, REPO_COMPOSITION and CITATION.cff around the current submission; retires PUSH_INSTRUCTIONS.md in favour of MAINTENANCE.md. Validation unchanged: 56 + 22 + 186 checks pass."

git push
```

You are on `main` and level with `origin/main`, so this is a fast-forward with no branch or
pull request needed.

**Two things in `git status` that are not problems.** The figure `.pdf` files are listed as
modified while the matching `.png` files are not, because matplotlib stamps a creation date
into PDF metadata; the drawn content is identical. And `manuscripts/jacs/` is now an empty
directory — git does not track empty directories, so it will not be committed, but it is
still on your disk and `rmdir manuscripts\jacs` clears it.

## Git LFS: what is stored and what it costs

```bat
git lfs version
git lfs env
```

Then count how many LFS files the history holds — **cmd**:

```bat
git lfs ls-files --all | find /c /v ""
```

**PowerShell**:

```powershell
git lfs ls-files --all | Measure-Object -Line
```

Then open **GitHub → Settings → Billing → Git LFS Data** and read the current
storage and bandwidth figures.

This matters because of the sizes:

| archive | size | previously committed |
|---|---|---|
| `checkpoints.7z` | 801 MiB | yes (3 commits) |
| `checkpoints_ad_hoc.7z` | 397 MiB | yes |
| `cv_predictions.7z` | 71 MiB | yes |
| `data_features.7z` | 59 MiB | yes |
| `ad_hoc_foldmatched_features.7z` | 49 MiB | yes |
| `ad_retention_scores.7z` | 32 MiB | yes |
| `downstream_rowlevel.7z` | 22 MiB | yes |
| `ad_hoc_foldmatched_results.7z` | 10 MiB | yes |
| `data_processed.7z` | 5 MiB | yes |
| **total** | **1.4 GiB** | |

GitHub's free allowance is 1 GiB of LFS storage and 1 GiB of bandwidth a month.
Because all nine were committed before, unchanged archives should not re-upload —
but you are near or past the free ceiling either way, and every historical
version still counts against storage.

**My recommendation, and it costs nothing extra:** the two checkpoint archives
are 1.17 GiB of the 1.4 GiB and they are model weights, which is exactly what a
deposition DOI is for. You need that DOI anyway for the manuscript's Data
availability statement. Put `checkpoints.7z` and `checkpoints_ad_hoc.7z` on
Zenodo, cite the DOI from `artifacts/README.md`, and leave the remaining seven
archives (247 MiB) in LFS, which fits the free allowance comfortably. If you
would rather keep everything in LFS, one data pack at $5/month covers 50 GiB of
storage and bandwidth and settles it.

Decide this before the archives are committed, since it changes what belongs in the repository.

---

## Reclaiming local disk

Local housekeeping only. Nothing here changes history or affects the remote.

```bat
git lfs prune
git gc --prune=now
git count-objects -vH
```

`git lfs prune` clears superseded objects from `.git/lfs/objects`; `git gc`
packs the 3,575 loose objects and drops the 1.43 GiB of garbage; the last
command confirms it, and should report `size-garbage: 0 bytes` with packs
present.

Expect `.git` to fall from about 66 GB to a few GB. `git lfs prune` keeps
anything referenced by a recent commit, so nothing you might still need goes.

---

## Verifying a checkout

**Nothing oversized slipped in.** Git itself will refuse anything over 100 MB
that is not an LFS pointer, but check the staged sizes yourself — this works in
both shells and prints one line per staged file over 5 MB, or nothing:

```bat
git diff --cached --name-only > "%TEMP%\staged.txt"
for /f "usebackq delims=" %f in ("%TEMP%\staged.txt") do @if exist "%f" for %s in ("%f") do @if %~zs GTR 5000000 echo %~zs  %f
```

In a `.bat` file rather than typed at the prompt, double each `%` to `%%`. The
PowerShell equivalent:

```powershell
git diff --cached --name-only | Where-Object { Test-Path $_ } |
  Get-Item | Where-Object { $_.Length -gt 5MB } | Select-Object Length, FullName
```

**The quarantine and scratch folders are not being committed** — expect no
output. **cmd**:

```bat
git status --short | findstr /i "_quarantine _to_delete Claude outputs"
```

**PowerShell**:

```powershell
git status --short | Select-String '_quarantine|_to_delete|Claude outputs'
```

**The submission still validates.** **cmd**:

```bat
set PYTHONPATH=src
python src\uq_remake\manuscript\validate_jacs_analyses.py
python src\uq_remake\manuscript\validate_scalefree_central.py
python src\uq_remake\manuscript\validate_manuscript_values.py
```

**PowerShell**:

```powershell
$env:PYTHONPATH="src"
python src\uq_remake\manuscript\validate_jacs_analyses.py
python src\uq_remake\manuscript\validate_scalefree_central.py
python src\uq_remake\manuscript\validate_manuscript_values.py
```

All three validators should exit 0, with 56, 22 and 186 checks.

---

## What is safe to delete

I moved things rather than deleting them, because the bridge cannot delete and
because you should look before anything goes. All of it is already gitignored,
so none of it can reach the push either way.

### Safe to delete now

| path | size | what it is |
|---|---|---|
| `_quarantine_20260905/claude_outputs_scratch/` | 2.3 MB | Copies of figures I sent you during drafting. Eight PNGs, all superseded by `manuscripts/natcompsci/figures/`. Two are from the old numbering (`figS4_raw_versus_normalized`, `figS7_embedding_retention`). |
| `_quarantine_20260905/natcompsci_figures_renumbered/` | 1.6 MB | The S4–S8 figures under their pre-renumbering names, replaced by S5–S9 when Fig. S4 was inserted. Regenerated by `build_si_figures.py`. |
| `_quarantine_20260905/jacs_abandoned_and_superseded/` | 2.8 MB | Two JCIM-era lines of work that were run and reported as not-reported (`external_set_emulation`, `single_pair_campaign`, the divalent-3d σ table), plus superseded JCIM figure drafts. The analyses are described in Supplementary S9.3; the modules that produce them stay in `src/`. |
| `manuscripts/natcompsci/figures/_to_delete/` | empty | Left behind because the bridge could not remove the directory. |
| `.git/index.lock` | 0 bytes | Handled in Step 0. |

Plus `_quarantine_20260905/jacs_prose_draft/` and `renamed_modules/`, added when the JCIM
material was retired. Delete the lot with:

**cmd**

```bat
rmdir /s /q _quarantine_20260905
rmdir /s /q manuscripts\natcompsci\figures\_to_delete
```

**PowerShell**

```powershell
Remove-Item -Recurse -Force _quarantine_20260905, manuscripts\natcompsci\figures\_to_delete
```

### Decide, do not just delete

- **`_deprecated/`** — the previous cleanup's quarantine, with its own
  `MANIFEST.md` and a dry-run `remove_originals.ps1`. It is gitignored and was
  never pushed. Read the manifest and run the script when you are ready; that is
  a separate decision from this pass and I did not touch it.
- **`.venv311/`** — a virtual environment inside the repository folder.
  Gitignored, so harmless to the push, but it does not belong here. Recreate it
  outside the repo with `requirements.txt` when convenient.
- **`manuscripts/jacs/`** — the JCIM draft. Untracked today. Keep it if you want
  the earlier submission in the record; if not, move it to `_quarantine_20260905/`
  rather than deleting, since `manuscripts/natcompsci/` cites its `tables/`
  directory for the derived CSVs.

### Deliberately left alone

- **`manuscripts/final/`** — the frozen JCIM submission. The Nature Computational
  Science supplementary reads several tables straight out of it
  (`iz_risk_score_spearman.csv`, the pair-acquisition tables,
  `iz_duplicate_measurement_summary.csv`, `iz_mahalanobis_feature_median_scores.csv`),
  and the validators assert against them. Nothing in it was moved or edited.
- **`src/uq_remake/manuscript/`** — I checked all 35 modules against the
  manuscripts, the SI, the README and the import graph. Every one is either cited
  in a document or imported by another module, including the JCIM-era generators,
  which are the provenance for the frozen tables above. Nothing there is
  provably dead, so I quarantined none of it. If you want that trimmed it needs a
  decision about whether the JCIM rebuild path stays reproducible, and that is
  yours to make rather than mine to guess.

---

## If a future push fails

**`this exceeds GitHub's file size limit of 100.00 MB`** — an archive was committed as a
real blob instead of an LFS pointer, which is what happens when the LFS filters are not
installed (see **Still live**). Fix with
`git rm --cached artifacts\<name>.7z`, confirm `.gitattributes` is committed,
then re-add.

**`batch response: This repository is over its data quota`** — the LFS storage
ceiling. Go back to the Zenodo option in Step 1, or buy a data pack.

**The push hangs partway through an LFS upload** — resume with `git lfs push --all origin main`,
which skips objects the server already holds.
