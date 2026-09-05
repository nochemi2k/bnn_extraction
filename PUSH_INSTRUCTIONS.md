# Getting this repository pushed

Written 5 September 2026, after the Nature Computational Science rework. Follow
the steps in order; each one says what it is for and how to tell it worked.

Everything below is run from the repository root, `C:\Users\IVan\Desktop\BNN Extraction`.

**Which shell you are in matters for a handful of these commands.** If your
prompt looks like `C:\Users\IVan\Desktop\BNN Extraction>` you are in Command
Prompt; if it starts with `PS ` you are in PowerShell. Every `git` command below
is identical in both. Where a command is not, both versions are given, marked
**cmd** and **PowerShell**. If you would rather have one set, type `powershell`
and press Enter to switch the current window, then use the PowerShell lines
throughout.

---

## What state the repository is actually in

Worth reading before you start, because three of these were not obvious.

| | |
|---|---|
| Working tree | 475 tracked files, **48 MB**. Largest tracked file 5.9 MB. Nothing here is near GitHub's 100 MB blob limit. |
| `.git/objects` | **3.5 GB**, none of it packed: 3,575 loose objects and 1.43 GiB the garbage collector has already flagged. This repository has never been `gc`'d. |
| `.git/lfs/objects` | **62 GB** in 644 objects, many of them 600–840 MB. These are superseded versions of the packed archives, kept in the local cache. Local only; `git lfs prune` clears them. |
| `artifacts/*.7z` | 1.5 GB in nine archives. All nine have been committed before, so their LFS objects should already exist on the remote and re-committing unchanged bytes uploads nothing. |
| Branch state | `HEAD` is level with `origin/main` (0 ahead, 0 behind). Nothing local is waiting to go out. |
| Untracked | `manuscripts/natcompsci/`, `manuscripts/jacs/`, and 20 new modules under `src/uq_remake/manuscript/`. This is the whole of the new work. |

**Three problems were found and two are already fixed.**

1. **`.gitignore` silently disabled Git LFS.** The un-ignore rules for
   `artifacts/**/*.7z` were followed, forty lines later, by a bare
   `artifacts/*.7z`. Last match wins, so every archive was ignored and none was
   ever handed to LFS. *Fixed* — the stray line is gone and a comment says why it
   must not come back. `git check-ignore -v artifacts/checkpoints.7z` now reports
   the un-ignore rule instead of the ignore rule.

2. **369 files reported as modified when nothing had changed.** `.gitattributes`
   had no `text` rule, so the index held LF and the Windows working tree held
   CRLF. `git diff --ignore-all-space` on any of them was empty. *Fixed* —
   `.gitattributes` now starts with `* text=auto eol=lf`, with `eol=crlf` kept
   for `.ps1`, `.bat` and `.cmd`. Step 2 below makes the fix take effect.

3. **`.git/index.lock` is stale** and is blocking every write operation. I could
   not remove it from here; the bridge is not permitted to delete files. **You
   have to do this one first.**

---

## Written before the last manuscript pass, still current

These instructions were drafted before the supplementary information was
reorganised (Methods moved to S1, everything renumbered, cross-references
audited). That pass changed three files inside `manuscripts/natcompsci/` and
`src/uq_remake/manuscript/validate_manuscript_values.py`, all of which Step 4
adds wholesale, so nothing below needs adjusting. The validator counts in Step 5
are unchanged at 56, 22 and 186.

---

## Step 0 — clear the stale lock

**cmd**

```bat
del ".git\index.lock"
git status
```

**PowerShell**

```powershell
Remove-Item .git\index.lock
git status
```

`git status` should now run without complaining. If it says the lock exists
again, a Git process is genuinely running: close VS Code and any Git GUI first.

---

## Step 1 — confirm LFS is installed and see where you stand on quota

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

Whichever you choose, do it *before* Step 4, since it changes what gets added.

---

## Step 2 — make the line-ending fix take effect, as its own commit

Do this on its own so the whitespace churn never mixes with content.

```bat
git add --renormalize .
git commit -m "Normalise line endings; add .gitattributes text and binary rules"
```

Before committing, count the staged files — expect roughly 370. **cmd**:

```bat
git status --short | find /c /v ""
```

**PowerShell**:

```powershell
git status --short | Measure-Object -Line
```

Sanity check before you commit — this should print nothing at all:

```powershell
git diff --cached --ignore-all-space --stat
```

If it prints file names, something other than line endings changed and you
should look at it before committing.

---

## Step 3 — reclaim the 65 GB of local disk

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

## Step 4 — add the new work

```bat
git checkout -b natcompsci-submission

git add manuscripts/natcompsci
git add src/uq_remake/manuscript
git add .gitignore .gitattributes PUSH_INSTRUCTIONS.md
git add manuscripts/jacs
git add artifacts
```

The last two lines are optional: `manuscripts/jacs` only if you want the JCIM
draft in the record, and `artifacts` only for the archives you decided in Step 1
to keep in LFS.

Then check that LFS actually caught the archives, which is the thing that was
broken:

```bat
git lfs status
git diff --cached --stat -- artifacts
```

Each archive should appear as a small pointer, not as megabytes of content. If
`git lfs status` lists nothing under artifacts, stop and re-check `.gitattributes`.

---

## Step 5 — verify before committing

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

## Step 6 — commit and push

A multi-line message inside one pair of quotes works in PowerShell but not in
Command Prompt, so use one `-m` per paragraph. This form works in **both**:

```bat
git commit -m "Nature Computational Science submission: manuscript, SI, figures and analysis modules" -m "Adds the reframed manuscript and supplementary information, ten supplementary figures, and the modules behind the new analyses (noise floor, benchmark context, embedding retention). Validation: 264 assertions across three modules." -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GsbWuPErNXVmG65n8gNwQY"

git push -u origin natcompsci-submission
```

Watch the LFS upload line. If it starts pushing hundreds of megabytes you did
not expect, cancel with Ctrl-C, and re-check which archives you staged in Step 4.

Then open the pull request:

```bat
gh pr create --fill --base main
```

---

## Step 7 — what to delete once you are satisfied

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

That is **6.7 MB** in total. Delete the lot with:

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

## If the push fails

**`this exceeds GitHub's file size limit of 100.00 MB`** — an archive was
committed as a real blob instead of an LFS pointer, which means `.gitattributes`
was not in place when it was staged. Fix with
`git rm --cached artifacts\<name>.7z`, confirm `.gitattributes` is committed,
then re-add.

**`batch response: This repository is over its data quota`** — the LFS storage
ceiling. Go back to the Zenodo option in Step 1, or buy a data pack.

**The push hangs partway through an LFS upload** — resume with
`git lfs push --all origin natcompsci-submission`, which skips objects the
server already holds.
