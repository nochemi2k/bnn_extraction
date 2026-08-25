# Pack large working trees into Git LFS 7z archives.
#
# Every archive stores paths from the repository root, so unpack_artifacts.ps1 can always
# extract with -o<repo-root>. Temp files go under _archive_tmp/ so artifacts/*.7z.tmp is
# never created next to the destination (GitHub rejects git blobs over 100 MB, and a
# stray .7z.tmp in history blocks push forever).
#
# Do not run this while Chemprop is writing into ad_hoc_comparison/.
param(
    [switch]$Active,
    [switch]$Predictions,
    [switch]$AdHocCheckpoints,
    [switch]$AdHocResults,
    [switch]$AdHocFeatures,
    [switch]$DataProcessed,
    [switch]$DataFeatures,
    [switch]$AdRetentionScores,
    [switch]$DownstreamRowLevel,
    [switch]$Weights,          # convenience: the three checkpoint/prediction packs
    [switch]$Bulk,             # convenience: the four data/results packs
    [switch]$All
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$SevenZip = Join-Path ${env:ProgramFiles} "7-Zip\7z.exe"
if (-not (Test-Path $SevenZip)) {
    throw "7-Zip not found at $SevenZip"
}

if ($Weights) { $Active = $true; $Predictions = $true; $AdHocCheckpoints = $true; $AdHocResults = $true; $AdHocFeatures = $true }
if ($Bulk)    { $DataProcessed = $true; $DataFeatures = $true; $AdRetentionScores = $true; $DownstreamRowLevel = $true }
if ($All -or -not ($Active -or $Predictions -or $AdHocCheckpoints -or $AdHocResults -or $AdHocFeatures -or
                   $DataProcessed -or $DataFeatures -or $AdRetentionScores -or $DownstreamRowLevel)) {
    $Active = $true; $Predictions = $true; $AdHocCheckpoints = $true; $AdHocResults = $true; $AdHocFeatures = $true
    $DataProcessed = $true; $DataFeatures = $true; $AdRetentionScores = $true; $DownstreamRowLevel = $true
}

$Tmp = Join-Path $Root "_archive_tmp"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

# Logs / TensorBoard / Chemprop sidecar dumps are not needed to reload weights.
$Exclude = @(
    "-xr!*.log",
    "-xr!*.tfevents*",
    "-xr!events.out.tfevents*",
    "-xr!__pycache__",
    "-xr!*.tmp",
    "-xr!chemprop_raw_preds.csv"
)

# Pack one or more repo-relative paths (files or directories) into a single archive.
function Pack-Paths {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Paths,
        [Parameter(Mandatory = $true)][string]$DestName,
        [switch]$Optional
    )
    $present = @()
    $absent = @()
    foreach ($rel in $Paths) {
        if (Test-Path -LiteralPath (Join-Path $Root $rel)) { $present += $rel } else { $absent += $rel }
    }
    if ($present.Count -eq 0) {
        if ($Optional) { Write-Host "SKIP $Name (nothing present)"; return }
        throw "Missing every source for $Name : $($Paths -join ', ')"
    }
    foreach ($a in $absent) { Write-Host "  note: $a not present, omitted from $DestName" }

    $tmpArc = Join-Path $Tmp $DestName
    if (Test-Path $tmpArc) { Remove-Item -Force $tmpArc }
    Write-Host "Packing $Name -> artifacts\$DestName"
    Push-Location $Root
    try {
        & $SevenZip a -t7z -mx=5 "-w$Tmp" @Exclude $tmpArc @present
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) { throw "7z failed for $Name (exit $LASTEXITCODE)" }
    $dest = Join-Path $Root "artifacts\$DestName"
    Move-Item -Force $tmpArc $dest
    Write-Host ("  {0:N1} MB" -f ((Get-Item $dest).Length / 1MB))
}

Set-Location $Root

# ---------------------------------------------------------------- weights / predictions
if ($Active) {
    Pack-Paths -Name "checkpoints/active" -Paths @("checkpoints\active") -DestName "checkpoints.7z"
}
if ($Predictions) {
    Pack-Paths -Name "results/cv_predictions" -Paths @("results\cv_predictions") -DestName "cv_predictions.7z" -Optional
}
if ($AdHocCheckpoints) {
    Pack-Paths -Name "checkpoints/ad_hoc_comparison" -Paths @("checkpoints\ad_hoc_comparison") -DestName "checkpoints_ad_hoc.7z"
}
if ($AdHocResults) {
    Pack-Paths -Name "results/downstream_cv/ad_hoc_selectivity_foldmatched" `
        -Paths @("results\downstream_cv\ad_hoc_selectivity_foldmatched") -DestName "ad_hoc_foldmatched_results.7z"
}
if ($AdHocFeatures) {
    Pack-Paths -Name "data/processed/features/ad_hoc_foldmatched" `
        -Paths @("data\processed\features\ad_hoc_foldmatched") -DestName "ad_hoc_foldmatched_features.7z" -Optional
}

# ------------------------------------------------------------------- derived bulk trees
# These used to sit in plain git (~640 MB of derived CSV/NPZ). They are all rebuildable
# from data/raw_m4/ plus the packed weights, and none of them is a manuscript source file:
# every number the paper displays comes from the small summary CSVs that stay in git.
if ($DataProcessed) {
    Pack-Paths -Name "data/processed tables" -Paths @(
        "data\processed\ds1.csv",
        "data\processed\ds2.csv",
        "data\processed\m4_processed.csv",
        "data\processed\audits\ligand_parse_audit.csv"
    ) -DestName "data_processed.7z" -Optional
}
if ($DataFeatures) {
    Pack-Paths -Name "data/processed/features (fold-local matrices)" -Paths @(
        "data\processed\features\rdkit_ds1",
        "data\processed\features\rdkit_ds2",
        "data\processed\features\chemprop_ds1",
        "data\processed\features\chemprop_ds2",
        "data\processed\features\chemprop_cmisplit"
    ) -DestName "data_features.7z" -Optional
}
if ($AdRetentionScores) {
    Pack-Paths -Name "results/downstream_cv/ad_retention row-level scores" -Paths @(
        "results\downstream_cv\ad_retention\fold_scores",
        "results\downstream_cv\ad_retention\fold_retention.csv"
    ) -DestName "ad_retention_scores.7z" -Optional
}
if ($DownstreamRowLevel) {
    Pack-Paths -Name "results/downstream_cv row-level dumps" -Paths @(
        "results\downstream_cv\ad\ad_fold_scores.csv",
        "results\downstream_cv\ad_hoc_selectivity\ds1_heldout_preds",
        "results\downstream_cv\pair_acquisition\pair_acquisition_fold_metrics.csv",
        "results\downstream_cv\pair_acquisition_tau_2_0p5\pair_acquisition_fold_metrics.csv"
    ) -DestName "downstream_rowlevel.7z" -Optional
}

Write-Host ""
Write-Host "Packed archives only. Unpacked trees stay gitignored. Never git add ."
Write-Host "  git add artifacts/*.7z artifacts/README.md .gitattributes .gitignore"
Write-Host "  git lfs push origin HEAD"
Write-Host "  git push"
Write-Host ""
Write-Host "Convenience switches: -Weights (checkpoints/predictions), -Bulk (data + results trees)."
