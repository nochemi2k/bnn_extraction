# Unpack Git LFS artifact archives into gitignored working trees.
# Every archive stores paths from the repository root. Always extract at the repo root.
#
#   git lfs install
#   git lfs pull
#   powershell -File scripts/unpack_artifacts.ps1
#
# Selective:
#   git lfs pull -I artifacts/checkpoints.7z
#   powershell -File scripts/unpack_artifacts.ps1 -Active
param(
    [switch]$Active,
    [switch]$Predictions,
    [switch]$AdHoc,
    [switch]$AdHocFeatures,
    [switch]$DataProcessed,
    [switch]$DataFeatures,
    [switch]$AdRetentionScores,
    [switch]$DownstreamRowLevel,
    [switch]$Weights,   # checkpoints + fold-matched ad hoc weights/results
    [switch]$Bulk,      # the four derived data/results trees
    [switch]$Rebuild,   # everything a --dry manuscript rebuild needs
    [switch]$All
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$SevenZip = Join-Path ${env:ProgramFiles} "7-Zip\7z.exe"
if (-not (Test-Path $SevenZip)) {
    throw "7-Zip not found at $SevenZip"
}

if ($Weights) { $Active = $true; $AdHoc = $true; $Predictions = $true; $AdHocFeatures = $true }
if ($Bulk)    { $DataProcessed = $true; $DataFeatures = $true; $AdRetentionScores = $true; $DownstreamRowLevel = $true }
# A tables-and-figures rebuild reads the processed tables, the fold-local feature matrices
# (iz_risk_ordering_figure recomputes the median-Mahalanobis score from them) and the
# row-level retention scores. It does not need model weights.
if ($Rebuild) { $DataProcessed = $true; $DataFeatures = $true; $AdRetentionScores = $true; $DownstreamRowLevel = $true }
if ($All) {
    $Active = $true; $Predictions = $true; $AdHoc = $true; $AdHocFeatures = $true
    $DataProcessed = $true; $DataFeatures = $true; $AdRetentionScores = $true; $DownstreamRowLevel = $true
}
if (-not ($Active -or $Predictions -or $AdHoc -or $AdHocFeatures -or
          $DataProcessed -or $DataFeatures -or $AdRetentionScores -or $DownstreamRowLevel)) {
    # Default: what a fresh clone needs to rebuild the paper, plus the active weights.
    $Active = $true; $AdHoc = $true
    $DataProcessed = $true; $DataFeatures = $true; $AdRetentionScores = $true; $DownstreamRowLevel = $true
}

$missing = @()

function Unpack-One {
    param(
        [Parameter(Mandatory = $true)][string]$RelArchive,
        [Parameter(Mandatory = $true)][string]$Label,
        [switch]$Optional
    )
    $arc = Join-Path $Root $RelArchive
    if (-not (Test-Path $arc)) {
        if ($Optional) {
            Write-Host "SKIP $Label (missing $RelArchive). Fetch with: git lfs pull -I $RelArchive" -ForegroundColor Yellow
            $script:missing += $RelArchive
            return
        }
        throw "Missing $arc (run: git lfs pull)"
    }
    # An LFS pointer is a ~130 byte text file; extracting one gives a confusing 7z error.
    if ((Get-Item $arc).Length -lt 1024) {
        throw "$RelArchive looks like an unfetched Git LFS pointer ($((Get-Item $arc).Length) bytes). Run: git lfs pull -I $RelArchive"
    }
    Write-Host "Unpacking $RelArchive ($Label)"
    & $SevenZip x $arc "-o$Root" -y | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z extract failed for $RelArchive" }
}

if ($Active) {
    Unpack-One -RelArchive "artifacts\checkpoints.7z" -Label "active-grid weights -> checkpoints/active/"
}
if ($Predictions) {
    Unpack-One -RelArchive "artifacts\cv_predictions.7z" -Label "OOF dumps -> results/cv_predictions/" -Optional
}
if ($AdHoc) {
    Unpack-One -RelArchive "artifacts\checkpoints_ad_hoc.7z" -Label "fold-matched ad hoc weights -> checkpoints/ad_hoc_comparison/" -Optional
    Unpack-One -RelArchive "artifacts\ad_hoc_foldmatched_results.7z" -Label "fold-matched ad hoc preds/metrics -> results/downstream_cv/ad_hoc_selectivity_foldmatched/" -Optional
}
if ($AdHocFeatures) {
    Unpack-One -RelArchive "artifacts\ad_hoc_foldmatched_features.7z" -Label "fold-matched DS1 features -> data/processed/features/ad_hoc_foldmatched/" -Optional
}
if ($DataProcessed) {
    Unpack-One -RelArchive "artifacts\data_processed.7z" -Label "DS1/DS2 tables -> data/processed/" -Optional
}
if ($DataFeatures) {
    Unpack-One -RelArchive "artifacts\data_features.7z" -Label "fold-local feature matrices -> data/processed/features/" -Optional
}
if ($AdRetentionScores) {
    Unpack-One -RelArchive "artifacts\ad_retention_scores.7z" -Label "retention row scores -> results/downstream_cv/ad_retention/" -Optional
}
if ($DownstreamRowLevel) {
    Unpack-One -RelArchive "artifacts\downstream_rowlevel.7z" -Label "row-level downstream dumps -> results/downstream_cv/" -Optional
}

Write-Host ""
if ($missing.Count -gt 0) {
    Write-Host "Skipped $($missing.Count) archive(s) that were not fetched:" -ForegroundColor Yellow
    foreach ($m in $missing) { Write-Host "  git lfs pull -I $m" }
    Write-Host ""
}
Write-Host "Done. See artifacts/README.md."
