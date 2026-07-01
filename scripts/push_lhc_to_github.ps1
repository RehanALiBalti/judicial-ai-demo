# Push LHC dataset to GitHub; server updates with git pull.
#
# Usage:
#   cd E:\python\ji\judicial-ai-demo
#   .\scripts\push_lhc_to_github.ps1
#   .\scripts\push_lhc_to_github.ps1 -IncludeIndexed   # after indexing (uses LFS for chroma/store)

param(
    [switch]$IncludeIndexed,
    [switch]$IncludeFccp
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$pdfCount = (Get-ChildItem "data\lhc\pdfs\*.pdf" -ErrorAction SilentlyContinue).Count
$pdfGb = [math]::Round(((Get-ChildItem "data\lhc\pdfs\*.pdf" | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host "LHC: $($pdfCount) PDFs (~$pdfGb GB) + manifest.json"
Write-Host "First push may take 30-60+ minutes depending on upload speed."
Write-Host ""

git add data/lhc/manifest.json data/lhc/pdfs/
if ($IncludeFccp) {
    git add data/fccp/manifest.json data/fccp/pdfs/
}
if ($IncludeIndexed) {
    if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
        Write-Host "Install Git LFS for indexed data:"
        Write-Host "  winget install GitHub.GitLFS"
        Write-Host "  git lfs install"
        exit 1
    }
    git lfs install
    if (-not (Test-Path "data\jams_store.json")) {
        Write-Host "ERROR: data\jams_store.json not found. Run --index-only first."
        exit 1
    }
    if (-not (Test-Path "data\chroma")) {
        Write-Host "ERROR: data\chroma not found. Run --index-only first."
        exit 1
    }
    $storeMb = [math]::Round((Get-Item "data\jams_store.json").Length / 1MB, 1)
    $chromaGb = [math]::Round(((Get-ChildItem "data\chroma" -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), 2)
    Write-Host "Indexed AI store: jams_store ${storeMb} MB, chroma ~${chromaGb} GB (Git LFS)"
    Write-Host "Using git add -f (files are in .gitignore but tracked via LFS)"
    git add -f data/jams_store.json
    git add -f data/chroma/
}

git status --short data/ | Select-Object -First 20
$total = (git status --short data/ | Measure-Object).Count
if ($total -gt 20) { Write-Host "... and $($total - 20) more files" }

$confirm = Read-Host "Commit and push to GitHub? (y/n)"
if ($confirm -ne "y") { exit 0 }

git config http.postBuffer 524288000
$commitMsg = "Add LHC judgment dataset ($pdfCount PDFs)"
if ($IncludeIndexed) {
    $commitMsg = "Add LHC dataset and indexed AI store (chroma + jams_store)"
}
git commit -m $commitMsg
git push

Write-Host ""
Write-Host "Server:"
Write-Host '  cd /opt/jams; sudo -u www-data git pull; sudo -u www-data git lfs pull'
Write-Host '  sudo bash /opt/jams/deploy/pull-data-from-git.sh'
