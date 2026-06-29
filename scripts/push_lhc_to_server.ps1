# Copy LHC manifest + PDFs from this PC to the JAMS server.
# Usage (PowerShell):
#   $env:JAMS_SERVER = "root@65.108.236.135"
#   .\scripts\push_lhc_to_server.ps1

$ErrorActionPreference = "Stop"
$Server = $env:JAMS_SERVER
if (-not $Server) {
    Write-Host "Set JAMS_SERVER first, e.g. `$env:JAMS_SERVER = 'root@65.108.236.135'"
    exit 1
}

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Lhc = Join-Path $Root "data\lhc"

if (-not (Test-Path (Join-Path $Lhc "manifest.json"))) {
    Write-Host "Run metadata sync locally first:"
    Write-Host "  .\.venv\Scripts\python.exe scripts\run_lhc_sync.py --metadata-only"
    exit 1
}

Write-Host "==> Creating remote folders"
ssh $Server "mkdir -p /opt/jams/data/lhc/pdfs && chown -R www-data:www-data /opt/jams/data/lhc"

Write-Host "==> Uploading manifest.json"
scp (Join-Path $Lhc "manifest.json") "${Server}:/opt/jams/data/lhc/manifest.json"

$PdfDir = Join-Path $Lhc "pdfs"
$Pdfs = Get-ChildItem $PdfDir -Filter "*.pdf" -ErrorAction SilentlyContinue
if ($Pdfs) {
    Write-Host "==> Uploading $($Pdfs.Count) PDF(s)"
    scp @($Pdfs.FullName) "${Server}:/opt/jams/data/lhc/pdfs/"
} else {
    Write-Host "==> No local PDFs to upload (metadata only)"
}

Write-Host "==> Fixing ownership"
ssh $Server "chown -R www-data:www-data /opt/jams/data/lhc"

Write-Host "Done. Refresh LHC Judgments tab on the server."
