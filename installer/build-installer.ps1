<#
.SYNOPSIS
  Builds the one-click Sam3 Annotation installer end to end:
  freeze the Python backend with PyInstaller, then package the Electron
  shell (with the frozen backend as extraResources) with electron-builder.

.DESCRIPTION
  Run from anywhere; paths are resolved relative to this script.
  Requires: Python 3 and Node.js/npm on PATH. Builds the backend inside an
  isolated venv (backend/.venv) rather than whatever Python is ambient on
  this machine -- PyInstaller's collect_all() pulls in every importable
  package it can find, so building against a shared/dev Python environment
  (with unrelated ML tooling, notebooks, etc. installed) silently bloats the
  frozen backend with gigabytes of things this app never uses. The venv
  keeps the frozen output to what backend/requirements.txt actually needs.
  Produces app/dist/Sam3 Annotation Setup.exe (a few GB -- it bundles
  CUDA-enabled torch so the installed app needs no separate Python/CUDA
  setup).

.PARAMETER SkipBackend
  Skip the PyInstaller step and reuse whatever is already in backend/dist.
  Useful when iterating on the Electron shell only.
#>
param(
    [switch]$SkipBackend
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "backend"
$appDir = Join-Path $root "app"

if (-not $SkipBackend) {
    $venvDir = Join-Path $backendDir ".venv"
    $venvPython = Join-Path $venvDir "Scripts\python.exe"

    if (-not (Test-Path $venvPython)) {
        Write-Host "==> Creating isolated build venv at backend\.venv" -ForegroundColor Cyan
        & python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    }

    Write-Host "==> Installing backend Python dependencies (isolated venv)" -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $backendDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    & $venvPython -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller install failed" }

    Write-Host "==> Freezing backend with PyInstaller (this takes a while -- torch is large)" -ForegroundColor Cyan
    Push-Location $backendDir
    try {
        & $venvPython -m PyInstaller sam3_backend.spec --noconfirm
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
    } finally {
        Pop-Location
    }

    $backendExe = Join-Path $backendDir "dist\sam3-backend\sam3-backend.exe"
    if (-not (Test-Path $backendExe)) {
        throw "expected $backendExe after PyInstaller build but it's missing"
    }
} else {
    Write-Host "==> Skipping backend build (-SkipBackend)" -ForegroundColor Yellow
}

Write-Host "==> Installing Electron app dependencies" -ForegroundColor Cyan
Push-Location $appDir
try {
    & npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

    Write-Host "==> Building installer with electron-builder" -ForegroundColor Cyan
    # This is an unsigned Windows-only build -- without this,
    # electron-builder still auto-discovers/downloads macOS codesigning
    # tooling (winCodeSign) it doesn't need, which then fails to extract on
    # Windows (its .7z contains symlinks a non-elevated account can't
    # recreate) and retries forever instead of erroring out.
    $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw "electron-builder failed" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "==> Done. Installer output:" -ForegroundColor Green
Get-ChildItem (Join-Path $appDir "dist") -Filter "*.exe" | ForEach-Object {
    Write-Host ("    {0}  ({1:N1} GB)" -f $_.FullName, ($_.Length / 1GB))
}
