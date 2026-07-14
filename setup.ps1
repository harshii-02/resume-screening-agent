# Quick setup for Windows PowerShell — run from repo root
# Usage:  .\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtualenv..."
    python -m venv .venv
}

Write-Host "Activating .venv and installing requirements..."
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example (add GROQ_API_KEY or OPENAI_API_KEY optionally)."
}

Write-Host ""
Write-Host "Running scoring unit tests..."
python tests/test_scoring.py

Write-Host ""
Write-Host "Setup done. Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python main.py"
Write-Host "  streamlit run app.py"
