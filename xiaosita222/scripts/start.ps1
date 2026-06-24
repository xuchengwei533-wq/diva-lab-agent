$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Warning "Created .env from .env.example. Fill DASHSCOPE_API_KEY before using the service."
}

$env:PYTHONPATH = $Root
$Port = if ($env:PORT) { $env:PORT } else { "47231" }
python -m uvicorn app.main:app --host 0.0.0.0 --port $Port
