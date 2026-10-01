# MLflow UI с backend в ./mlruns репозитория (те же данные, что пишет train.py)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Mlruns = Join-Path $Root "mlruns"
New-Item -ItemType Directory -Force -Path $Mlruns | Out-Null

$Backend = ([System.Uri](Resolve-Path $Mlruns)).AbsoluteUri
Write-Host "MLflow backend: $Backend"
Write-Host "Open http://127.0.0.1:5000 — experiment: lab3-cifar10-mlp"

$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
Set-Location $Root
uv run mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri $Backend `
  --default-artifact-root $Backend
