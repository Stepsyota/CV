# Быстрая проверка: torchvision (без Kaggle), 2 эпохи, без MLflow
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
Set-Location $Root

Write-Host "=== torch ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host "=== train (MLP-Mixer, aug #2, 2 epochs) ==="
uv run python scripts/train.py `
  --config configs/aug_variant_2.yaml `
  --data-source torchvision `
  --download-test `
  --epochs 2 `
  --no-mlflow

Write-Host "OK. Check checkpoints/aug_table_2/history.json"
