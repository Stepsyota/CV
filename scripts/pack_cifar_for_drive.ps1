# Опционально: zip только папки cifar-10 для загрузки на Drive (распаковать в Colab или на Drive).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Cifar = Join-Path $Root "cifar-10"
if (-not (Test-Path $Cifar)) {
    throw "Нет папки $Cifar"
}
$OutDir = Join-Path $env:USERPROFILE "Downloads"
$ZipPath = Join-Path $OutDir "cifar-10-kaggle.zip"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path $Cifar -DestinationPath $ZipPath
Write-Host "Created: $ZipPath"
Write-Host "На Drive распакуйте так, чтобы был путь .../CV-lab3/cifar-10/trainLabels.csv"
