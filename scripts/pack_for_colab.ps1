# Архив кода для Colab (без venv, данных и больших артефактов).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$OutDir = Join-Path $env:USERPROFILE "Downloads"
$ZipPath = Join-Path $OutDir "CV-lab3-project.zip"

$Temp = Join-Path $env:TEMP ("cv-lab3-pack-" + [guid]::NewGuid().ToString("n"))
New-Item -ItemType Directory -Path $Temp | Out-Null

$excludeDirs = @(".venv", "cifar-10", "checkpoints", "artifacts", "mlruns", ".git", "__pycache__", ".cursor")
$items = Get-ChildItem -Path $Root -Force
foreach ($item in $items) {
    if ($excludeDirs -contains $item.Name) { continue }
    Copy-Item -Path $item.FullName -Destination (Join-Path $Temp $item.Name) -Recurse -Force
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $Temp "*") -DestinationPath $ZipPath

Remove-Item $Temp -Recurse -Force
Write-Host "Created: $ZipPath"
Write-Host "Upload to Google Drive: My Drive/CV-lab3/CV-lab3-project.zip"
Write-Host "Upload cifar-10 folder to: My Drive/CV-lab3/cifar-10"
