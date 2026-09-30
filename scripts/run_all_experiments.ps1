$ErrorActionPreference = "Stop"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
Set-Location $PSScriptRoot\..

$configs = @(
    "configs/aug_variant_2.yaml",
    "configs/aug_variant_6.yaml",
    "configs/aug_variant_7.yaml"
)

foreach ($cfg in $configs) {
    Write-Host "========== Training $cfg =========="
    uv run python scripts/train.py --config $cfg --download-test
}

Write-Host "========== Preprocessing demo =========="
uv run python scripts/demo_preprocessing.py

Write-Host "========== Report =========="
uv run python scripts/generate_report.py

Write-Host "Done."
