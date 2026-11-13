# Build every model and the ICU board from scratch (Windows). Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\build_all.ps1 -NhamcsDir "C:\data\nhamcs" [-Exasol]
# Needs: NHAMCS ED Stata files (2018-2022) in -NhamcsDir. PhysioNet 2019 is downloaded automatically (Kaggle mirror).
param([Parameter(Mandatory=$true)][string]$NhamcsDir, [switch]$Exasol)
$ErrorActionPreference = "Stop"
$src = if ($Exasol) { "exasol" } else { "csv" }
if ($Exasol -and -not $env:EXA_PASS) { $env:EXA_PASS = (Get-Content "$HOME\.exasol-starter-kit\credentials\personal_sys_password" -Raw).Trim() }
Write-Host "1/4 Stage 1 data (NHAMCS)";            python pipeline\nhamcs.py --dir $NhamcsDir --cache nhamcs_cache $(if ($Exasol) {"--exasol"})
Write-Host "2/4 Stage 1 model";                    python pipeline\nhamcs_stage1.py --cache nhamcs_cache --model logreg $(if ($Exasol) {"--exasol"})
Write-Host "3/4 PhysioNet 2019 data";              python pipeline\sepsis2019_load.py $(if (-not $Exasol) {"--no-exasol"})
Write-Host "4/4 ICU models + board (real patients)"; python pipeline\physionet_icu.py --source $src
Write-Host "Done. Start the app:  powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1 $src"
