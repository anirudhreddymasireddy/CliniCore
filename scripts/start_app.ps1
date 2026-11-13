# Start CliniCore on Windows (one web app on http://127.0.0.1:8765).
#   powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1            # local files (results/, models/, app_state/)
#   powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1 exasol     # read results and write staff feedback in Exasol
param([string]$Source = "csv")
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:CLINICORE_SOURCE = $Source
$env:CLINICORE_PIPELINE = $root
if ($Source -eq "exasol" -and -not $env:EXA_PASS) {
  $env:EXA_PASS = (Get-Content "$HOME\.exasol-starter-kit\credentials\personal_sys_password" -Raw).Trim()
}
Start-Job { Start-Sleep 4; Start-Process "http://127.0.0.1:8765" } | Out-Null
Set-Location "$root\app\backend"
Write-Host "CliniCore ($Source) on http://127.0.0.1:8765 - stop with Ctrl+C"
python -m uvicorn main:app --host 127.0.0.1 --port 8765
