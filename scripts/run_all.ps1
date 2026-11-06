# Unattended Phase 3 run: setup, smoke, timing, full (restarted after a crash), verify, push the results.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
# Safe to run again after any stop: once timing has written its sizes, it goes straight to full and keeps them.
param(
    [double]$Hours = 24,
    [int]$MaxFullTries = 5,
    [int]$RetryWaitSeconds = 300
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force "results\raw" | Out-Null
$log = Join-Path $root "results\raw\run_all.log"
$python = ".venv\Scripts\python"

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"  # so the log keeps up while output is piped
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $HOME "hf_cache" }

function Invoke-Step([string]$label, [string]$command) {
    # cmd merges stderr into stdout: Windows PowerShell would otherwise turn every stderr line into an error record.
    "=== $(Get-Date -Format s) $label" | Tee-Object -FilePath $log -Append | Out-Host
    cmd /c "$command 2>&1" | Tee-Object -FilePath $log -Append | Out-Host
    return $LASTEXITCODE
}

function Stop-Run([string]$message) {
    "=== STOPPED: $message" | Tee-Object -FilePath $log -Append | Out-Host
    exit 1
}

$sizesMeasured = Select-String -Path "configs\run.yaml" -Pattern "^measured:" -Quiet

if (-not $sizesMeasured) {
    # Read-Host keeps the token out of the PowerShell history; it stays in this window's memory only.
    if (-not $env:HF_TOKEN -and -not (Test-Path (Join-Path $env:HF_HOME "token"))) {
        $env:HF_TOKEN = Read-Host "Paste your Hugging Face token"
    }
    if ((Invoke-Step "setup" "py -3.11 scripts\bootstrap.py") -ne 0) { Stop-Run "setup failed. Fix the error above, then run this script again." }
    if ((Invoke-Step "smoke" "$python scripts\run_phase3.py smoke") -ne 0) { Stop-Run "smoke failed, so the long run did not start." }
    if ((Invoke-Step "timing" "$python scripts\run_phase3.py timing --hours $Hours") -ne 0) { Stop-Run "timing failed, so the long run did not start." }
}

$finished = $false
foreach ($try in 1..$MaxFullTries) {
    if ((Invoke-Step "full, try $try of $MaxFullTries" "$python scripts\run_phase3.py full") -eq 0) { $finished = $true; break }
    if ($try -lt $MaxFullTries) { Start-Sleep -Seconds $RetryWaitSeconds }
}
if (-not $finished) { Stop-Run "full kept failing. Finished work is safe; run this script again after fixing the error." }

if ((Invoke-Step "verify" "$python scripts\run_phase3.py verify") -ne 0) { Stop-Run "verify failed, so nothing was pushed." }

Invoke-Step "commit" "git add results configs\run.yaml" | Out-Null
Invoke-Step "commit" "git commit -m ""Add Phase 3 detector results""" | Out-Null
if ((Invoke-Step "push" "git push") -ne 0) { Stop-Run "the results are committed but the push failed. Run git push." }
"=== DONE: results pushed" | Tee-Object -FilePath $log -Append | Out-Host
