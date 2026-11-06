# Unattended Phase 3 run: setup, smoke, timing, full (restarted after a crash), verify, push the results.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
# Safe to run again after any stop: each stage that finished is skipped and full continues from the last saved item.
# To pause from another window: New-Item results\raw\PAUSE -ItemType File -Force
param(
    [double]$Hours = 24,
    [int]$MaxFullTries = 5,
    [int]$RetryWaitSeconds = 300,
    [string]$StartupDir = [Environment]::GetFolderPath("Startup"),
    [switch]$AutoResumed  # started by the Startup folder after a restart, not by hand
)

$PausedExit = 75  # PAUSED_EXIT in run_phase3.py

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force "results\raw" | Out-Null
$log = Join-Path $root "results\raw\run_all.log"
$python = ".venv\Scripts\python"
$pauseFile = "results\raw\PAUSE"
$startupFile = Join-Path $StartupDir "g6_run.cmd"

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"  # so the log keeps up while output is piped
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $HOME "hf_cache" }

# Windows drops this lock when the process dies, even in a power cut, so it can never go stale.
# Two runs writing the same files would corrupt them, and the automatic restart makes a double start easy.
try {
    $lock = [IO.File]::Open((Join-Path $root "results\raw\run_all.lock"), "OpenOrCreate", "ReadWrite", "None")
} catch {
    "=== This run is already running in another window, so nothing was started."
    exit 1
}

function Invoke-Step([string]$label, [string]$command) {
    # cmd merges stderr into stdout: Windows PowerShell would otherwise turn every stderr line into an error record.
    "=== $(Get-Date -Format s) $label" | Tee-Object -FilePath $log -Append | Out-Host
    cmd /c "$command 2>&1" | Tee-Object -FilePath $log -Append | Out-Host
    return $LASTEXITCODE
}

function Remove-AutoResume { Remove-Item $startupFile -Force -ErrorAction SilentlyContinue }

function Stop-Run([string]$message) {
    Remove-AutoResume  # restarting would only hit the same problem
    "=== STOPPED: $message" | Tee-Object -FilePath $log -Append | Out-Host
    exit 1
}

function Stop-Paused {
    Remove-AutoResume  # a paused run stays paused until it is started by hand
    "=== PAUSED: everything finished so far is saved. Run this script again to continue." | Tee-Object -FilePath $log -Append | Out-Host
    exit 0
}

function Invoke-Stage([string]$label, [string]$command, [string]$failure) {
    $code = Invoke-Step $label $command
    if ($code -eq $PausedExit) { Stop-Paused }
    if ($code -ne 0) { Stop-Run $failure }
}

if (Test-Path $pauseFile) {
    # Asked for a pause, then the PC went down first: honour it unless the user started this by hand.
    if ($AutoResumed) { Stop-Paused }
    Remove-Item $pauseFile -Force
}

$restart = "@echo off`r`ncd /d `"$root`"`r`npowershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours $Hours -AutoResumed`r`npause`r`n"
Set-Content -Path $startupFile -Value $restart -Encoding Ascii

$sizesMeasured = Select-String -Path "configs\run.yaml" -Pattern "^measured:" -Quiet

if (-not $sizesMeasured) {
    # Read-Host keeps the token out of the PowerShell history; it stays in this window's memory only.
    if (-not $env:HF_TOKEN -and -not (Test-Path (Join-Path $env:HF_HOME "token"))) {
        $env:HF_TOKEN = Read-Host "Paste your Hugging Face token"
    }
    if (-not (Test-Path "results\raw\smoke\passed")) {
        Invoke-Stage "setup" "py -3.11 scripts\bootstrap.py" "setup failed. Fix the error above, then run this script again."
        Invoke-Stage "smoke" "$python scripts\run_phase3.py smoke" "smoke failed, so the long run did not start."
    }
    Invoke-Stage "timing" "$python scripts\run_phase3.py timing --hours $Hours" "timing failed, so the long run did not start."
}

$finished = $false
foreach ($try in 1..$MaxFullTries) {
    $code = Invoke-Step "full, try $try of $MaxFullTries" "$python scripts\run_phase3.py full"
    if ($code -eq $PausedExit) { Stop-Paused }
    if ($code -eq 0) { $finished = $true; break }
    if ($try -lt $MaxFullTries) { Start-Sleep -Seconds $RetryWaitSeconds }
}
if (-not $finished) { Stop-Run "full kept failing. Finished work is safe; run this script again after fixing the error." }

Invoke-Stage "verify" "$python scripts\run_phase3.py verify" "verify failed, so nothing was pushed."

Invoke-Step "commit" "git add results configs\run.yaml" | Out-Null
Invoke-Step "commit" "git commit -m ""Add Phase 3 detector results""" | Out-Null  # fails harmlessly when a backup already holds everything
if (git status --porcelain results configs/run.yaml) { Stop-Run "the results could not be committed. See the git error above." }
if ((Invoke-Step "push" "git push") -ne 0) { Stop-Run "the results are committed but the push failed. Run git push." }
Remove-AutoResume
"=== DONE: results pushed" | Tee-Object -FilePath $log -Append | Out-Host
