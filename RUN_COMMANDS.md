# Phase 3 run: all the commands

Every command for the GPU PC, in one place. Run them in PowerShell. All of them except the first start from the project folder:

```powershell
cd $HOME\f26-g6-hallucination-detectors
```

First-time setup is in `GPU_SETUP.md`, Parts 1 and 2.

## Right now, after the shutdown

Your finished work is most likely still on this PC, so do not delete anything in `results`.

**1. See where the run got to.** Send me these lines if they look odd:

```powershell
Get-Content results\raw\run_all.log -Tail 30 -Encoding Unicode
```

**2. Find out why the PC went down:**

```powershell
Get-WinEvent -FilterHashtable @{LogName='System';Id=41,1074,6008} -MaxEvents 5 | Format-List TimeCreated,Id,Message
```

| Id | Meaning |
|---|---|
| 41 or 6008 | Sudden loss of power, or a crash |
| 1074 | A deliberate restart or shutdown. The message says which program asked for it, for example Windows Update |

**3. Get the new code.** This one pull is safe even though the run had started, because it changes how results are saved and not how they are computed:

```powershell
git pull
```

If it refuses because of local changes, send me the output and do not discard anything.

**4. Start it again.** It continues from the last saved item:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
```

## Before a long run

Stop Windows from sleeping or hibernating:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Also pause updates (Settings, Windows Update, "Pause updates"), keep the PC plugged in, and close other programs that use the GPU.

## Start or continue the run

Use the same command both times. Leave the window open; you can minimise it.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
```

Finished stages are skipped, and the full run continues from the last saved item. Once timing has finished, `-Hours` has no effect and the sizes stay as they are.

If it says the run is already running, it is. Look for the other window. Starting it twice is refused on purpose.

## Pause

Open a second PowerShell window, go to the project folder, then:

```powershell
New-Item results\raw\PAUSE -ItemType File -Force
```

The run finishes the item it is on, saves, pushes a backup, and exits with `=== PAUSED`. That is usually seconds, and can take a few minutes during a classifier training. The GPU is then free, and you can shut the PC down.

## Resume after a pause

Run the start command again:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
```

The model takes a minute or two to load, and it also re-checks the saved files first, so the first new item can take a few minutes to appear.

## Emergency stop

Press Ctrl-C in the run window, or close it. Nothing finished is lost, and at most the current item is redone.

Prefer the pause above if you can. After Ctrl-C the run starts again at your next login, while a paused run stays paused.

## After a restart or power cut

The run starts again by itself when you **sign in**, in its own window. It does not start before you sign in, so sign in after any restart.

A paused or finished run does not start again.

If no window appears within a minute or so, start it by hand:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
```

## See how it is doing

The last lines of the log:

```powershell
Get-Content results\raw\run_all.log -Tail 30 -Encoding Unicode
```

What is left to do. This is safe while the run is going, and each language that is not finished is listed:

```powershell
.venv\Scripts\python scripts\run_phase3.py verify --partial
```

## Check the backups

The run pushes answers, scores and predictions about once an hour, counted from when it started, and once more on a pause. Features and samples (about 7 GB) stay on this PC.

The latest backups in the log:

```powershell
Select-String -Path results\raw\run_all.log -Pattern "Backup|backup failed" -Encoding Unicode | Select-Object -Last 3
```

What GitHub has:

```powershell
git fetch
git log origin/main --oneline -5
```

A `WARNING: backup failed` line does not stop the run. If you see it repeatedly, send it to me. The most likely causes are no internet, or a new commit on GitHub that this PC does not have. Do not run `git pull` to fix that during a run; ask me first.

## Stop early and send what exists

For when the deadline is close. Pause as above, wait for `=== PAUSED`, then:

```powershell
.venv\Scripts\python scripts\run_phase3.py verify --partial
git add results configs\run.yaml
git commit -m "Add partial Phase 3 detector results"
git push
```

If `git commit` says there is nothing to commit, the pause's backup already holds everything, and you only need the `git push`.

You can continue later with the start command.

## Abandon the run

If you are done with a run that was not paused or finished, remove the automatic restart so it does not start at your next login:

```powershell
Remove-Item "$([Environment]::GetFolderPath('Startup'))\g6_run.cmd"
```

## What you see and what it means

| What you see | What it means |
|---|---|
| Lines about `MAM`, `Classifiers` or `Sampling` | Running normally. Leave it |
| `SMOKE PASS`, then `Total ETA` further up | Passed its checks. The ETA is the measured run time |
| `Backup pushed at ...` | The hourly copy reached GitHub |
| `WARNING: backup failed` | The copy did not reach GitHub, and the run carries on. Tell me if it repeats |
| `=== PAUSED` | Paused as you asked. Everything is saved |
| `This run is already running` | Another window has it. Nothing new was started |
| `=== STOPPED` | It stopped by itself. Send the last 30 lines of the log |
| `=== DONE` | Finished and pushed |

## Rule for pulling

After this one pull, do not run `git pull` while a run is unfinished unless I tell you a change is safe. Results from two versions of the detector code must not mix.
