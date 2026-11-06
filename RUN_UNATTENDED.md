# Starting the unattended Phase 3 run

For a PC that has finished Parts 1 and 2 of `GPU_SETUP.md` (cloned, setup done, test question answered). Run everything in PowerShell. Starting takes about 5 minutes, then the run looks after itself.

Pausing, resuming and every other command are listed in `RUN_COMMANDS.md`.

## 1. Update the code

```powershell
cd $HOME\f26-g6-hallucination-detectors
git pull
```

Check that the script arrived. This must print `True`:

```powershell
Test-Path scripts\run_all.ps1
```

## 2. Make sure the token is saved

```powershell
Test-Path $HOME\hf_cache\token
```

If it prints `False`, save the token once and paste it when asked:

```powershell
New-Item -ItemType Directory -Force $HOME\hf_cache | Out-Null
Read-Host "Paste token" | Set-Content -NoNewline -Encoding ascii $HOME\hf_cache\token
```

## 3. Make sure nothing interrupts the PC

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Then:

- Pause updates: Settings, Windows Update, "Pause updates". Choose the longest option.
- Keep the PC plugged in.
- Close other programs that use the GPU, such as games, video editors and other AI tools.

Locking the screen is fine. Signing out or shutting down stops the run until you next sign in, when it starts again by itself.

## 4. Start it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Hours 24
```

Leave this window open. You can minimise it.

It reruns setup (quick now), then smoke, timing and the full run. If the full run crashes, it restarts up to 5 times. It pushes the small result files about once an hour, and at the end it checks the results and pushes them by itself. The full run saves after every item and continues from there, so at most one item is redone. A stop during smoke or timing repeats that stage, and a smoke that passed is not repeated.

`-Hours 24` is the time budget for the sampling detectors. The timing stage shrinks sampling to fit it. MAM always runs in full, so a very low number can be exceeded.

## 5. One look after about an hour

This is the one check worth doing, because a problem in the first stages stops the run and it then waits for you. Open a second PowerShell window:

```powershell
cd $HOME\f26-g6-hallucination-detectors
Get-Content results\raw\run_all.log -Tail 30 -Encoding Unicode
```

| What you see | What it means |
|---|---|
| Lines about `MAM`, `Classifiers` or `Sampling` | Running normally. Leave it |
| `SMOKE PASS` further up, and `Total ETA` | Passed its checks. The ETA is the measured run time |
| `=== PAUSED` | Paused as you asked. Everything is saved |
| `=== STOPPED` | It stopped. Send the last 30 lines of the log |
| `=== DONE` | Finished and pushed |

Also send `results\raw\smoke\examples.txt` when you can, so the sample answers get read. The run does not wait for this.

## If you need results before it finishes

Pause it, then send what exists. The commands are under "Stop early and send what exists" in `RUN_COMMANDS.md`.

To continue later, run the command from step 4 again. It goes straight back to the full run and keeps its sizes.

The report also needs time, and the deadline is Nov 8, so plan to stop it with enough time left if it has not finished.

## If the PC restarts or loses power

Sign in. The run starts again by itself in its own window and continues from the last saved item. If no window appears, run the command from step 4.

Do not run `git pull` once the full run has started, because results from two code versions would mix.
