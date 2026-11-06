# Running Phase 3 on the GPU machine

Steps for the Windows PC with the RTX A4000 (16 GB). Run every command in PowerShell.

Time needed: about 15 minutes to prepare, 30 to 60 minutes for setup and the model download, 15 to 30 minutes for the first two checks, then an unattended run of up to about 30 hours.

## Part 1: prepare the PC

### 1. Check the NVIDIA driver

```powershell
nvidia-smi
```

Look for `CUDA Version` in the top right of the output.

- 13.2 or higher: go to step 2.
- Below 13.2, or the command is not found: go to https://www.nvidia.com/drivers, choose NVIDIA RTX / Quadro, then RTX A4000, then Windows 11. Install the latest driver, restart, and run `nvidia-smi` again.

### 2. Install Python 3.11 and git

Check what is already there:

```powershell
py -3.11 --version
git --version
```

Install whichever failed:

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
```

If winget says no package was found, run `winget source update` and try again, or use the installers:

- Python 3.11: https://www.python.org/downloads/windows/ (a 3.11 Windows 64-bit installer; tick "Add python.exe to PATH")
- Git: https://git-scm.com/download/win

Close PowerShell, open a new window, and run the two version checks again.

### 3. Log in to GitHub

The repository is private. Git opens a browser the first time it needs access.

```powershell
git ls-remote https://github.com/ahmad1khan2/f26-g6-hallucination-detectors HEAD
```

Sign in as `ahmad1khan2` and authorize. It worked if it prints a long commit hash followed by `HEAD`.

Then set your identity, so commits from this PC look like your own. Use your own GitHub username and email:

```powershell
git config --global user.name "YOUR_USERNAME"
git config --global user.email "YOUR_EMAIL"
```

The GitHub CLI (`gh`) is not needed.

### 4. Get a Hugging Face token

1. Open https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct while logged in and confirm you have access.
2. Open https://huggingface.co/settings/tokens, choose "Create new token", then "Read". Copy it somewhere safe.

The setup script asks for it when needed. Never put the token in a repository file or share it in a chat.

### 5. Check disk space and turn off sleep

```powershell
Get-PSDrive C | Select-Object @{n='FreeGB';e={[math]::Round($_.Free/1GB)}}
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

You need at least 50 GB free. The last two lines stop the PC sleeping while plugged in. Keep it plugged in during the run.

## Part 2: set up the project

### 6. Clone and run the setup script

Clone to a folder outside OneDrive, because the run writes many files.

```powershell
cd $HOME
git clone --recurse-submodules https://github.com/ahmad1khan2/f26-g6-hallucination-detectors
cd f26-g6-hallucination-detectors
py -3.11 scripts\bootstrap.py
```

The script creates `.venv`, installs the CUDA build of torch and the pinned packages, asks for your Hugging Face token, downloads the model (about 16 GB), and answers one test question. It is safe to run again if it stops.

It worked if it prints a question, an answer, the GPU name and the peak VRAM. It also prints two `setx` lines at the end. Run them once, so later windows find the model:

```powershell
setx HF_HOME "$HOME\hf_cache"
setx PYTHONUTF8 1
```

Open a new PowerShell window and return to the folder with `cd $HOME\f26-g6-hallucination-detectors`.

## Part 3: run the experiments

### 7. Smoke test

```powershell
.venv\Scripts\python scripts\run_phase3.py smoke
```

This runs 20 items per dataset through every detector, then the GPU tests. It must end with PASS. It also prints the path of a file with example answers. Send that file for checking before going on: prompts and label rules are frozen at this point.

### 8. Timing test

```powershell
.venv\Scripts\python scripts\run_phase3.py timing --hours 30
```

Choose `--hours` to fit the time you have. This measures real speed, writes `configs/run.yaml`, and prints the plan and the total time. Sampling shrinks to fit the hours. MAM always runs in full, so it can take longer than the hours you give if they are too few.

### 9. Full run

```powershell
.venv\Scripts\python scripts\run_phase3.py full
```

If it stops for any reason (crash, restart, Ctrl-C), run the same command again. It continues from where it stopped and redoes nothing. You can stop it whenever you need results, write up what has finished, and resume later.

### 10. Check the results

```powershell
.venv\Scripts\python scripts\run_phase3.py verify
```

It must pass. Warnings about languages with too few items of one class are expected, and are written to `results\flags.txt`. They are not failures.

### 11. Send the results back

```powershell
git add results configs\run.yaml
git commit -m "Add Phase 3 detector results"
git push
```

`results\raw` (features and samples, about 7 GB or more) is not pushed and stays on this PC. Keep it until Phase 4 is finished.

The same three commands also work part-way through a run, after stopping `full`, if a report needs partial results.

## If something goes wrong

| Problem | What to do |
|---|---|
| `nvidia-smi` is not found | Install or update the NVIDIA driver (step 1) |
| Setup warns that the pinned torch has no wheel | The driver is too old for the tested torch. Update the driver and run the setup script again |
| 401 or 403 when downloading the model | The token is wrong, or the account has no Llama access (step 4). Run `.venv\Scripts\hf auth login` and try again |
| `CUDA is not available` | Run the setup script again, then check `nvidia-smi` works |
| Out of memory during sampling | Close other programs that use the GPU. Run `timing` again; it lowers the batch size by itself |
| Text in the console looks garbled | Run `setx PYTHONUTF8 1` and open a new window. The saved files are unaffected |
| The PC restarted during the run | Run the `full` command again |
