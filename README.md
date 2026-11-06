# f26-g6-hallucination-detectors

AI2002 Artificial Intelligence, Fall 2026, Track B (Research & Development), Group 6.

We test whether the finding of Datta et al. (EACL 2026), that hallucination detectors lose far less performance than task accuracy when a model moves from English to low-resource languages, still holds on a broader set of languages (X-FACTR) and on long answers (PsiloQA).

Base paper: Datta, Chilukuri, Kumar, Ghosh & Zafar (2026). Do LLM hallucination detectors suffer from low-resource effect? EACL 2026. https://aclanthology.org/2026.eacl-long.136/

## Team

- Ahmad Ali Khan (Lead, 24L-3049)
- Talha Pasha (24L-3005)
- Moeed Amir (24L-3060)
- Bilal Kashif (24L-3036)

## Phases

1. Paper selection and proposal (submitted)
2. Data acquisition and setup
3. Implementation and execution
4. Evaluation and comparative analysis
5. Technical report on results
6. Final report (LaTeX)

## Setup

```
python -m venv %USERPROFILE%\venvs\f26g6
%USERPROFILE%\venvs\f26g6\Scripts\activate
pip install -r requirements.lock.txt
python scripts/check_env.py
git submodule update --init
```

`requirements.txt` lists the direct dependencies; `requirements.lock.txt` pins the exact versions (Python 3.11). Further run instructions are added as each phase lands.

## Running Phase 3 on the GPU machine

Windows, PowerShell. Needs Python 3.11, git, an NVIDIA driver and access to this private repository. Turn off Windows sleep before the long runs; the first model run downloads about 16 GB. Full steps and troubleshooting are in `GPU_SETUP.md`.

```powershell
git clone --recurse-submodules https://github.com/ahmad1khan2/f26-g6-hallucination-detectors
cd f26-g6-hallucination-detectors
py -3.11 scripts\bootstrap.py                                    # venv, CUDA torch, pinned dependencies, HF login, GPU check
.venv\Scripts\python scripts\run_phase3.py smoke                 # 20 items per dataset, then pytest -m gpu
.venv\Scripts\python scripts\run_phase3.py timing --hours 30     # measures speed, writes configs\run.yaml, prints the ETA
.venv\Scripts\python scripts\run_phase3.py full                  # resumable: after any stop, rerun the same command
.venv\Scripts\python scripts\run_phase3.py verify                # tests\test_results.py and results\flags.txt
.venv\Scripts\python scripts\run_phase3.py verify --partial      # the same for a run stopped early
```

Afterwards: `git add results; git commit; git push`. `results/raw/` stays on that machine.

Tests:

```powershell
pytest                # everything except the GPU tests
pytest tests/unit     # one level: unit, integration, regression or system
pytest -m gpu         # real-model tests, GPU machine only
```

## License

MIT. The authors' released code used in this project is also MIT licensed.
