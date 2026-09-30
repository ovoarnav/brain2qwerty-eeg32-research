# Start here — Brain2Qwerty V1 laptop harness

The kit is runnable code, not a prompt asking Codex to build its own evaluator.
Use a NEW folder/environment. Do not replace your existing V1 environment.
Start with synthetic data. GPU/ROCm/DirectML/NPU, downloaded EEG, LLM weights and cloud accounts are not needed.
Codex itself uses your existing authenticated Codex connection; it is not running the LLM on the laptop CPU.

## 1. Install the small CPU environment (Windows PowerShell)
From the extracted b2q_v1_harness folder, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv-lab
.\.venv-lab\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-lab\Scripts\python.exe -m pip install numpy==2.3.5
.\.venv-lab\Scripts\python.exe lab.py doctor
.\.venv-lab\Scripts\python.exe lab.py selftest
```

Python 3.13 also works in the environment used to test this package; use your installed compatible
Python interpreter to create the venv. PowerShell activation is not required. Do not “fix” dependency
errors by upgrading your original scientific environment.

On Linux/macOS use `python3 -m venv .venv-lab` and `.venv-lab/bin/python` instead.
On macOS, install `torch==2.10.0 numpy==2.3.5` from PyPI, not the CPU wheel index used for Windows/Linux.

`selftest` runs 16 baseline checks and seven deliberately broken candidates. It writes
`results/selftest.json` and `results/DEV_DAY_REPORT.md`. Those faults are not Codex failures.

## 2. Attach your local V1 checkout (recommended; read-only)

```powershell
.\.venv-lab\Scripts\python.exe lab.py attach-repo "C:\path\to\brain2qwerty-eeg32-research"
```

Replace only that path with your existing checkout. The command copies a small allowlist of V1 source
files into results/repo_audit/source; it does not read EEG files or edit your repository. It records Git
HEAD/content hashes and exercises the real grouping/sampler methods using synthetic metadata.
This operation executes selected code from the checkout: use your trusted repo, not arbitrary Python.
Missing attach-repo does not prevent the proxy run, but leaves source-fidelity evidence incomplete.

## 3. Run Codex through the harness
Git must be installed and available on PATH (`git --version`).
Ensure the Codex CLI is installed and signed in (`codex --version`; use official Codex login/setup).
Do not paste API keys into this folder. The controller does not install/authenticate Codex for you.

```powershell
.\.venv-lab\Scripts\python.exe lab.py agent --rounds 3
```

The run is explicit: this command launches Codex and can consume your Codex usage allowance.
Defaults: 3 rounds, at most 1 repair per round, 2 CPU threads; each agent invocation times out at
600 seconds and the experiment loop has a 2400-second budget. Training has a 500,000-parameter gate
and 160 updates per seed, seeds 7 and 17. The reported-token cap is checked BETWEEN calls; it is not
an inside-call dollar/token guarantee. There is no guaranteed hard RAM limit on Windows.

The controller uses your configured Codex model. To pin one, add `--model` with a model identifier
actually available in YOUR installation, rather than copying an invented model name.

Codex may fail on a permission/configuration problem. Preserve that error. Never rerun this harness
with danger-full-access or disable OS security just to make it proceed.

## 4. Bring these artifacts to the session

Open `results/codex_<timestamp>_<id>/DEV_DAY_REPORT.md`.
Also inspect the first failing `round_*/attempt_*.evaluation.json`, its `.patch`, and `math_diff.json`
in the corresponding workspace. Raw `.events.jsonl` contains Codex tool activity and token usage when
reported. Review logs/source for privacy before sharing. No raw EEG is needed for the session.

## 5. Stronger fidelity check: your actual V1 classes
Use the Python from your ORIGINAL, already-installed V1 environment, not the tiny venv above:

```powershell
& "C:\path\to\your-original-v1-env\Scripts\python.exe" tools/native_smoke.py --repo "C:\path\to\brain2qwerty-eeg32-research" --output results/native_smoke.json
```

This builds actual V1/neuraltrain classes with an explicitly reduced configuration and random
keystroke-shaped tensors. It never calls Data.build or starts full training. A missing dependency,
shape mismatch or native-source incompatibility is written as BLOCKED, not silently treated as success.
This native path was not verified in the build environment because your installed V1 stack is absent.
Its result is distinct from the proxy and remains a smoke test, not original model reproduction.

## Optional manual Codex app workflow

```powershell
.\.venv-lab\Scripts\python.exe lab.py prepare results\manual_workspace
```

Open ONLY that generated workspace in the Codex app, ask it to read AGENTS.md and TASK.md and perform
round 1, then evaluate it from your operator terminal:

```powershell
.\.venv-lab\Scripts\python.exe lab.py evaluate results\manual_workspace
```

The CLI/controller route is preferred because it records the bounded autonomous loop automatically.

## Optional context ablation
Run `lab.py agent --rounds 1 --context minimal` and `--context structured` with the same model/budget.
The minimal condition omits the math context and inspector, not safety rules. This is a single-case
context/tool ablation, not evidence of broadly superior harness design.

## Real-window interface fixture
`tools/fixture_smoke.py` accepts an operator-created local NPZ with the documented arrays. It checks
an untrained proxy against real shapes only. It does not download data, change splits, export data to
Codex, or measure real brain-decoding accuracy. Keep subject IDs and coordinate frames consistent.

The native smoke bridge checks the existing original classes. It does not automatically port a
proxy architecture patch into your full V1 model; that is a separate integration step.
