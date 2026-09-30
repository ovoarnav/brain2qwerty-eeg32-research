$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.12 -m venv .venv-lab
if ($LASTEXITCODE -ne 0) { throw "venv failed" }
& .\.venv-lab\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Torch install failed" }
& .\.venv-lab\Scripts\python.exe -m pip install numpy==2.3.5
if ($LASTEXITCODE -ne 0) { throw "NumPy install failed" }
& .\.venv-lab\Scripts\python.exe lab.py doctor
& .\.venv-lab\Scripts\python.exe lab.py selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed" }
