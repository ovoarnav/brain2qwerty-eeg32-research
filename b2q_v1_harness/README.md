# Portable Brain2Qwerty V1 developer harness

This is a **new implementation** based on the supplied workflow documents; the original harness code was unavailable. It includes a full native Brain2Qwerty V1 CPU/checkpoint runner and a separate synthetic proxy for contract development. Proxy embedding results do not measure Brain2Qwerty decoding performance. There is no deployment stage.

## Run the actual full EEG32 Brain2Qwerty V1 on CPU

Use the original pinned V1 environment, `/workspace/venvs/brain2qwerty`, rather than the small proxy environment. The new runner builds the unchanged 623,548,457-parameter Conv + temporal-attention + sentence-transformer model. It maps supplied checkpoint tensors onto CPU, loads the state strictly, and does not restore the training optimizer or retrain the model.

```bash
cd /workspace/brain2qwerty-eeg32-research
# Your own trusted GPU-trained Lightning checkpoint; no recordings needed for a shape pass.
/workspace/venvs/brain2qwerty/bin/python b2q_v1_harness/tools/native_cpu_pass.py \
  --checkpoint /path/to/eeg32/best.ckpt --trusted-checkpoint

# Add a local native EEG batch to examine predictions on recorded inputs.
/workspace/venvs/brain2qwerty/bin/python b2q_v1_harness/tools/native_cpu_pass.py \
  --checkpoint /path/to/eeg32/best.ckpt --trusted-checkpoint \
  --fixture /path/to/eeg32_dataset_gate.npz --split test \
  --predictions b2q_v1_harness/results/native_predictions.npz

# Explicit architecture-only execution and upstream-gradient check.
/workspace/venvs/brain2qwerty/bin/python b2q_v1_harness/tools/native_cpu_pass.py \
  --untrained --backward
```

Without `--checkpoint` or `--untrained`, the runner searches the RunPod checkpoint locations `/workspace/results/eeg32/best*.ckpt` and `/workspace/results/brain2qwerty/eeg32/best*.ckpt`. A missing/ambiguous checkpoint is BLOCKED; it never substitutes random weights automatically. `--trusted-checkpoint` is for your own Lightning files containing pickled optimizer/metric configuration; a tensor-only state_dict uses the default restricted loader. The current cloud machine does not mount your original RunPod volume.

An optional NPZ accepts `neuro [N,32,25]`, `channel_positions [N,32,2]`, `subject_id [N]` or `[N,1]`, and `sentence_uids [N]`; optional `feature [N]` or `[N,1]` provides batch-only loss/mismatch, and optional `channel_names [32]` verifies the exact ordered EEG32 montage. Windows, coordinate frame and subject-index mapping must match the training pipeline. The existing `experiments/b2q_v1_eeg32_amd/dataset_gate.py` NPZ format is accepted directly with `--split`. Interleaved rows are grouped before the native forward and predictions are restored to their original row order; original native source is unchanged.

The [native CPU report](verification/NATIVE_CPU_REPORT.md) records a successful full-size CPU forward/backward and checkpoint roundtrip. These used generated noise and explicitly untrained weights. Your GPU checkpoint remains a separate missing artifact, not a failed CPU capability. The [RunPod branch](https://github.com/ovoarnav/brain2qwerty-eeg32-research/tree/experiment/b2q-v1-eeg32-runpod) documents the training paths and keeps checkpoints outside Git.

## Proxy embedding comparison

The full [embedding comparison report](verification/embedding_comparison/REPORT.md) covers 10 local-embedding variants, five matched seeds, two protocols and 100 completed training runs. On the balanced synthetic split, baseline test mismatch was 31.21%; 48-wide linear projection reached 20.00%; the validation-selected nonlinear 48-wide projection reached 19.48%. The original small fixture ranked variants differently. Read the controls, per-seed variation, projection-only control and embedding-rank measurements before drawing conclusions. These are synthetic results, not real EEG accuracy.

To run the experiment in the separate small environment:

```bash
python embedding_study.py --protocol balanced --output results/new-balanced
python embedding_study.py --protocol legacy --output results/new-legacy
python -m pip install -r requirements-analysis.txt
python embedding_analysis.py --balanced results/new-balanced --legacy results/new-legacy --output results/new-plots
python -m unittest -v test_controller test_tools test_embedding_study test_embedding_analysis
```

Output folders must be new. Default settings are five seeds and 160 updates per seed. Candidate files, full histories, CSVs, contracts, manifests, embedding geometry and PNG/PDF plots are committed under verification/embedding_comparison/. The separate embedding experiment archive includes every trained checkpoint; credentials and Python environments are excluded. The original model/scientific EEG32 comparison remains unchanged.

## Start on your machine

Keep this environment separate from the original V1 environment. Python 3.12+, Git, and an authenticated Codex CLI are prerequisites for agent runs. The self-test needs no Codex or credentials.

Windows PowerShell, from the extracted folder:

```powershell
.\setup.ps1
.\.venv-lab\Scripts\python.exe lab.py attach-repo "C:\path\to\brain2qwerty-eeg32-research"
.\.venv-lab\Scripts\python.exe lab.py agent --rounds 3
```

Linux/macOS:

```bash
bash setup.sh
.venv-lab/bin/python lab.py attach-repo /path/to/brain2qwerty-eeg32-research
.venv-lab/bin/python lab.py agent --rounds 3
```

Codex runs through your existing authenticated connection, not a model on the CPU. Install/sign in through official Codex setup if required. A run consumes your Codex allowance. `--model` optionally selects a model available in your installation. The harness invokes Codex with `--sandbox workspace-write`; do not rerun with sandbox bypass if permissions fail. Preserve errors in the report.

## This prepared cloud workspace

The repository is `/workspace/brain2qwerty-eeg32-research`. The small environment is `/workspace/venvs/b2q-harness`:

```bash
cd /workspace/brain2qwerty-eeg32-research/b2q_v1_harness
/workspace/venvs/b2q-harness/bin/python lab.py doctor
/workspace/venvs/b2q-harness/bin/python lab.py selftest
/workspace/venvs/b2q-harness/bin/python lab.py agent --rounds 3
```

The original V1 environment is `/workspace/venvs/brain2qwerty`. For native validation:

```bash
source /workspace/setup/brain2qwerty/activate.sh
python b2q_v1_harness/tools/native_smoke.py --repo "$PWD" --output b2q_v1_harness/results/native_smoke.json
```

On another compatible Linux cloud machine with Python 3.12, uv, GCC/G++ and writable `/workspace`, reproduce the original pinned environment with `bash b2q_v1_harness/tools/setup_native_cloud.sh` from the full repository checkout. This installs the full lock including CUDA wheels and KenLM, using a build-only CMake constraint. NeuralSet also needs its hardcoded home cache directory provisioned; the script creates it and may require filesystem permission on a restricted machine. The portable small harness can instead use `setup.sh` without the native stack.

The cloud task executes remotely while your laptop is closed. Sending new instructions and inspecting remote results requires internet access. No scheduled job or always-running service is installed: start a Codex cloud task to do more work. The repository branch and downloadable ZIP preserve portable code; local virtual environments and credentials are excluded. Cloud configuration publication remains a product action, not something this harness implements.

## What is enforced

`selftest` runs 16 contracts and seven explicitly seeded faults, plus three controller failure regressions. Contracts check logits, input variants, single-row operation, sentence/subject identity, padding, row order, sensor-coordinate permutation, coordinate and sensor dependence, label independence, gradients, checkpoint loading, representation metadata, parameter budget and evaluation repeatability. Some checks share a fixture and assertion; the number is not a coverage score. Seeded faults are not agent mistakes.

`agent` starts at most three rounds, with one repair per round, two CPU threads, a 600-second timeout per invocation and a 2400-second loop budget. Evaluation is an external subprocess; nonzero exits, missing/stale output and training exceptions cannot become PASS. Training has a 500,000-parameter gate and 160 updates each on seeds 7 and 17. It compares synthetic held-out character mismatch on a fixed 61-character fixture. A candidate must pass contracts and strictly improve the incumbent mean to be promoted. This reused developer fixture is not a scientific validation set. `--token-budget` defaults to 100000 reported input/output tokens, checked between calls; missing usage events are not evidence of zero actual usage. It is not an inside-call or dollar cap. No OS memory cap is implemented. POSIX timeouts kill the invocation's process group; Windows cleanup is limited to the direct process.

Each candidate workspace gets only the model and agent instructions. The operator evaluator stays outside it. **This is an operational separation, not a hardened hostile-code sandbox**: candidate Python executes with the operator's user permissions, and host Codex sandbox behavior must be verified separately. Use only trusted generated code. No raw EEG or credentials should enter candidate workspaces. A nonzero Codex exit is reported as BLOCKED, without a scripted replacement.

## Inspect and move results

Live outputs are ignored under `results/`. Read the latest `codex_*/DEV_DAY_REPORT.md`, attempt evaluation JSON, patch, candidate `math_diff.json`, and raw events. Review raw events and stderr for privacy before sharing. Portable validation evidence checked into `verification/` excludes raw agent events and local source snapshots.

`attach-repo` copies and hashes a fixed allowlist of source files without changing the checkout. It does not itself execute sampler methods. `native_smoke.py` separately builds actual V1/neuraltrain classes with reduced configuration, synthetic 32-channel tensors and interleaved sentences; it performs forward/backward, exercises the native sampler, and checks whether native logits restore interleaved input order. Missing dependencies or incompatibility are BLOCKED. On the current source the interleaved-order probe is BLOCKED: native logits stay in sentence-grouped order. The production sampler groups sentences, so this probe does not establish a failure on production batches; it identifies a stricter proxy contract that the native path does not provide. The smoke does not run real-data preprocessing or full training.

Manual route: `lab.py prepare /new/workspace`, edit only its `candidate.py` using Codex, then `lab.py evaluate /new/workspace`. `--context minimal` versus `structured` changes supplied mathematical context and the starting-model inspection report, not evaluation or agent rules. The structured workspace includes `INSPECTION.json` with a source hash, actual module shapes, input/parameter gradients and contracts. It describes the starting candidate; regenerate it after code changes. Actual context ablation, KenLM, GPU support, scientific comparison and porting proxy patches into V1 remain unverified.

## Inspect a candidate and check a local interface fixture

From the harness root, using the small environment:

```bash
python tools/inspect_candidate.py verification/manual_candidate.py --output results/inspection.json
python tools/fixture_smoke.py /local/operator-fixture.npz --output results/fixture_smoke.json
python -m unittest -v test_controller test_tools
```

The inspector executes trusted candidate code without training, traces top-level module input/output shapes, checks input and parameter gradients, and runs the 16 contracts. The evaluator remains the authority for training and promotion. Neither inspection nor a passing synthetic fixture verifies real EEG performance.

An operator fixture must contain exactly four NPZ arrays: `x` (floating point `[N,C,T]`), `coordinates` (floating point `[N,C,2]` in the same sensor order/frame), `subjects` and `sentences` (anonymous integer `[N]` identity codes). Windows must already be aligned; coordinate normalization and montage selection must already match the real pipeline. Do not include labels, subject names or raw metadata. `allow_pickle=False`, bounded dimensions, finite values and exact alignment are checked. The tool runs an untrained proxy locally and writes shapes/status only, without copying or uploading the fixture. `--source-kind synthetic` explicitly labels a generated developer fixture; the default operator input has unverified provenance. Actual operator EEG fixtures remain unrun.

Read [the Markdown review](verification/MD_REVIEW.md) for what was actionable, completed or blocked, and [the session packet](verification/SESSION_PACKET.md) for the evidence and five discussion questions.

`docs/ORIGINAL_*.md` preserve the uploaded documents as historical references. Their claimed numbers apply to the unavailable original implementation, not this code. Use this README and `verification/BUILD_VERIFICATION.md` for current capabilities and results.
