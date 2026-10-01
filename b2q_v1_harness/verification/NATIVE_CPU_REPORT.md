# Full native Brain2Qwerty V1 EEG32 CPU verification

The actual unchanged 623,548,457-parameter V1 architecture runs on the current CPU cloud machine. This verification uses the original pinned PyTorch 2.6.0 environment and `BrainModule`, `SimpleConvTimeAgg`, temporal attention, the native sentence transformer and 29-character head. It does not use the four-statistic proxy.

## Measured execution

Two generated noise windows `[2,32,25]` with the exact named EEG32 montage and FULL-61 reference coordinate frame produced finite embeddings `[2,2048]` and logits `[2,29]`. Initial weights are explicitly untrained; no model accuracy is reported.

- Forward/backward run: forward 0.397 seconds, backward 0.967 seconds, 7.13 seconds including initialization and gradient inspection; peak process RSS 6,807.8 MiB (6.65 GiB).
- Input-signal and coordinate gradients were finite and nonzero. Present parameter gradients were finite; channel merger, initial convolution, convolution sequence, temporal attention, sentence transformer and head each received nonzero gradients. The unlabeled-noise objective is mean-squared logits, solely a connectivity probe. No optimizer update was made.
- Forward/checkpoint run: forward 0.404 seconds, 5.22 seconds including initialization and roundtrip; peak process RSS 4,167.5 MiB (4.07 GiB), including the second loaded model. These are two-window measurements, not full-dataset throughput estimates.
- A temporary 2,494,359,016-byte state_dict from generated **untrained** full native weights was saved and strictly reloaded into a meta-device skeleton on CPU. Embeddings and logits matched exactly. The temporary file was deleted. This verifies the full native loader, not access to the user's GPU weights.

Machine-readable evidence: [native_cpu_full32.json](native_cpu_full32.json) and [native_cpu_checkpoint_roundtrip.json](native_cpu_checkpoint_roundtrip.json). The latter includes hashes of the unchanged native source/config/montage. Input provenance and checkpoint provenance remain explicit.

## Prior GPU artifacts

Fetched `experiment/b2q-v1-eeg32-runpod` at `d4de47598eea0561a1120a1a99d8ba50b63adb81`. Its `scripts/cloud/runpod_train_eeg32.sh` writes to `/workspace/results/eeg32`. `cloud_results/README.md` explicitly keeps recordings, checkpoints and raw predictions outside Git. The fetched branch contains launch/benchmark scripts, not a committed trained checkpoint or completed GPU metrics.

The current machine has no native checkpoint at the documented output paths, no RunPod volume mounted, no RunPod credential binding, and no checkpoint attached to this chat. Git branches/tags and current local artifact locations were inspected; none supplied the trained weights. GitHub release API inspection was unavailable because the network proxy returned CONNECT 403. This does not establish that the prior GPU run failed or that its external checkpoint is gone.

Loading the user's trained model requires its accessible checkpoint or original volume connection. A CPU shape pass needs only that checkpoint. For meaningful EEG predictions, provide a matching cached/preprocessed batch too; the runner accepts the existing dataset-gate NPZ. No new GPU rental or retraining is required for these passes.

[native_gpu_checkpoint_missing.json](native_gpu_checkpoint_missing.json) records the default runner's BLOCKED result and unsuccessful exit when no native best checkpoint exists. It did not launch a random-weight fallback.

## Runner and validation

`tools/native_cpu_pass.py` has two explicit weight modes: supplied native checkpoint or `--untrained`. Default checkpoint discovery fails with BLOCKED when no unique artifact exists. Tensor-only loading is restricted by default; `--trusted-checkpoint` supports configuration objects in the user's own Lightning checkpoint. CPU mapping and memory-mapped tensors avoid restoring optimizer state or allocating a duplicate randomly initialized full model. Incompatible/proxy state_dicts are rejected strictly.

Eight new tests cover native checkpoint equality after CPU loading, incompatible weights, proxy rejection, native sentence grouping/order/label alignment, exact montage validation, cross-subject UID collisions, existing dataset-gate fixtures, trusted Lightning configuration handling and missing-checkpoint behavior. Together with the existing project suite: **26 passed**, with three existing dependency warnings. Ruff checks and formatting passed for the runner and new tests.

The remaining scientific embedding comparison has not been performed on this trained checkpoint. Native CPU capability, checkpoint transport, recorded-input evaluation, and changed-embedding experiments are separate outcomes.

## Reproduce

From the full repository checkout in the original pinned environment:

```bash
python b2q_v1_harness/tools/native_cpu_pass.py --untrained --backward --output b2q_v1_harness/results/native-full32.json
python b2q_v1_harness/tools/native_cpu_pass.py --untrained --verify-checkpoint-roundtrip --output b2q_v1_harness/results/native-roundtrip.json
python -m pytest -q tests
```

The checkpoint roundtrip temporarily needs about 2.5 GB of disk space. Original full-size scientific configs, preprocessing, electrode montage and native package files remain unchanged.
