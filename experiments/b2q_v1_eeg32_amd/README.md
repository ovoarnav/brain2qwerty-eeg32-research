# Brain2Qwerty V1 EEG32 AMD experiment

This directory contains the reproducible execution gates for the pre-hardware
comparison. The scientific intervention is restricted to input-channel picks:
all available 61 EEG electrodes (`full`) versus the fixed 32-electrode tuple in
`brain2qwerty_v1.eeg.EEG32_CHANNELS` (`eeg32`). Both arms use the
official V1 encoder, sentence transformer, preprocessing, split, vocabulary,
loss, and optimizer.

## Backend gate

Run a complete synthetic V1 forward, backward, and optimizer step before loading
real data:

```powershell
python experiments/b2q_v1_eeg32_amd/backend_gate.py `
  --backend directml --channels 32 --optimizer-step
```

`torch-directml` is isolated because its current release pins an older PyTorch
than the reproduction environment. A passing gate proves graph execution; it
does not erase that dependency deviation or guarantee a speedup.

## Dataset and training entrypoints

```powershell
$env:BRAIN2QWERTY_STUDIES = "C:\path\to\SpanishBCBL-parent"
$env:BRAIN2QWERTY_CACHE = "C:\path\to\cache"
$env:BRAIN2QWERTY_RESULTS = "C:\path\to\outputs\b2q_v1_eeg32_amd"

python -m brain2qwerty_v1.main cache --modality eeg --condition full --debug `
  --batch-size 1 --num-workers 0
python -m brain2qwerty_v1.main debug --modality eeg --condition eeg32 `
  --batch-size 1 --num-workers 0 --accumulate-grad-batches 64
```

A bounded end-to-end pilot can use the production Lightning path without
pretending to be a full result:

```powershell
python -m brain2qwerty_v1.main debug --modality eeg --condition eeg32 `
  --batch-size 4 --num-workers 0 --max-epochs 1 `
  --limit-train-batches 2 --limit-val-batches 1 --limit-test-batches 1
```

Remove `--debug` / use `train` only after both conditions pass the integration,
overfit, parity, and pilot gates described in the experiment specification.
