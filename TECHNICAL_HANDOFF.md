# Brain2Qwerty V1 EEG32 Technical Handoff

## Purpose

This experiment compares **FULL 61-channel EEG → official Brain2Qwerty V1** with
**fixed 32-channel EEG → the same official V1 model**. The question is whether a
portable-oriented montage retains useful information under the 623,548,457-
parameter architecture. An earlier compact decoder measured FULL top-1 17.3%,
EEG32 top-1 16.7%, and EEG32 top-5 55.4%; this experiment has not yet produced
full scientific training results.

## Implementation

`brain2qwerty_v1/config/xp_config.py` clones the official V1 model, loss,
optimizer, scheduler, windowing, labels, seed, and paraphrase-aware split, then
selects the Pinet2024 EEG study and either all EEG channels or `EEG32_CHANNELS`.
`brain2qwerty_v1/eeg.py` defines the exact ordered montage:

```text
F3 F4 F5 F6 Fz FC1 FC2 FC3 FC4 FC5 FC6 FT7 FT8 C1 C2 C3 C4 C5 C6 Cz
T7 T8 CP1 CP2 CP3 CP4 CP5 CP6 CPz TP7 TP8 Pz
```

`brain2qwerty_v1/main.py` loads BrainVision events, applies preprocessing and
keystroke windows (`start=-0.2`, `duration=0.5`), groups sentences, builds
train/validation/test loaders, constructs the convolutional neural encoder and
sentence Transformer from the upstream model configs, and trains with the
official CTC path. Validation CER selects the best checkpoint; early stopping
and test evaluation are handled by Lightning. `brain2qwerty_v1/utils.py`
provides channel positions. A min-max bug was found: normalizing after selecting
EEG32 changed retained coordinates. The fix uses one standard-1005 FULL-61
reference frame before selection, so retained FULL/EEG32 coordinates and signals
match exactly.

The data path is BrainVision EEG → study loader → preprocessing → aligned
windows → sentence grouping → split → V1 model. The audited cache contains
146,549 aligned keystrokes, 20 participants, 61 usable FULL channels, and 32
EEG32 channels. The split is the existing paraphrase-aware Brain2Qwerty split.

## Commands

```powershell
python -m brain2qwerty_v1.main train --modality eeg --condition full
python -m brain2qwerty_v1.main train --modality eeg --condition eeg32
```

Defaults are seed 33, 300 epochs, patience 30, batch size 64, AdamW (`lr=5e-5`,
`weight_decay=1e-4`), OneCycleLR, and best-validation-CER checkpoint selection.

## Verified gates and hardware

Dataset loading, exact tensor construction, full-model forward, backward,
AdamW, tiny overfit, FULL/EEG32 parity, and Lightning train/validation/test all
passed. Tiny overfit reached 100% accuracy: FULL loss 3.3756→0.0111; EEG32
3.4277→0.0260. These are implementation checks, not scientific results.

ROCm 7.2.1 / PyTorch 2.9.1 exposed no ROCm-capable device. DirectML/PyTorch
2.4.1 completed full forward but crashed in backward with Windows access
violation `0xC0000005`. CPU/PyTorch 2.6.0 passed forward/backward/optimizer.
Measured batch times are FULL 5.912 s and EEG32 5.743 s; about 1,832 batches per
epoch per condition; 3.01 h and 2.92 h per epoch; paired 300-epoch projection
about 1,779 h (74.1 continuous days). Compute, not implementation correctness,
is the blocker.

## Not done

There is no valid full 300-epoch FULL result, EEG32 result, or scientific
comparison yet. No external language model, architecture reduction, or cloud
rewrite has been made.

## Next task

> The next task is to adapt the compute/device/environment layer for a Linux NVIDIA CUDA cloud machine while keeping the scientific experiment unchanged, then run short GPU validation followed by the full FULL-61 vs EEG32 training comparison.

Likely cloud-only files are `brain2qwerty_v1/main.py` (accelerator/device
selection), environment/dependency files, and launch scripts. Do not change
`eeg.py`, the scientific configs in `config/xp_config.py`, preprocessing,
split/transforms, model configs, loss, optimizer, channel coordinates, or the
parity gates.
