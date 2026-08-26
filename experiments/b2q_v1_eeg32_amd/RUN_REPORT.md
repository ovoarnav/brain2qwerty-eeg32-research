# Brain2Qwerty V1 FULL-61 vs EEG32 local run report

Run date: 2026-08-25/26 (America/Toronto)  
Repository: `ovoarnav/brain2qwerty`  
Starting commit: `db21bc96e11046ce041e12fc4dd76467d0ab6380`  
Branch: `experiment/b2q-v1-eeg32-amd`

## Outcome

The EEG implementation, all mandatory correctness gates, and bounded production
pilots are complete. A scientifically valid 300-epoch FULL/EEG32 result was not
claimed or launched locally because neither AMD training backend passed and the
measured faithful CPU fallback projects to about **74 continuous days** for the
paired run before validation and checkpoint overhead.

No model reduction was made. Both conditions use the official 623,548,457-
parameter V1 encoder and sentence Transformer.

## Backend decision

| Backend | Forward | Backward | Optimizer | Decision |
|---|---:|---:|---:|---|
| ROCm 7.2.1 / PyTorch 2.9.1 | not reached | not reached | not reached | rejected: `hipInfo` reports no ROCm-capable device |
| DirectML / PyTorch 2.4.1 | pass, 7.87 s | process crash | not reached | rejected: Windows access violation `0xC0000005` during full V1 backward |
| CPU / PyTorch 2.6.0 | pass | pass | pass | selected fallback |

ROCm was installed from AMD's official Windows 7.2.1 packages. DirectML ran the
complete 2048-wide encoder and depth-4 sentence Transformer, but its backward
pass crashed the process. CPU completed finite forward/backward and AdamW gates
for both 32 and 61 channels.

## Mandatory gates

| Gate | Evidence | Result |
|---|---|---|
| Environment | ROCm, DirectML, CPU probes | CPU fallback selected |
| Dataset | first usable BrainVision timeline through official study code | pass |
| EEG32 | exact ordered 32-name tensor | pass |
| Full forward | unchanged 623.5M model on real EEG | pass |
| Backward/update | finite gradients + AdamW | pass on CPU |
| Tiny overfit | ten repeated real-batch updates | pass both arms |
| FULL/EEG32 parity | signals, positions, labels, timing, splits | pass, max diff `0.0` |

The one-timeline data gate contained 2,011 keystrokes and 64 sentences:
1,602 train, 186 validation, and 223 test. FULL batches were `[64, 61, 25]`;
EEG32 batches were `[64, 32, 25]`.

The parity gate caught and fixed a subtle issue in `neuralset`: min-max position
normalization moved retained EEG32 sensors when applied after selection. Both
arms now use the same standard-1005 FULL-61 coordinate frame, so retained signals
and positions are exactly identical and only omitted sensors differ.

## Tiny-overfit evidence

| Condition | Initial loss | Loss after 10 updates | Final batch accuracy |
|---|---:|---:|---:|
| FULL-61 | 3.3756 | 0.0111 | 100% |
| EEG32 | 3.4277 | 0.0260 | 100% |

These are implementation checks on four examples, not generalization results.

## Bounded Lightning pilots

The production `Experiment.run()` path completed two training batches, one
validation batch, and one test batch per condition.

| Condition | Train loss | Val loss | Val CER | Test loss | Test CER |
|---|---:|---:|---:|---:|---:|
| FULL-61 | 4.8709 | 3.4942 | 0.75 | 3.2188 | 0.75 |
| EEG32 | 4.2159 | 4.0312 | 0.75 | 3.0942 | 0.75 |

These random-initialization-scale, tiny-pilot values prove logging, callbacks,
sentence grouping, validation, and test execution. They must not be compared to
the prior compact-decoder results or presented as scientific metrics.

## Measured CPU projection

Batch size 64 timings, excluding gate-only gradient scans:

- FULL-61: 5.912 s/training batch.
- EEG32: 5.743 s/training batch.
- Estimated train batches per condition per epoch: 1,832, using the audited
  146,549 cached keystrokes and the 80% training split.
- FULL: 3.01 h/epoch; EEG32: 2.92 h/epoch.
- Paired 300-epoch projection: 1,779 h = 74.1 days continuous, before validation,
  testing, cache misses, checkpoint writes, and interruptions.

Early stopping may reduce that time, but it cannot be assumed in advance. Each
full checkpoint is about 7.5 GB in this environment.

## Faithful CPU commands

Use a short cache root on Windows because Exca embeds extractor configuration in
directory names:

```powershell
$env:BRAIN2QWERTY_STUDIES = "C:\path\to\SpanishBCBL"
$env:BRAIN2QWERTY_CACHE = "C:\b2q"
$env:BRAIN2QWERTY_RESULTS = "C:\path\to\outputs\b2q_v1_eeg32_amd"

python -m brain2qwerty_v1.main train --modality eeg --condition full `
  --batch-size 64 --num-workers 0 --no-pin-memory
python -m brain2qwerty_v1.main train --modality eeg --condition eeg32 `
  --batch-size 64 --num-workers 0 --no-pin-memory
```

The defaults retain 300 epochs, patience 30, seed 33, AdamW/OneCycleLR, official
windowing/preprocessing/split, and best-validation-CER checkpoint selection.

## Validation

- `pytest -q`: 18 passed.
- Ruff formatting/checks: passed.
- No external language model used.
- No hidden-width, depth, virtual-channel, Transformer, loss, or label changes.
