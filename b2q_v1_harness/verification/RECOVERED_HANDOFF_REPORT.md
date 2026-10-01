# Recovered real EEG handoff and native CPU passes

Inspected the user's `portable_neural_ai_chatgpt_handoff_complete.zip` on October 1, 2026. Archive SHA256: `f3898ecf8d23fc02db45004a4d0b7ab37dcbdec8a5b3677a5ee401f2426a5c56`; compressed size 5,261,597 bytes. ZIP integrity passed. Extracted source/results to `/workspace/recovered/portable_neural_ai_handoff`, excluding cached bytecode. The Markdown handoff describes provenance and historical work; its suggested live-hardware/product tasks were not treated as the user's request.

## What was actually recovered

- Source for a calibrated `StandardScaler + LogisticRegression` EEG classifier and its preprocessing, calibration, montage studies, tests and replay app.
- Four participants' 32-channel held-out prediction archives: **3,942 rows** total.
- **48 sentence replay trials containing 1,744 real EEG windows**, each `[32,50]`, sampled at 100 Hz over -0.2:+0.3 seconds and stored in microvolts.
- The exact motor-32 montage matches the native repository's `EEG32_CHANNELS`.
- Historical metrics, reports and decision records from the earlier CPU feasibility study.

The archive contains **no `.ckpt`, `.pt`, `.pth`, `.safetensors`, `.joblib`, `.pkl` or `.onnx` trained weights**. Its `outputs/models` files contain labels, classes, saved probabilities and identity arrays, not fitted coefficients or neural state_dicts. Its `.gitignore` explicitly excludes model `.joblib`, `.pt` and `.ckpt` files. The raw recordings, full train/validation epoch cache and upstream clone are also excluded by the handoff. Its own notes identify the compact classifier as distinct from full Brain2Qwerty V1 training.

This is useful real-data evidence, but it is not the later GPU-trained neural checkpoint. No conclusion about the existence or success of that separate GPU run follows from its absence here.

## Verified saved predictions on CPU

Recomputed the saved probability metrics and compared them with `personal_only / maximum` rows in the supplied `calibration_scaling.csv`. All six metrics matched within `1e-10` (observed maximum difference below `9e-17`). Every replay sentence's labels, alphabet and probability rows matched the corresponding full prediction export; float32 replay probabilities used tolerances `atol=1e-7`, `rtol=1e-6`.

| Participant | Saved prediction rows | Replay windows | Recomputed top-1 | Recomputed top-5 |
|---|---:|---:|---:|---:|
| 002 | 1,047 | 408 | 15.76% | 51.10% |
| 006 | 1,011 | 442 | 17.80% | 57.37% |
| 007 | 906 | 466 | 15.56% | 56.18% |
| 010 | 978 | 428 | 18.10% | 57.46% |

Unweighted participant means are **16.81% top-1** and **55.53% top-5**. This recomputes historical held-out classifier outputs; it is not a new model forward, retraining result or neural embedding comparison. The original saved alphabets contain 26 or 30 symbols and are not silently mapped into the native V1 29-class head.

The supplied sprint suite passed **17 tests** in the pinned native CPU environment. One scikit-learn metric warning reports predictions for a class absent from a participant's test labels; the values still reproduce the recorded CSV.

## Actual native Brain2Qwerty on recorded inputs

For each participant, passed the first complete replay sentence through the **unchanged full 623,548,457-parameter native V1** on CPU, with explicitly untrained weights. The bridge converts microvolts to volts and anti-aliases 100 Hz to 50 Hz using `scipy.signal.resample_poly(up=1, down=2)`, yielding 25 samples per window. Fixed FULL-61 reference-frame coordinates select the exact EEG32 montage. Each run emitted finite `[N,2048]` embeddings and `[N,29]` logits.

| Participant | Recorded input shape | Forward seconds | Peak process RSS MiB | Result |
|---|---|---:|---:|---|
| 002 | `[37,32,25]` | 1.68 | 3,176 | PASS |
| 006 | `[28,32,25]` | 1.28 | 3,130 | PASS |
| 007 | `[37,32,25]` | 1.95 | 3,169 | PASS |
| 010 | `[22,32,25]` | 1.18 | 3,099 | PASS |

Total: **124 recorded-input windows**. Times exclude construction/warmup; these are execution measurements, not a full-dataset throughput benchmark.

The bridge cannot recover the trained robust scaler, original per-subject index mapping or full native extraction from the raw recording. Subject index zero is an explicit untrained placeholder. Labels are not exported and decoding accuracy is not computed for native passes. Fixtures carry `native_untrained_only=True`; the native runner rejects their use with checkpoint evaluation. This prevents an interface-only conversion from becoming a trained-model EEG performance claim.

Machine-readable evidence: [recovered_handoff_audit.json](recovered_handoff_audit.json), [native_recorded_input_002.json](native_recorded_input_002.json), [native_recorded_input_006.json](native_recorded_input_006.json), [native_recorded_input_007.json](native_recorded_input_007.json), [native_recorded_input_010.json](native_recorded_input_010.json). Source/replay/prediction hashes are preserved. Raw EEG arrays, raw labels and sentence text remain local and are not committed or copied into the source ZIP.

Nine native runner tests passed after adding the untrained-only fixture guard; the existing 18 project tests passed in the prior native verification. Ruff checks and formatting passed for both tools and native runner tests. Original scientific configuration, preprocessing and native model code remain unchanged.

## Reproduce and remaining artifact

From the full repository in the original pinned native Python environment:

```bash
python b2q_v1_harness/tools/inspect_handoff.py \
  --handoff /workspace/recovered/portable_neural_ai_handoff \
  --fixture-dir b2q_v1_harness/results/recovered_handoff \
  --output b2q_v1_harness/results/handoff-audit.json
python b2q_v1_harness/tools/native_cpu_pass.py --untrained \
  --fixture b2q_v1_harness/results/recovered_handoff/real-replay-002-untrained-only.npz \
  --output b2q_v1_harness/results/native-recorded-002.json
```

The inspector consumes the user's own trusted ragged NumPy replay exports, which require object-array loading. It does not launch the uploaded app, download recordings, use test labels to fit a model, invoke the suggested live pilot, or infer missing weights from saved probabilities.

To compare changed embeddings in the trained model, the remaining required artifact is the GPU run's native checkpoint (typically `best.ckpt` under `/workspace/results/eeg32` on the original RunPod volume), plus a matching native-preprocessed batch and subject mapping for meaningful EEG evaluation. A checkpoint-only CPU shape pass remains possible with the runner already supplied. This archive does not contain that checkpoint.
