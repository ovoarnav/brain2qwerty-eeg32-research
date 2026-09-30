# Current harness verification

This is a newly generated harness, not the unavailable kit described in the uploaded historical documents.

Verified in the current Linux cloud machine with Python 3.12.14, PyTorch 2.10.0+cpu and NumPy 2.3.5:

- All 16 baseline contracts passed.
- All seven deliberately injected faults were caught. These are seeded faults, not Codex mistakes.
- All three self-test controller regressions passed.
- All five independent controller unit tests passed, including fresh failed output after a training exception, stale output rejection, unsuccessful exit rejection, reported usage counting and timeout handling.
- Fixed 160-update synthetic training completed on seeds 7 and 17 for the baseline and manual width-migration candidate.
- The candidate built in this active Codex session had local [7,48], contextual [7,32], fused [7,32] and passed all 16 external contracts.
- Baseline mean synthetic held-out character mismatch: 0.081967; candidate: 0.106557. The candidate was rejected under the strict-improvement rule. Neither number is EEG accuracy or paper CER.

## Native V1 evidence

Used the separate original pinned Python environment, PyTorch 2.6.0+cu124. Reduced native V1 classes passed forward and backward with finite gradients and expected [4,32] embeddings / [4,29] logits. The real sampler returned [0,2,1,3] for interleaved A,B,A,B metadata.

The stronger interleaved-order fidelity probe is **BLOCKED**: native logits remain in sentence-grouped order rather than original input order. They match expected rows indexed [0,2,1,3]. This identifies a difference between the strict proxy contract and native source behavior; the production sampler groups sentences, so it is not evidence that normal production batches fail. Original V1 files were not changed. See native_smoke.json for the observed result.

## Autonomous CLI limitation

Attempted the actual three-round controller using the installed Codex CLI with workspace-write sandbox requested. Its first model connection repeatedly failed with proxy HTTP CONNECT 403. The scoped CLI process was terminated after the confirmed network denial, and the controller recorded BLOCKED. No autonomous round completed, and host CLI sandbox enforcement was not established. No credentials, proxy settings or security controls were bypassed. The manual active-Codex workspace experiment is separately labelled; it does not stand in for a successful CLI loop.

Portable evidence includes baseline/seeded-fault checks, native smoke, manual evaluation, model, representation metadata, patch, and sanitized blocked CLI summary. Raw events/stderr and local source snapshots remain ignored in the cloud workspace. No EEG, environment secrets or virtual environments are included.

Windows/macOS execution, real EEG training, GPU/DirectML/ROCm, KenLM, context ablation, hostile-code isolation, and application deployment were not tested. Cloud configuration draft fields were not saved: their tool is unavailable. The prepared machine and published Git branch are distinct from environment publication.
