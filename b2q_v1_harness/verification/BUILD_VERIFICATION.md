# Current harness verification

This is a newly generated harness, not the unavailable kit described in the uploaded historical documents.

## Embedding experiments completed October 1, 2026

The [embedding report](embedding_comparison/REPORT.md) supersedes the earlier single-candidate experiment as evidence about embedding changes. Ran 10 variants × five seeds × two protocols = 100 completed fits, 16,000 fixed-budget AdamW updates. All variants passed the 16 contracts in both protocols. The larger balanced protocol uses independent shuffled 232/116/232 synthetic train/validation/test examples; the legacy protocol uses 61-character noise replicas. Context/head architecture and initialization are matched for each seed, context/fused widths stay 32, and validation selection is recorded before test evaluation.

Balanced mean test character mismatch: baseline 31.21%, 32-wide projection control 20.17%, linear48 20.00%, linear64 16.21%, validation-selected mlp48 19.48%. The legacy fixture reverses several rankings. Centered embedding spectra confirm linear widths >=4 retain rank 4 in this four-statistic proxy; nonlinear sensor/projector variants expand the sampled basis. Full per-seed errors, curves, candidate code, hashes, selection records and geometry are preserved; all trained tensor checkpoints are included in the separate results archive. No real EEG performance or nested CLI execution is claimed.

All 16 controller/tool/study/analysis tests passed. Ruff checks/formatting passed for the five new experiment/analysis Python files. PNG/PDF figures were generated and visually inspected. The original 18-test project suite and native backend results below are unchanged historical validation for this branch.

Verified in the current Linux cloud machine with Python 3.12.14, PyTorch 2.10.0+cpu and NumPy 2.3.5:

- All 16 baseline contracts passed.
- All seven deliberately injected faults were caught. These are seeded faults, not Codex mistakes.
- All three self-test controller regressions passed.
- All five independent controller unit tests passed, including fresh failed output after a training exception, stale output rejection, unsuccessful exit rejection, reported usage counting and timeout handling.
- Fixed 160-update synthetic training completed on seeds 7 and 17 for the baseline and manual width-migration candidate.
- The candidate built in this active Codex session had local [7,48], contextual [7,32], fused [7,32] and passed all 16 external contracts.
- Baseline mean synthetic held-out character mismatch: 0.081967; candidate: 0.106557. The candidate was rejected under the strict-improvement rule. Neither number is EEG accuracy or paper CER.

## Native V1 evidence

The full repository suite passed all 18 tests, including V1 and V2 synthetic checks (three dependency warnings). The complete 623,548,457-parameter V1 graph passed CPU forward, backward with finite gradients, and AdamW optimizer update for both 32 and 61 channels. See backend-cpu-32.json and backend-cpu-61.json. These gates establish graph execution, not learning on EEG or a full training run. The native cloud setup script successfully repeated the pinned installation and dependency consistency check.

Used the separate original pinned Python environment, PyTorch 2.6.0+cu124. Reduced native V1 classes passed forward and backward with finite gradients and expected [4,32] embeddings / [4,29] logits. The real sampler returned [0,2,1,3] for interleaved A,B,A,B metadata.

The stronger interleaved-order fidelity probe is **BLOCKED**: native logits remain in sentence-grouped order rather than original input order. They match expected rows indexed [0,2,1,3]. This identifies a difference between the strict proxy contract and native source behavior; the production sampler groups sentences, so it is not evidence that normal production batches fail. Original V1 files were not changed. See native_smoke.json for the observed result.

## Autonomous CLI limitation

Attempted the actual three-round controller using the installed Codex CLI with workspace-write sandbox requested. Its first model connection repeatedly failed with proxy HTTP CONNECT 403. The scoped CLI process was terminated after the confirmed network denial, and the controller recorded BLOCKED. No autonomous round completed, and host CLI sandbox enforcement was not established. No credentials, proxy settings or security controls were bypassed. The manual active-Codex workspace experiment is separately labelled; it does not stand in for a successful CLI loop.

Portable evidence includes baseline/seeded-fault checks, native smoke, manual evaluation, model, representation metadata, patch, and sanitized blocked CLI summary. Raw events/stderr and local source snapshots remain ignored in the cloud workspace. No EEG, environment secrets or virtual environments are included.

## Markdown follow-up tools

Reviewed all 15 source Markdown documents and recorded actions/discrepancies in MD_REVIEW.md. SESSION_PACKET.md answers the five uploaded discussion questions against actual evidence. Implemented a deterministic operator-side inspector with source hashes, module shape traces, signal/coordinate gradients, parameter gradients and existing contracts. Baseline and manual candidate inspection both passed; the width adapter is independently observed rather than trusting math_diff.json.

Implemented the operator-local NPZ interface checker. Explicitly generated synthetic 32- and 61-channel fixtures passed with logits [7,29]. No actual operator EEG fixture was supplied or tested; arrays stay in ignored results/ and are excluded from the ZIP. Misaligned coordinate arrays are rejected. Structured preparation now includes a starting-model INSPECTION.json; minimal preparation omits context and inspection while retaining identical candidate, task and rules.

All 10 controller/tool unit tests passed (five controller, five tools), the 16-contract/seven-fault self-test passed, and Ruff checks/formatting passed for the four changed Python files. After a workspace restart, retried one structured autonomous round; its model connection again returned proxy HTTP CONNECT 403. The scoped invocation was stopped and recorded BLOCKED; no architecture round or actual context ablation completed. See cli_restart_blocked_summary.json.

Windows/macOS execution, actual real-window fixtures, real EEG training, GPU/DirectML/ROCm, KenLM, context ablation, hostile-code isolation, and application deployment were not tested. Cloud configuration draft fields were not saved: their tool is unavailable. The prepared machine and published Git branch are distinct from environment publication.
