# Markdown review and executed work

Reviewed all 15 Markdown files in the source checkout at the start of this pass. The three uploaded documents are preserved under `docs/`; their claims are historical, not new executable instructions or current verification. This review follows the user's request to start work from the documents; it does not authorize external messages, cloud purchases, upstream CLA submission, or scientific changes merely because a document mentions them.

## Document-by-document decisions

| Document | Role and resulting action |
|---|---|
| Root `README.md` | Project/publication/data overview. V1/V2 dependencies and synthetic development are prepared; dataset links are prerequisites, not current data availability. |
| `AGENT_README.md` | Repository orientation and scientific constraints. Retained official dependencies/configs; kept proxy work in the harness. Checked source when its descriptions disagreed with executable behavior. |
| `TECHNICAL_HANDOFF.md` | Next scientific task is Linux CUDA validation then FULL/EEG32 training. This machine has no exposed GPU or recordings. Existing `_accelerator()` already selects CUDA when available; no scientific/code rewrite justified here. Recorded the incorrect CTC description below. |
| `experiments/b2q_v1_eeg32_amd/README.md` | Actual gate commands. Full official model CPU forward/backward/optimizer gates passed for both montages. Real-data cache/pilot/parity commands await recordings; DirectML is neither available nor suitable for this pinned Linux environment. |
| `experiments/b2q_v1_eeg32_amd/RUN_REPORT.md` | Historical local-machine evidence. Preserved its provenance, timings and real-data claims; did not present them as this cloud machine's results. Its compute estimates discourage an unbounded faithful CPU training run. |
| `brain2qwerty_v1/README.md` | Install, data, train/eval and optional decoding. Pinned setup and V1 CLI are verified. No checkpoints, real data, ARPA language model or real decoding outputs exist for this cloud run. |
| `brain2qwerty_v2/README.md` | V2 workflow, embargoed data and model dependencies. Existing CPU tests exercise V2 model/loss components, not LLM-assisted full training. Dataset/model/GPU prerequisites remain absent. Python floor in this document is stale. |
| `CONTRIBUTING.md` | Validation/style guidance. Added meaningful harness failure/tool tests and ran Ruff on changed Python files. Native model changes, upstream PR and CLA submission are not part of this pass. |
| `CODE_OF_CONDUCT.md` | Community guidance, not an execution script. No messages or reports sent to anyone. |
| Harness `README.md` | Current runnable workflow. Added operator inspector and NPZ fixture commands, exact schemas, provenance limitations and links to this review/session packet. |
| `docs/ORIGINAL_START_HERE.md` | Original kit's intended workflow. Built/ran setup, self-tests, read-only audit, native smoke and manual experiment. Implemented the missing local fixture checker and deterministic inspection support; structured/minimal preparation now differs in context/inspection while retaining identical candidate/task/rules. Actual CLI context ablation remains blocked. |
| `docs/ORIGINAL_BUILD_VERIFICATION.md` | Historical verification of unavailable code. Its 0.573770/0.688525 metrics and scripted loop were not reproduced or claimed. Current implementation has independent metrics and real failure evidence. |
| `docs/SESSION_BRIEF.md` | Session presentation and five questions. Produced `SESSION_PACKET.md` with concrete evidence, qualified answers and a short demonstration sequence. The old session date/agenda are not verified current scheduling instructions. |
| `verification/BUILD_VERIFICATION.md` | Current evidence inventory. Extended it with tool test results and separately labelled synthetic NPZ checks. Native ordering and proxy-network blockers stay visible. |
| `verification/DEV_DAY_REPORT.md` | Actual manual experiment comparison. Candidate passed contracts but was worse on the synthetic fixture, so it was not promoted. Linked the richer current session packet rather than inventing a failing Codex architecture trace. |

## Discrepancies resolved against executable code

1. The technical handoff calls V1's training path CTC. `brain2qwerty_v1/config/xp_config.py` specifies `CrossEntropyLoss`, and `BrainModule` uses per-keystroke cross-entropy. CTC is part of V2. No loss was changed to match stale prose.
2. The V2 README says Python 3.10+. `pyproject.toml` requires Python >=3.12 for this checkout. Both prepared environments use 3.12.14.
3. Uploaded build verification says Codex CLI/native dependencies were absent in its build environment. They are present here. The actual nested CLI is blocked by HTTP CONNECT 403; native forward/backward works but the stricter interleaved-order contract fails. Historical absence and present access failure are different findings.
4. Original fixture and inspector instructions referenced files the unavailable kit would have supplied. Their existence was not assumed. New implementations now provide checked shapes/gradients/source hashes and operator-local NPZ validation.
5. Native V1 groups by `sentence_UID` and emits grouped rows. The proxy restores arbitrary interleaved order and uses a composite subject/sentence identity. The stricter proxy contract must be distinguished from native assumptions and sampler behavior; passing the proxy is not proof of full native fidelity.

## What this pass produced

- A source-hashed tensor/gradient inspector, baseline and width-adapter inspection JSON.
- An operator-local NPZ checker, explicitly synthetic 32/61-channel fixture reports; no EEG uploaded or downloaded.
- Five additional tool tests, alongside five existing controller tests. Preparation conditions share model/task/rules; structured adds mathematical context and a starting-model inspection report.
- This document review and an evidence-based session packet answering all five supplied questions.
- Updated source, verification and portable ZIP on the existing harness branch.

Remaining required outside capabilities for the larger goals: a usable nested Codex model connection for autonomous/context-ablation experiments, an actual operator fixture for real-window interface evidence, and data plus Linux CUDA compute for faithful FULL/EEG32 training. No credential values are requested in chat and no paid infrastructure was provisioned.
