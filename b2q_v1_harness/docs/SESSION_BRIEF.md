# Dev Day — September 29, 2026

## 20-second explanation
“I’m testing agentic CI for ML experimentation. My test case is Brain2Qwerty V1. I built a small
CPU environment where Codex changes an upstream representation and has to preserve sensor semantics,
sentence ordering, masks and gradients. An external runner checks, trains and accepts or rejects it.
I’m bringing the actual failure traces to understand what the harness should expose or enforce.”

## What to show
Open the latest DEV_DAY_REPORT.md, the first failing *.evaluation.json, the corresponding *.patch,
and the candidate math_diff.json. Separate seeded faults from real Codex mistakes.
A failed native_smoke.json is valuable environment evidence; do not hide it behind proxy success.

## Five questions grounded in this run
1. When the agent misses a mathematical dependency, what tells you to improve context versus build
   a deterministic inspection tool versus narrow the allowed edits?
2. How would you expose tensor/gradient and data-identity contracts to Codex without stale docs
   becoming more authoritative than executable code?
3. What is the practical laptop boundary between agent-writable work and a protected evaluator?
4. How should experiment memory and recovery work across fresh sessions after repeated failures?
5. How do you validate that the cheap testbed preserves the important failure modes of the real model?

## Keep the scope honest
This fits the CI side of “agentic CI/CD”; no real deployment is implemented. Architecture experiments
add ML-specific concerns beyond normal software CI. You did not need to specify the dataset in the
original session request to discuss those harness questions. The session's exact agenda is not verified.
Don't say “normal CI is easy” or claim the session will solve the underlying neuroscience.
