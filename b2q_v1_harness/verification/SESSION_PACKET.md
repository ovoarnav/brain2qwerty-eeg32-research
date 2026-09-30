# Brain2Qwerty agentic CI session packet

## What we can demonstrate

“I have a small CPU harness that checks a representation change before training and decides whether to retain it. The new 48-to-32 adapter preserves the tested interfaces and gradients but worsens the synthetic developer metric, so it is rejected. A native bridge also exposes an ordering-contract difference with V1. The nested autonomous Codex connection is blocked by a proxy, and I kept that failure separate from the manual experiment.”

This is CI for developer experiments. It is not deployment, real EEG accuracy, or proof that width 48 is generally worse. The uploaded September 29 session date and agenda are historical and unverified.

## Five-minute demonstration

1. Open `DEV_DAY_REPORT.md` / `manual_comparison.json`: the actual manual candidate is rejected, not silently promoted.
2. Open `manual.patch` / `manual_candidate.py`: local sensor representation changes from 32 to 48 with a trainable adapter back to 32; native model source is unchanged.
3. Open `manual_inspection.json`: actual sensor output width 48, adapter/context/head widths 32, signal/coordinate gradients and adapter parameter gradients. `source_sha256` ties the evidence to code. `math_diff.json` alone is self-reported metadata; the inspector and contracts independently observe module shapes.
4. Open `selftest.json`: seven seeded faults are caught. These are deliberately constructed regressions, not actual Codex errors.
5. Open `native_smoke.json` and `cli_blocked_summary.json`: native grouped-order behavior and model-connection HTTP 403 remain explicit. There is no successful autonomous agent architecture-failure trace to show.

| Measurement | Baseline | Manual active-Codex candidate |
|---|---:|---:|
| External contracts | 16 passed | 16 passed |
| Local width | 32 | 48 |
| Contextual/fused widths | 32 / 32 | 32 / 32 |
| Seed 7 synthetic character mismatch | 0.081967 | 0.114754 |
| Seed 17 synthetic character mismatch | 0.081967 | 0.098361 |
| Mean synthetic character mismatch | 0.081967 | 0.106557 |
| Promotion | Incumbent | Rejected |

Training used 160 updates per seed and a fixed 61-character synthetic developer fixture with held-out noisy signals. Reusing it for development does not create a scientific validation set. The complete native 623.5M model CPU gates for 32 and 61 channels prove graph/update execution, not measured EEG generalization.

## 1. Context, deterministic inspection, or narrower edits?

Use deterministic checks for facts the computer can observe: tensor sizes, trainable adapter gradients, label independence, identity isolation and padding invariance. The new inspector provides actual traces and hashes rather than more prose about shapes. Supply context for the intended intervention and assumptions: change local width, preserve contextual width, and group by identity.

Narrow edits when a candidate would alter labels, preprocessing, split, evaluator, optimizer, or the actual scientific montage comparison. The current proxy task permits `candidate.py` changes only, but it is not an adversarial enforcement boundary. Our manual candidate did not miss the adapter dependency; seeded width-mismatch faults do not establish that Codex commonly does. A real matched context ablation is still needed to compare those interventions.

## 2. Expose contracts without stale docs becoming authoritative?

Use executable operator-owned contracts as the acceptance rule. `INSPECTION.json` in structured preparation describes the starting model and includes a source hash; it must be regenerated after edits. Keep task prose short and link it to observed inputs/outputs. The external evaluator reruns contracts and training on current code instead of trusting candidate claims.

The review found a concrete counterexample to trusting prose: the technical handoff calls V1's loss CTC, but the executable config specifies cross-entropy. Keep that difference in the evidence record rather than changing the loss to satisfy the document. Shape metadata is similarly insufficient without observed hooks and gradient checks.

## 3. Practical boundary between candidate and protected evaluator?

Today: candidate files live in a generated Git workspace; contracts/controller stay outside it; the CLI requests workspace-write sandbox. The external evaluator is another process, but candidate Python still has the operator user's filesystem permissions. The CLI connection failure prevented validating its actual sandbox behavior.

A stronger boundary would run candidates as a separate OS user/container with only candidate input mounted writable, evaluator/tests mounted read-only, controlled data access and process/resource limits. That is a proposed improvement, not implemented security. No sandbox bypass was used and no protection is inferred from an external process alone. Windows cleanup/memory behavior remains a limitation.

## 4. Experiment memory and recovery across sessions?

Current recovery evidence is durable source/patches, per-run IDs, source hashes, evaluation JSON, rejection/comparison reports and preserved network errors. The controller rejects stale/missing output and nonzero evaluator exits. Existing tests verify that a training exception produces fresh failed output even after contracts pass.

Cross-session automatic resumption is not implemented. Recover manually from the Git branch and a reviewed incumbent; verify code hashes, regenerate inspection, and rerun the external evaluator. Do not reinterpret an old PASS as evidence for edited code or resume from a rejected candidate as the incumbent. Token counts, when reported, are checked between calls; they are not an inside-call spending guarantee.

## 5. Does the cheap testbed preserve native failure modes?

There is useful overlap: sensor/coordinate inputs, two representation stages, ragged sentence groups, masks, gradients and checkpoints. The full native CPU gates and reduced native smoke verify that the original classes run in their own pinned stack.

There is also a demonstrated mismatch: native `_transformer_forward` emits sentence-grouped rows while the proxy restores arbitrary interleaved order. Its sampler groups sentences, so the probe does not establish a production failure. The proxy uses composite subject/sentence identities; native relies on sentence UIDs. There are no real operator windows here. The new NPZ checker verifies interfaces only, with synthetic fixtures clearly labelled. Before claiming fidelity, add operator-created windows and native/production metadata checks; neither would prove EEG accuracy by itself.

## Next runnable actions

```bash
# From b2q_v1_harness in the small environment:
python tools/inspect_candidate.py verification/manual_candidate.py --output results/current-inspection.json
python -m unittest -v test_controller test_tools
python lab.py selftest
```

With an operator-created local fixture: `python tools/fixture_smoke.py /local/fixture.npz`. It stays local and uses an untrained proxy. With a working nested Codex connection, compare `lab.py agent --rounds 1 --context minimal` and `--context structured` using the same supported model and budget; preparation differences are tested, but no effect on agent success has been established. With real data and CUDA compute, run the existing native gates/parity/pilot before scientific training. Those outside prerequisites are separate from the completed CPU harness work.
