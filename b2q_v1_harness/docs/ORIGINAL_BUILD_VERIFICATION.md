# Build verification — what actually ran

Build/test environment: Linux, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5.
Windows installation and your existing V1 environment were NOT available for verification.
The original private fork was inspected read-only through GitHub; it was not cloned into this environment.

## Executed

- Python compilation checks: passed.
- Controller failure regressions: **3/3 passed** (training exception, stale output, unsuccessful process exit cannot become PASS).
- Baseline contracts: **16/16 passed**.
- Fault injections: **7/7 caught**.
- Fixed 160-step synthetic training on seeds 7 and 17: completed.
- Full controller loop with a clearly labelled SCRIPTED test double: completed.
  This exercised candidate writing, external checks, training, rejection, event parsing, patch capture and reporting.
  It did NOT test an actual Codex model, Codex authentication, or Codex sandbox enforcement.

## Informative result

The scripted repair added a trainable 48->32 adapter. Its representation shapes were
local [41,48], contextual [41,32], fused [41,32]. All 16 checks passed.

On this tiny, reused synthetic developer set (61 characters), mean raw character error was:
- baseline: 0.573770
- scripted width-migration candidate: 0.688525

The controller did NOT promote it. This demonstrates that mathematical interface validity and
empirical usefulness are different gates. These numbers are NOT real EEG results, paper CER,
or evidence that one width is generally better. They are a two-seed developer smoke experiment.

## Seven injected failures

- wrong_output_shape: caught by finite_one_logit_row_per_keystroke, channel_and_window_shape_variants, single_keystroke, upstream_gradient_connectivity, baseline_disabled_path_equivalence.
- detached_upstream: caught by upstream_gradient_connectivity.
- padding_mask_removed: caught by ragged_padding_invariance, baseline_disabled_path_equivalence.
- cross_sentence_mix: caught by sentence_isolation, ragged_padding_invariance, subject_sentence_composite_identity, baseline_disabled_path_equivalence.
- row_order_lost: caught by interleaved_rows_restore_original_order.
- label_leak: caught by label_canary_independence.
- width_mismatch: caught by finite_one_logit_row_per_keystroke, channel_and_window_shape_variants, single_keystroke, interleaved_rows_restore_original_order, joint_sensor_coordinate_permutation, coordinate_path_has_effect, sentence_isolation, ragged_padding_invariance, subject_sentence_composite_identity, label_canary_independence, upstream_gradient_connectivity, checkpoint_roundtrip, declared_representation_shapes.

## Not run

- Actual Codex/LLM experiment: CLI not installed in build environment.
- Original V1/neuraltrain native smoke: original environment/dependencies not installed here.
- Real EEG training/evaluation, EEG32-versus-FULL comparison, or KenLM decoding: not attempted.
- Audit against a local checkout: attach-repo must be run with your local repository.
- Context ablation with actual Codex: not run; instructions are supplied.

Read START_HERE.md for the commands that produce the missing local evidence. A local environment
failure should be preserved in the report, not hidden with an unlabelled surrogate fallback.
