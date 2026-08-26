# Cloud result summaries

Runtime logs, CUDA caches, datasets, checkpoints, and raw predictions remain
outside Git. Commit only concise Markdown summaries after a completed gate:

- `environment.txt` and `pytest.log` are execution artifacts, not committed by default.
- `BENCHMARK.md` and `COST_PROJECTION.md` record the benchmark cost gate.
- `RUN_LOG.md`, `FULL_RESULT.md`, `EEG32_RESULT.md`, and `FINAL_COMPARISON.md`
  are created only from actual output; no metric is inferred or fabricated.
