# Developer harness report

**Current embedding evidence:** see [the 100-run embedding comparison](embedding_comparison/REPORT.md), including width, projection, activation and normalization changes, per-seed errors, loss curves and measured embedding geometry. The single candidate below is retained as earlier evidence and is insufficient to assess the intervention by itself.

See BUILD_VERIFICATION.md for scope, limitations and provenance.

See SESSION_PACKET.md for the demonstration sequence and evidence-based answers to all five uploaded session questions. MD_REVIEW.md records the complete source-document review and remaining prerequisites.

```json
{
  "execution_route": "manual active Codex session",
  "baseline": {
    "seeds": [
      7,
      17
    ],
    "updates_per_seed": 160,
    "raw_character_error": [
      0.08196721225976944,
      0.08196721225976944
    ],
    "mean_raw_character_error": 0.08196721225976944,
    "scope": "synthetic developer smoke; not paper CER or EEG results"
  },
  "candidate": {
    "seeds": [
      7,
      17
    ],
    "updates_per_seed": 160,
    "raw_character_error": [
      0.1147540956735611,
      0.09836065769195557
    ],
    "mean_raw_character_error": 0.10655737668275833,
    "scope": "synthetic developer smoke; not paper CER or EEG results"
  },
  "promoted": false
}
```
