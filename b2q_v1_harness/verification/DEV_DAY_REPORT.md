# Developer harness report

See BUILD_VERIFICATION.md for scope, limitations and provenance.

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
