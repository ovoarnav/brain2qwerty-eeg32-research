"""Verify that FULL and EEG32 real-data gates differ only by channel picks."""

import argparse
import json
from pathlib import Path

import numpy as np

from brain2qwerty_v1.eeg import EEG32_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--full-tensors", type=Path, required=True)
    parser.add_argument("--eeg32-report", type=Path, required=True)
    parser.add_argument("--eeg32-tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_report = json.loads(args.full_report.read_text(encoding="utf-8"))
    eeg32_report = json.loads(args.eeg32_report.read_text(encoding="utf-8"))
    full = np.load(args.full_tensors)
    eeg32 = np.load(args.eeg32_tensors)

    full_names = full_report["channel_names"]
    eeg32_names = eeg32_report["channel_names"]
    eeg32_indices = [full_names.index(name) for name in EEG32_CHANNELS]
    checks = {
        "full_channel_count_61": len(full_names) == 61,
        "eeg32_channel_count_32": len(eeg32_names) == 32,
        "eeg32_exact_order": tuple(eeg32_names) == EEG32_CHANNELS,
        "sentence_split_identical": full_report["sentence_split_hash"]
        == eeg32_report["sentence_split_hash"],
        "split_counts_identical": full_report["split_counts"]
        == eeg32_report["split_counts"],
    }
    split_details = {}
    for split in sorted(set(full_report["splits"]) & set(eeg32_report["splits"])):
        full_x = full[f"{split}_neuro"][:, eeg32_indices, :]
        eeg32_x = eeg32[f"{split}_neuro"]
        full_pos = full[f"{split}_channel_positions"][:, eeg32_indices, :]
        eeg32_pos = eeg32[f"{split}_channel_positions"]
        detail = {
            "segments_identical": full_report["splits"][split]["segment_hash"]
            == eeg32_report["splits"][split]["segment_hash"],
            "first_batch_segments_identical": np.array_equal(
                full[f"{split}_segment_signatures"],
                eeg32[f"{split}_segment_signatures"],
            ),
            "labels_identical": np.array_equal(
                full[f"{split}_feature"], eeg32[f"{split}_feature"]
            ),
            "subject_ids_identical": np.array_equal(
                full[f"{split}_subject_id"], eeg32[f"{split}_subject_id"]
            ),
            "selected_signal_max_abs_diff": float(np.max(np.abs(full_x - eeg32_x))),
            "selected_positions_max_abs_diff": float(
                np.max(np.abs(full_pos - eeg32_pos))
            ),
        }
        detail["selected_signal_identical"] = bool(
            np.allclose(full_x, eeg32_x, rtol=1e-6, atol=1e-6)
        )
        detail["selected_positions_identical"] = bool(
            np.allclose(full_pos, eeg32_pos, rtol=0.0, atol=0.0)
        )
        split_details[split] = detail

    passed = all(checks.values()) and all(
        all(value for key, value in detail.items() if key.endswith("identical"))
        for detail in split_details.values()
    )
    report = {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "splits": split_details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
