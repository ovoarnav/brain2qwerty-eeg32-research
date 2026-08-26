"""Inspect one real-data condition and emit machine-checkable gate artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import studies  # noqa: F401  (registers Pinet2024Eeg)
from brain2qwerty_v1.config.xp_config import eeg_debug_config
from brain2qwerty_v1.main import Experiment


def _hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _segment_signature(segment) -> dict:
    return {
        "start": float(segment.start),
        "duration": float(segment.duration),
        "sentence_uid": str(segment.trigger.extra["sentence_UID"]),
        "button": str(segment.trigger.extra.get("button")),
        "subject": str(segment.trigger.extra.get("subject")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("full", "eeg32"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    args = parser.parse_args()

    cfg = eeg_debug_config(args.condition)
    cfg["data"].update(
        batch_size=64,
        val_batch_size=64,
        test_batch_size=64,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    data = Experiment(**cfg).data
    events = data.build_events()
    loaders = data.build()

    neural_rows = events[events.type == "Eeg"]
    keystrokes = events[events.type == "Keystroke"]
    split_counts = {
        str(key): int(value) for key, value in keystrokes.split.value_counts().items()
    }
    sentence_split = sorted(
        {
            (str(row.sentence_UID), str(row.split))
            for row in keystrokes[["sentence_UID", "split"]].itertuples()
        }
    )

    channels_by_type = getattr(data.neuro, "_channels", {})
    channel_names = [
        name for name, _ in sorted(channels_by_type.items(), key=lambda item: item[1])
    ]
    report = {
        "status": "pass",
        "condition": args.condition,
        "study": data.study.__class__.__name__,
        "timeline_count": int(neural_rows.shape[0]),
        "keystrokes": int(keystrokes.shape[0]),
        "split_counts": split_counts,
        "sentence_split_hash": _hash_json(sentence_split),
        "sentence_split_count": len(sentence_split),
        "channel_count": len(channel_names),
        "channel_names": channel_names,
        "splits": {},
    }

    arrays = {}
    for split, loader in loaders.items():
        signatures = [_segment_signature(segment) for segment in loader.dataset.segments]
        batch = next(iter(loader))
        split_report = {
            "segments": len(signatures),
            "segment_hash": _hash_json(signatures),
            "first_batch_segment_hash": _hash_json(
                [_segment_signature(segment) for segment in batch.segments]
            ),
            "shapes": {name: list(value.shape) for name, value in batch.data.items()},
        }
        report["splits"][split] = split_report
        for name, value in batch.data.items():
            arrays[f"{split}_{name}"] = value.detach().cpu().numpy()
        arrays[f"{split}_segment_signatures"] = np.asarray(
            [
                json.dumps(_segment_signature(segment), sort_keys=True)
                for segment in batch.segments
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.tensors, **arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
