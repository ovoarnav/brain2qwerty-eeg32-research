import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from native_cpu_pass import EEG32_CHANNELS, synthetic_arrays
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


def metrics(truth, probabilities, classes):
    predictions = classes[probabilities.argmax(1)]
    ranks = np.argsort(probabilities, axis=1)[:, ::-1]
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(truth, probabilities, labels=classes)),
    }
    result.update(
        {
            f"top_{k}_accuracy": float(
                np.mean((classes[ranks[:, :k]] == truth[:, None]).any(1))
            )
            for k in (1, 3, 5)
        }
    )
    return result


def inspect(handoff, fixture_dir):
    handoff = Path(handoff)
    fixture_dir = Path(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(handoff / "outputs/calibration_scaling.csv")
    records = []
    for subject in ("002", "006", "007", "010"):
        prediction_path = (
            handoff / f"outputs/models/character_predictions_{subject}_motor32.npz"
        )
        replay_path = handoff / f"outputs/replay_assets/replay_{subject}.npz"
        # These are the user's own inspected NumPy replay exports. Their ragged
        # arrays/string identifiers use object dtype, as in build_replay_assets.py.
        with np.load(prediction_path, allow_pickle=True) as data:
            truth = data["y_true"].astype(str)
            probabilities = data["probabilities"].copy()
            classes = data["classes"].astype(str)
            sentences = data["sentence_UID"].astype(str)
        if (
            probabilities.shape != (len(truth), len(classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
        ):
            raise ValueError("Saved probability rows are invalid")
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        calculated = metrics(truth, probabilities, classes)
        saved = reference[
            (reference.subject == int(subject))
            & (reference.method == "personal_only")
            & (reference.budget == "maximum")
        ].iloc[0]
        max_metric_delta = max(
            abs(value - float(saved[key])) for key, value in calculated.items()
        )
        if max_metric_delta > 1e-10:
            raise ValueError("Saved predictions do not reproduce the recorded metrics")
        with np.load(replay_path, allow_pickle=True) as replay:
            channels = replay["channels"].astype(str)
            np.testing.assert_array_equal(channels, np.asarray(EEG32_CHANNELS))
            np.testing.assert_array_equal(classes, replay["classes"])
            count = 0
            shapes = []
            for index, uid in enumerate(replay["trial_ids"].astype(str)):
                rows = np.flatnonzero(sentences == uid)
                epochs = np.asarray(replay["epochs_uv"][index], dtype=np.float32)
                replay_probabilities = np.asarray(
                    replay["probabilities"][index], dtype=np.float32
                )
                if epochs.shape != (len(rows), 32, 50) or not np.isfinite(epochs).all():
                    raise ValueError("Replay EEG windows are invalid or misaligned")
                np.testing.assert_array_equal(
                    truth[rows], np.asarray(replay["truths"][index]).astype(str)
                )
                np.testing.assert_allclose(
                    probabilities[rows], replay_probabilities, atol=1e-7, rtol=1e-6
                )
                count += len(rows)
                shapes.append(list(epochs.shape))
            first = np.asarray(replay["epochs_uv"][0], dtype=np.float32)
            # An interface-only bridge: preserve the recorded physical window,
            # convert microvolts to volts and anti-alias 100 Hz -> 50 Hz. We lack
            # the trained scaler, full recording and original subject mapping.
            native = resample_poly(first * 1e-6, up=1, down=2, axis=-1).astype(np.float32)
            coords = synthetic_arrays(1)["channel_positions"][0]
            fixture = fixture_dir / f"real-replay-{subject}-untrained-only.npz"
            np.savez_compressed(
                fixture,
                neuro=native,
                channel_positions=np.broadcast_to(coords, (len(native), 32, 2)),
                subject_id=np.zeros((len(native), 1), dtype=np.int64),
                sentence_uids=np.repeat("recorded-sentence-shape-probe", len(native)),
                channel_names=channels,
                native_untrained_only=np.asarray(True),
            )
            trials = len(replay["trial_ids"])
        records.append(
            {
                "subject": subject,
                "prediction_rows": len(truth),
                "alphabet_size": len(classes),
                "recomputed_metrics": calculated,
                "max_difference_from_saved_metrics": max_metric_delta,
                "replay_trials": trials,
                "replay_windows": count,
                "replay_shapes": shapes,
                "replay_alignment": "PASS",
                "prediction_sha256": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
                "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                "native_fixture": str(fixture),
                "native_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "native_fixture_shape": list(native.shape),
            }
        )
    weight_suffixes = {".ckpt", ".pt", ".pth", ".safetensors", ".joblib", ".pkl", ".onnx"}
    checkpoints = [
        str(path.relative_to(handoff))
        for path in handoff.rglob("*")
        if path.suffix.lower() in weight_suffixes
    ]
    return {
        "status": "PASS",
        "scope": "recomputed saved predictions and real EEG replay alignment; no classifier forward or retraining",
        "model_in_bundle": "calibrated StandardScaler + LogisticRegression; fitted weights absent",
        "neural_checkpoint_candidates": checkpoints,
        "prediction_rows_total": sum(r["prediction_rows"] for r in records),
        "replay_trials_total": sum(r["replay_trials"] for r in records),
        "replay_windows_total": sum(r["replay_windows"] for r in records),
        "mean_participant_top1": float(
            np.mean([r["recomputed_metrics"]["top_1_accuracy"] for r in records])
        ),
        "mean_participant_top5": float(
            np.mean([r["recomputed_metrics"]["top_5_accuracy"] for r in records])
        ),
        "native_fixture_contract": {
            "purpose": "untrained full native V1 recorded-input execution only",
            "source_sampling_hz": 100,
            "source_window_seconds": [-0.2, 0.3],
            "source_units": "microvolts",
            "conversion": "multiply by 1e-6, scipy resample_poly up=1 down=2",
            "native_sampling_hz": 50,
            "native_training_preprocessing_reproduced": False,
            "subject_index_mapping": "placeholder zero, only for untrained execution",
            "labels_exported": False,
            "checkpoint_mode_allowed": False,
        },
        "subjects": records,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the user's trusted portable-neural-ai handoff and prepare untrained-only native CPU probes."
    )
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.handoff, args.fixture_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "subjects"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
