import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_cpu_pass import (  # noqa: E402
    EEG32_CHANNELS,
    ENCODER,
    TRANSFORMER,
    build_module,
    peak_rss_mib,
    prepare_batch,
    read_fixture,
    restore_checkpoint,
    synthetic_arrays,
    validate_weight_mode,
)

VARIANTS = ("identity", "zero", "scale_half", "scale_double", "unit_l2")


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(str((parameter.shape, parameter.dtype)).encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def replay_arrays(root):
    coordinates = synthetic_arrays(1)["channel_positions"][0]
    windows, identities = [], []
    counts = {}
    for subject in ("002", "006", "007", "010"):
        path = Path(root) / f"outputs/replay_assets/replay_{subject}.npz"
        with np.load(path, allow_pickle=True) as replay:
            np.testing.assert_array_equal(replay["channels"], np.asarray(EEG32_CHANNELS))
            count = 0
            for index, epoch in enumerate(replay["epochs_uv"]):
                epoch = np.asarray(epoch, dtype=np.float32)
                if (
                    epoch.ndim != 3
                    or epoch.shape[1:] != (32, 50)
                    or not np.isfinite(epoch).all()
                ):
                    raise ValueError("Expected finite recorded windows [N,32,50]")
                windows.append(
                    resample_poly(epoch * 1e-6, 1, 2, axis=-1).astype(np.float32)
                )
                identities.extend([f"shape-{subject}-{index}"] * len(epoch))
                count += len(epoch)
        counts[subject] = count
    neuro = np.concatenate(windows)
    return {
        "neuro": neuro,
        "channel_positions": np.broadcast_to(coordinates, (len(neuro), 32, 2)).copy(),
        "subject_id": np.zeros((len(neuro), 1), dtype=np.int64),
        "sentence_uids": np.asarray(identities),
        "channel_names": np.asarray(EEG32_CHANNELS),
        "native_untrained_only": np.asarray(True),
    }, {
        "kind": "surviving-recorded-EEG-replay",
        "participant_windows": counts,
        "conversion": "microvolts to volts, anti-aliased 100 Hz to 50 Hz",
        "native_training_preprocessing_reproduced": False,
        "trained_subject_mapping_recovered": False,
        "labels_used": False,
    }


def encode_in_chunks(module, batch, chunk_size):
    if chunk_size < 1:
        raise ValueError("Encoder chunk size must be positive")
    if module.training:
        raise ValueError("Chunked native encoder requires eval mode")
    encoded = []
    with torch.no_grad():
        for start in range(0, len(batch.segments), chunk_size):
            stop = min(start + chunk_size, len(batch.segments))
            part = SimpleNamespace(
                data={name: value[start:stop] for name, value in batch.data.items()}
            )
            encoded.append(module.forward(part))
            print(
                f"[native_embedding_probe] encoded {stop}/{len(batch.segments)} windows",
                flush=True,
            )
    return torch.cat(encoded)


def intervene(embeddings, name):
    if name == "identity":
        return embeddings
    if name == "zero":
        return torch.zeros_like(embeddings)
    if name == "scale_half":
        return embeddings * 0.5
    if name == "scale_double":
        return embeddings * 2
    if name == "unit_l2":
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    raise ValueError(f"Unknown intervention: {name}")


def measure_variants(module, batch, embeddings, inverse, labels=None):
    records = []
    module.eval()
    with torch.no_grad():
        baseline = module._transformer_forward(batch, embeddings)[inverse]
        for name in VARIANTS:
            started = time.perf_counter()
            changed = intervene(embeddings, name)
            logits = module._transformer_forward(batch, changed)[inverse]
            if logits.shape != baseline.shape or not torch.isfinite(logits).all():
                raise ValueError(
                    "Embedding intervention broke the native output contract"
                )
            delta = logits - baseline
            if name == "identity":
                torch.testing.assert_close(logits, baseline, atol=0, rtol=0)
            row = {
                "variant": name,
                "embedding_shape": list(changed.shape),
                "mean_embedding_l2": float(changed.norm(dim=-1).mean()),
                "logits_shape": list(logits.shape),
                "logit_delta_rms": float(delta.square().mean().sqrt()),
                "logit_delta_max_abs": float(delta.abs().max()),
                "argmax_change_fraction": float(
                    (logits.argmax(1) != baseline.argmax(1)).float().mean()
                ),
                "finite_logits": True,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if labels is not None:
                row.update(
                    cross_entropy=float(module.loss(logits, labels)),
                    character_mismatch=float((logits.argmax(1) != labels).float().mean()),
                    metric_scope="provided native batch only, not sequence CER or a full scientific split",
                )
            records.append(row)
            print(
                f"[native_embedding_probe] {name}: delta RMS={row['logit_delta_rms']:.6g}, changed argmax={row['argmax_change_fraction']:.3f}",
                flush=True,
            )
    return records


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.handoff:
        if not args.untrained:
            raise ValueError(
                "The surviving replay bridge requires --untrained; trained preprocessing and subject mapping are missing"
            )
        arrays, provenance = replay_arrays(args.handoff)
    else:
        arrays = read_fixture(args.fixture, args.split)
        provenance = {
            "kind": "operator-native-NPZ",
            "training_preprocessing_and_subject_mapping": "operator-supplied; not independently verified",
        }
    validate_weight_mode(arrays, args.untrained)
    batch, inverse, labels, inputs = prepare_batch(arrays, max_rows=args.max_rows)
    if args.untrained:
        module = build_module()
        # Same lazy-parameter materialization as the original production entrypoint.
        from native_cpu_pass import materialize_lazy_params

        warmup = SimpleNamespace(
            data={name: value[:1] for name, value in batch.data.items()}
        )
        materialize_lazy_params(module, [warmup])
        module.eval()
        weights = {
            "kind": "explicit-untrained-native-V1",
            "checkpoint_loaded": False,
            "seed": args.seed,
        }
    else:
        module, checkpoint_info = restore_checkpoint(
            args.checkpoint, trusted=args.trusted_checkpoint
        )
        weights = {
            "kind": "checkpoint-native-V1",
            "checkpoint_loaded": True,
            **checkpoint_info,
        }
    before = parameter_hash(module)
    encoder_started = time.perf_counter()
    embeddings = encode_in_chunks(module, batch, args.chunk_size)
    encoder_seconds = time.perf_counter() - encoder_started
    if (
        embeddings.shape != (len(inverse), ENCODER["hidden"])
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("Native encoder embeddings are invalid")
    records = measure_variants(module, batch, embeddings, inverse, labels)
    after = parameter_hash(module)
    if before != after:
        raise ValueError("Model parameters changed during an inference-only probe")
    return {
        "status": "PASS",
        "scope": "inference interventions on the actual native V1 embedding; no width migration or retraining",
        "architecture": {"encoder": ENCODER, "transformer": TRANSFORMER},
        "weights": weights,
        "input_provenance": provenance,
        **inputs,
        "encoder_chunk_size": args.chunk_size,
        "encoder_seconds": encoder_seconds,
        "parameters": sum(p.numel() for p in module.parameters()),
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "all_model_parameters_unchanged": True,
        "optimizer_created": False,
        "original_row_order_restored": True,
        "variants": records,
        "peak_rss_mib": peak_rss_mib(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe the actual V1 embedding with a fixed native transformer/head on CPU. Never trains or replaces missing weights."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--checkpoint", type=Path)
    modes.add_argument("--untrained", action="store_true")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixture", type=Path)
    inputs.add_argument("--handoff", type=Path)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (
            args.threads < 1
            or not 1 <= args.chunk_size <= 256
            or not 1 <= args.max_rows <= 4096
        ):
            raise ValueError(
                "Use positive threads, chunk-size in [1,256], max-rows in [1,4096]"
            )
        report = run(args)
    except Exception as error:
        report = {
            "status": "BLOCKED",
            "scope": "native embedding probe did not complete",
            "error": f"{type(error).__name__}: {error}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"architecture", "grouped_input_indices"}
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
