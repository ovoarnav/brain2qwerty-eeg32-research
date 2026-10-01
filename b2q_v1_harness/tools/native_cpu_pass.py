# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
from brain2qwerty_v1.eeg import EEG32_CHANNELS, EEG61_CHANNELS
from brain2qwerty_v1.pl_module import BrainModule
from brain2qwerty_v1.utils import _full_eeg_layout_positions, materialize_lazy_params
from neuraltrain.models import BaseModelConfig


def peak_rss_mib():
    try:
        import resource
    except ImportError:
        return None
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def build_module(device="cpu", encoder_config=None, transformer_config=None):
    encoder_config = ENCODER if encoder_config is None else encoder_config
    transformer_config = TRANSFORMER if transformer_config is None else transformer_config
    width = encoder_config["hidden"]
    with torch.device(device):
        return BrainModule(
            model=BaseModelConfig(**encoder_config).build(
                n_in_channels=32, n_outputs=width
            ),
            transformer=BaseModelConfig(**transformer_config).build(dim=width),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={},
            optimizer=None,
        )


def checkpoint_candidates():
    roots = [
        Path("/workspace/results/eeg32"),
        Path("/workspace/results/brain2qwerty/eeg32"),
    ]
    return sorted({p for root in roots for p in root.glob("best*.ckpt")})


def restore_checkpoint(path, trusted=False, module_factory=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is not on this machine: {path}")
    # Memory-map optimizer tensors too, but never construct an optimizer or resume training.
    # PyTorch's restricted loader is the default; the explicit trusted option also
    # supports Lightning files that pickle the repo's optimizer/metric configuration.
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=not trusted)
    if not isinstance(payload, dict):
        raise ValueError("Expected a native V1 state_dict or Lightning checkpoint")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing native V1 state_dict")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("The state_dict must contain named tensors")
    if not all(
        any(key.startswith(prefix) for key in state)
        for prefix in ("model.", "transformer.", "linear.")
    ):
        raise ValueError(
            "This is not the native BrainModule Conv + sentence-transformer checkpoint"
        )
    module = (module_factory or (lambda: build_module(device="meta")))()
    # Assign CPU checkpoint tensors into a meta skeleton: avoid an extra 2.5 GB
    # randomly initialized model and reject missing/incompatible weights strictly.
    module.load_state_dict(state, strict=True, assign=True)
    for value in list(module.parameters()) + list(module.buffers()):
        if (
            isinstance(value, torch.nn.UninitializedParameter)
            or value.device.type != "cpu"
        ):
            raise ValueError(
                "Checkpoint left a parameter or buffer uninitialized/off CPU"
            )
    module.eval()
    info = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "format": "lightning" if "state_dict" in payload else "state_dict",
        "strict_state_dict_load": True,
        "weights_only_loader": not trusted,
        "optimizer_restored": False,
    }
    if payload.get("verification_fixture_kind") == "generated-untrained-full-V1":
        info["verification_fixture_kind"] = payload["verification_fixture_kind"]
    for key in ("epoch", "global_step"):
        if isinstance(payload.get(key), int):
            info[key] = payload[key]
    return module, info


def synthetic_arrays(rows=2, seed=33):
    if not 1 <= rows <= 64:
        raise ValueError("Synthetic row count must be between 1 and 64")
    generator = np.random.default_rng(seed)
    mapping = _full_eeg_layout_positions()
    reference = np.asarray([mapping[name] for name in EEG61_CHANNELS])
    spread = np.ptp(reference, axis=0)
    spread[spread == 0] = 1
    coordinates = (
        np.asarray([mapping[name] for name in EEG32_CHANNELS]) - reference.min(0)
    ) / spread
    return {
        "neuro": generator.standard_normal((rows, 32, 25)).astype(np.float32),
        "channel_positions": np.broadcast_to(
            coordinates.astype(np.float32), (rows, 32, 2)
        ).copy(),
        "subject_id": np.zeros((rows, 1), dtype=np.int64),
        "sentence_uids": np.asarray([f"synthetic-{i // 2}" for i in range(rows)]),
        "channel_names": np.asarray(EEG32_CHANNELS),
    }


def read_fixture(path, split="test"):
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 256 * 1024 * 1024:
            raise ValueError("NPZ fixture exceeds 256 MiB uncompressed")
    with np.load(path, allow_pickle=False) as data:
        if "neuro" in data:
            required = {"neuro", "channel_positions", "subject_id", "sentence_uids"}
            if not required.issubset(data.files):
                raise ValueError(f"Native fixture requires {sorted(required)}")
            names = required | (
                {"feature", "channel_names", "native_untrained_only"} & set(data.files)
            )
            return {name: data[name].copy() for name in names}
        # Consume the existing real-data dataset_gate.py output without changing it.
        prefix = f"{split}_"
        required = {
            prefix + name
            for name in ("neuro", "channel_positions", "subject_id", "segment_signatures")
        }
        if not required.issubset(data.files):
            raise ValueError(f"No native fixture or dataset-gate {split} batch found")
        arrays = {
            name: data[prefix + name].copy()
            for name in ("neuro", "channel_positions", "subject_id")
        }
        signatures = data[prefix + "segment_signatures"]
        arrays["sentence_uids"] = np.asarray(
            [json.loads(str(value))["sentence_uid"] for value in signatures]
        )
        if prefix + "feature" in data:
            arrays["feature"] = data[prefix + "feature"].copy()
    return arrays


def validate_weight_mode(arrays, untrained):
    if "native_untrained_only" in arrays:
        marker = arrays["native_untrained_only"]
        if marker.shape != () or marker.dtype.kind != "b":
            raise ValueError("native_untrained_only must be a scalar boolean")
        if bool(marker) and not untrained:
            raise ValueError(
                "This recorded-input bridge lacks the trained preprocessing/subject mapping; it is restricted to --untrained"
            )


def prepare_batch(arrays, max_rows=64):
    x = arrays["neuro"]
    if x.ndim != 3:
        raise ValueError("neuro must be [N,32,25]")
    rows, channels, points = x.shape
    if not (1 <= rows <= max_rows and channels == 32 and points == 25):
        raise ValueError(
            "Expected a bounded EEG32 batch [N,32,25] from the native 50 Hz / 0.5 s pipeline"
        )
    coords = arrays["channel_positions"]
    if arrays["subject_id"].shape not in {(rows,), (rows, 1)}:
        raise ValueError("subject_id must be [N] or [N,1]")
    subjects = arrays["subject_id"].reshape(-1)
    uids = arrays["sentence_uids"]
    if (
        coords.shape != (rows, 32, 2)
        or subjects.shape != (rows,)
        or uids.shape != (rows,)
    ):
        raise ValueError(
            "Signals, coordinates, subjects and sentence identities are misaligned"
        )
    if x.dtype.kind != "f" or coords.dtype.kind != "f" or subjects.dtype.kind not in "iu":
        raise ValueError(
            "Expected floating-point signals/coordinates and integer subject indices"
        )
    if not np.isfinite(x).all() or not np.isfinite(coords).all():
        raise ValueError("Signals and coordinates must be finite")
    if uids.dtype.kind not in "USiu":
        raise ValueError("Sentence identities must be strings or integer codes")
    if subjects.dtype.kind == "u" and subjects.max() > np.iinfo(np.int64).max:
        raise ValueError("Subject indices exceed int64")
    if subjects.min() < 0:
        raise ValueError(
            "Subject indices must be nonnegative and match the training encoder"
        )
    verified = "channel_names" in arrays
    if verified and tuple(arrays["channel_names"].tolist()) != EEG32_CHANNELS:
        raise ValueError(
            "Channel names/order differ from the exact trained EEG32 montage"
        )
    groups = OrderedDict()
    for i, uid in enumerate(uids.tolist()):
        groups.setdefault(str(uid), []).append(i)
    for indices in groups.values():
        if len(set(subjects[indices].tolist())) != 1:
            raise ValueError(
                "Native sentence_UID must uniquely identify a sentence across subjects"
            )
    order = np.asarray(
        [i for indices in groups.values() for i in indices], dtype=np.int64
    )
    data = {
        "neuro": torch.from_numpy(np.asarray(x[order], dtype=np.float32).copy()),
        "channel_positions": torch.from_numpy(
            np.asarray(coords[order], dtype=np.float32).copy()
        ),
        "subject_id": torch.from_numpy(
            np.asarray(subjects[order], dtype=np.int64).copy()
        ).unsqueeze(1),
    }
    labels = None
    if "feature" in arrays:
        if arrays["feature"].shape not in {(rows,), (rows, 1)}:
            raise ValueError("feature must be [N] or [N,1]")
        targets = arrays["feature"].reshape(-1)
        if (
            targets.shape != (rows,)
            or targets.dtype.kind not in "iu"
            or targets.min() < 0
            or targets.max() >= 29
        ):
            raise ValueError("feature must contain one integer class in [0,28] per row")
        labels = torch.from_numpy(np.asarray(targets, dtype=np.int64).copy())
        data["feature"] = labels[torch.from_numpy(order)].unsqueeze(1)
    segments = [
        SimpleNamespace(trigger=SimpleNamespace(extra={"sentence_UID": str(uids[i])}))
        for i in order
    ]
    batch = SimpleNamespace(data=data, segments=segments)
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        digest.update(name.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    info = {
        "input_shape": list(x.shape),
        "position_shape": list(coords.shape),
        "sentence_groups": len(groups),
        "grouped_input_indices": order.tolist(),
        "input_sha256": digest.hexdigest(),
        "montage_names_verified": verified,
    }
    return batch, torch.from_numpy(np.argsort(order)), labels, info


def gradient_summary(module, batch):
    stages = {}
    for prefix in (
        "model.merger.",
        "model.initial_linear.",
        "model.encoder.",
        "model.time_agg_out.",
        "transformer.",
        "linear.",
    ):
        grads = [
            p.grad
            for name, p in module.named_parameters()
            if name.startswith(prefix) and p.grad is not None
        ]
        stages[prefix.rstrip(".")] = {
            "parameters_with_gradient": len(grads),
            "finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
            "nonzero": any(bool(g.abs().sum() > 0) for g in grads),
        }
    inputs = {}
    for name in ("neuro", "channel_positions"):
        grad = batch.data[name].grad
        inputs[name] = {
            "present": grad is not None,
            "finite": grad is not None and bool(torch.isfinite(grad).all()),
            "nonzero": grad is not None and bool(grad.abs().sum() > 0),
        }
    all_grads = [p.grad for p in module.parameters() if p.grad is not None]
    finite = bool(all_grads) and all(bool(torch.isfinite(g).all()) for g in all_grads)
    passed = (
        finite
        and all(s["finite"] and s["nonzero"] for s in stages.values())
        and all(s["finite"] and s["nonzero"] for s in inputs.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "all_present_parameter_gradients_finite": finite,
        "stages": stages,
        "inputs": inputs,
    }


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(33)
    checkpoint = args.checkpoint
    if not args.untrained and checkpoint is None:
        candidates = checkpoint_candidates()
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Found {len(candidates)} native best checkpoints; specify --checkpoint PATH. The RunPod volume is not automatically mounted here."
            )
        checkpoint = candidates[0]
    arrays = (
        read_fixture(args.fixture, args.split)
        if args.fixture
        else synthetic_arrays(args.rows)
    )
    validate_weight_mode(arrays, args.untrained)
    batch, inverse, labels, info = prepare_batch(arrays, max_rows=args.max_rows)
    print("[native_cpu_pass] build full native V1 / load checkpoint", flush=True)
    if args.untrained:
        module = build_module()
        materialize_lazy_params(module, [batch])
        module.eval()
        weights = {"kind": "explicit-untrained-native-V1", "checkpoint_loaded": False}
    else:
        module, checkpoint_info = restore_checkpoint(
            checkpoint, trusted=args.trusted_checkpoint
        )
        weights = {
            "kind": "checkpoint-native-V1",
            "checkpoint_loaded": True,
            **checkpoint_info,
        }
    if batch.data["subject_id"].max() >= module.model.merger.heads.shape[0]:
        raise ValueError(
            "Subject index is outside the checkpoint's per-subject channel merger"
        )
    loaded_seconds = time.perf_counter() - started
    print(
        "[native_cpu_pass] native Conv + temporal attention + sentence transformer + head",
        flush=True,
    )
    if args.backward:
        for name in ("neuro", "channel_positions"):
            batch.data[name].requires_grad_()
    forward_started = time.perf_counter()
    with torch.set_grad_enabled(args.backward):
        embeddings = module.forward(batch)
        grouped_logits = module._transformer_forward(batch, embeddings)
        logits = grouped_logits[inverse]
        if embeddings.shape != (len(inverse), ENCODER["hidden"]) or logits.shape != (
            len(inverse),
            29,
        ):
            raise ValueError("Native representation/logit shapes are incompatible")
        if not torch.isfinite(embeddings).all() or not torch.isfinite(logits).all():
            raise ValueError("Native embeddings/logits contain nonfinite values")
        forward_seconds = time.perf_counter() - forward_started
        gradients = None
        if args.backward:
            # Unlabelled synthetic windows only establish connectivity, not accuracy.
            objective = (
                module.loss(logits, labels)
                if labels is not None
                else logits.square().mean()
            )
            backward_started = time.perf_counter()
            objective.backward()
            backward_seconds = time.perf_counter() - backward_started
            gradients = gradient_summary(module, batch)
            gradients.update(
                objective="cross_entropy"
                if labels is not None
                else "mean_squared_logits_connectivity_probe",
                objective_value=float(objective.detach()),
                backward_seconds=backward_seconds,
            )
            if gradients["status"] != "PASS":
                raise ValueError(f"Native gradient connectivity failed: {gradients}")
    metrics = None
    if labels is not None:
        metrics = {
            "cross_entropy": float(module.loss(logits.detach(), labels)),
            "character_mismatch": float(
                (logits.detach().argmax(1) != labels).float().mean()
            ),
            "scope": "provided batch only; no full held-out dataset or sequence CER claim",
        }
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        # Preserve original input row order; avoid exporting subject/recording metadata.
        np.savez_compressed(
            args.predictions,
            logits=logits.detach().numpy(),
            embeddings=embeddings.detach()[inverse].numpy(),
        )
    roundtrip = None
    if args.verify_checkpoint_roundtrip:
        if not args.untrained or args.backward:
            raise ValueError(
                "Roundtrip verification requires --untrained without --backward"
            )
        print(
            "[native_cpu_pass] full native generated-weight checkpoint roundtrip",
            flush=True,
        )
        roundtrip_started = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix="native-v1-untrained-roundtrip-"
        ) as folder:
            path = Path(folder) / "generated-untrained-full-V1.ckpt"
            torch.save(
                {
                    "state_dict": module.state_dict(),
                    "verification_fixture_kind": "generated-untrained-full-V1",
                },
                path,
            )
            clone, checkpoint_info = restore_checkpoint(path)
            with torch.no_grad():
                restored_embeddings = clone.forward(batch)
                restored_logits = clone._transformer_forward(batch, restored_embeddings)[
                    inverse
                ]
            torch.testing.assert_close(restored_embeddings, embeddings, atol=0, rtol=0)
            torch.testing.assert_close(restored_logits, logits, atol=0, rtol=0)
            roundtrip = {
                "status": "PASS",
                "scope": "generated untrained native weights; not the user's GPU checkpoint",
                "checkpoint_bytes": checkpoint_info["bytes"],
                "strict_state_dict_load": checkpoint_info["strict_state_dict_load"],
                "embeddings_and_logits_bitwise_equal": True,
                "temporary_checkpoint_removed": True,
                "elapsed_seconds": time.perf_counter() - roundtrip_started,
            }
    source_hashes = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "brain2qwerty_v1/pl_module.py",
            "brain2qwerty_v1/config/model_config.py",
            "brain2qwerty_v1/eeg.py",
        )
    }
    return {
        "status": "PASS",
        "scope": "actual full Brain2Qwerty V1 EEG32 CPU pass; no proxy or architecture reduction",
        "backend": "cpu",
        "torch": torch.__version__,
        "threads": args.threads,
        "architecture": {"encoder": ENCODER, "transformer": TRANSFORMER},
        "native_source_sha256": source_hashes,
        "weights": weights,
        "input_kind": "provided-local-NPZ" if args.fixture else "synthetic-noise",
        "fixture_restricted_to_untrained": bool(
            arrays.get("native_untrained_only", False)
        ),
        **info,
        "parameters": sum(p.numel() for p in module.parameters()),
        "embedding_shape": list(embeddings.shape),
        "logits_shape": list(logits.shape),
        "finite_embeddings_and_logits": True,
        "original_row_order_restored": True,
        "model_in_eval_mode": not module.training,
        "load_and_setup_seconds": loaded_seconds,
        "forward_seconds": forward_seconds,
        "gradients": gradients,
        "batch_metrics": metrics,
        "checkpoint_roundtrip": roundtrip,
        "peak_rss_mib": peak_rss_mib(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the full native EEG32 V1 on CPU, optionally loading its trained checkpoint. Uses the original V1 environment."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--checkpoint", type=Path)
    modes.add_argument(
        "--untrained",
        action="store_true",
        help="Explicit architecture-only pass; never reported as checkpoint evaluation",
    )
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="Allow pickle configuration in your own trusted Lightning checkpoint",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Local native EEG32 NPZ batch or existing dataset_gate.py NPZ",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="Synthetic noise windows when no fixture is supplied",
    )
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also verify native upstream gradient connectivity; does not update weights",
    )
    parser.add_argument(
        "--verify-checkpoint-roundtrip",
        action="store_true",
        help="With --untrained, save/load a temporary full native state_dict and compare exact predictions",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional local NPZ output in original input row order",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "b2q_v1_harness/results/native_cpu_pass.json",
    )
    args = parser.parse_args()
    try:
        if args.threads < 1 or not 1 <= args.max_rows <= 256:
            raise ValueError("Use positive thread count and max-rows in [1,256]")
        if args.verify_checkpoint_roundtrip and (not args.untrained or args.backward):
            raise ValueError(
                "Roundtrip verification requires --untrained without --backward"
            )
        report = run(args)
    except Exception as error:
        report = {
            "status": "BLOCKED",
            "scope": "native V1 CPU pass did not complete",
            "checkpoint_pass_completed": False,
            "error": f"{type(error).__name__}: {error}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
