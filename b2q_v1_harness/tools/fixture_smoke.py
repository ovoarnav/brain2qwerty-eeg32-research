import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from checks import budget
from lab import ROOT, load, write


def smoke(path, candidate=ROOT / "candidate.py", source_kind="operator-unverified"):
    result = {
        "status": "BLOCKED",
        "source_kind": source_kind,
        "scope": "untrained proxy shape/interface smoke; no brain-decoding accuracy",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(info.file_size for info in archive.infolist()) > 100 * 1024 * 1024:
                raise ValueError("fixture exceeds 100 MiB uncompressed limit")
        with np.load(path, allow_pickle=False) as arrays:
            required = {"x", "coordinates", "subjects", "sentences"}
            if set(arrays.files) != required:
                raise ValueError(
                    "NPZ must contain exactly x, coordinates, subjects, sentences; no labels or raw subject metadata"
                )
            x, coordinates, subjects, sentences = [
                arrays[name] for name in ("x", "coordinates", "subjects", "sentences")
            ]
        if x.ndim != 3:
            raise ValueError("x must have shape [N,C,T]")
        n, channels, time = x.shape
        if not (
            0 < n <= 1024
            and 0 < channels <= 512
            and 0 < time <= 4096
            and x.size <= 10_000_000
        ):
            raise ValueError("fixture dimensions exceed supported bounded shape limits")
        if (
            coordinates.shape != (n, channels, 2)
            or subjects.shape != (n,)
            or sentences.shape != (n,)
        ):
            raise ValueError("expected coordinates [N,C,2] and subjects/sentences [N]")
        if x.dtype.kind != "f" or coordinates.dtype.kind != "f":
            raise ValueError("signals and coordinates must be floating-point arrays")
        if subjects.dtype.kind not in "iu" or sentences.dtype.kind not in "iu":
            raise ValueError(
                "subjects and sentences must be anonymous integer identity codes"
            )
        if not np.isfinite(x).all() or not np.isfinite(coordinates).all():
            raise ValueError("signals and coordinates must be finite")
        if subjects.dtype.kind == "u" and subjects.max() > np.iinfo(np.int64).max:
            raise ValueError("subject identities exceed int64 range")
        if sentences.dtype.kind == "u" and sentences.max() > np.iinfo(np.int64).max:
            raise ValueError("sentence identities exceed int64 range")
        torch.manual_seed(3)
        model = load(candidate)().eval()
        budget(model)
        inputs = [
            torch.from_numpy(np.array(value, dtype=dtype, copy=True))
            for value, dtype in [
                (x, np.float32),
                (coordinates, np.float32),
                (subjects, np.int64),
                (sentences, np.int64),
            ]
        ]
        with torch.no_grad():
            logits = model(*inputs)
        if logits.shape != (n, 29) or not torch.isfinite(logits).all():
            raise ValueError("expected finite logits [N,29]")
        result.update(
            status="PASS",
            input_shape=list(x.shape),
            coordinates_shape=list(coordinates.shape),
            logits_shape=list(logits.shape),
            representations=model.representation_shapes,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Local NPZ shape smoke. Never uploads or copies the fixture."
    )
    parser.add_argument("fixture")
    parser.add_argument("--candidate", default=str(ROOT / "candidate.py"))
    parser.add_argument("--output", default="results/fixture_smoke.json")
    parser.add_argument(
        "--source-kind",
        choices=["synthetic", "operator-unverified"],
        default="operator-unverified",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = smoke(args.fixture, args.candidate, args.source_kind)
    write(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
