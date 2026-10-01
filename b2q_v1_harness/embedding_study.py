import argparse
import csv
import hashlib
import statistics
import sys
import time
from pathlib import Path

import torch
from checks import budget, contracts
from lab import ROOT, load, write

VARIANTS = {
    "baseline32": {
        "width": 32,
        "activation": "linear",
        "adapter": "identity",
        "normalize": False,
    },
    "linear2": {
        "width": 2,
        "activation": "linear",
        "adapter": "linear",
        "normalize": False,
    },
    "linear16": {
        "width": 16,
        "activation": "linear",
        "adapter": "linear",
        "normalize": False,
    },
    "linear32_control": {
        "width": 32,
        "activation": "linear",
        "adapter": "linear",
        "normalize": False,
    },
    "linear48": {
        "width": 48,
        "activation": "linear",
        "adapter": "linear",
        "normalize": False,
    },
    "linear64": {
        "width": 64,
        "activation": "linear",
        "adapter": "linear",
        "normalize": False,
    },
    "linear96": {
        "width": 96,
        "activation": "linear",
        "adapter": "linear",
        "normalize": False,
    },
    "gelu48": {
        "width": 48,
        "activation": "gelu",
        "adapter": "linear",
        "normalize": False,
    },
    "mlp48": {"width": 48, "activation": "linear", "adapter": "mlp", "normalize": False},
    "normalized48": {
        "width": 48,
        "activation": "linear",
        "adapter": "linear",
        "normalize": True,
    },
}


def tensors_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_hash(state):
    return tensors_hash([state[key] for key in sorted(state)])


def datasets(protocol):
    """Fixed synthetic inputs; no fixture selection based on candidate results."""
    if protocol == "legacy":
        generator = torch.Generator().manual_seed(140)
        labels = torch.arange(61) % 29
        subjects = torch.arange(61) // 31
        sentences = torch.arange(61) // 7
        coordinates = torch.rand(61, 8, 2, generator=generator)
        signal = labels.float()[:, None, None] / 14 - 1
        splits = {}
        for name in ("train", "validation", "test"):
            x = signal + 0.2 * torch.randn(61, 8, 12, generator=generator)
            splits[name] = (x, coordinates, subjects, sentences, labels)
        return splits
    if protocol != "balanced":
        raise ValueError(protocol)
    splits = {}
    for name, repetitions, seed in [
        ("train", 8, 140),
        ("validation", 4, 141),
        ("test", 8, 142),
    ]:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(29).repeat(repetitions)
        labels = labels[torch.randperm(len(labels), generator=generator)]
        n = len(labels)
        subjects = torch.arange(n) // 56
        sentences = torch.arange(n) // 7 % 8
        coordinates = torch.rand(n, 32, 2, generator=generator)
        signal = labels.float()[:, None, None] / 14 - 1
        x = signal + 0.2 * torch.randn(n, 32, 25, generator=generator)
        splits[name] = (x, coordinates, subjects, sentences, labels)
    return splits


def candidate_source(name):
    if name == "baseline32":
        return (ROOT / "candidate.py").read_text()
    template = (
        (ROOT / "embedding_model.py")
        .read_text()
        .replace("class Model(nn.Module):", "class _EmbeddingModel(nn.Module):", 1)
    )
    return (
        template
        + "\n\nMODEL_SPEC = "
        + repr(VARIANTS[name])
        + "\n\nclass Model(_EmbeddingModel):\n    def __init__(self):\n        super().__init__(**MODEL_SPEC)\n"
    )


def initialized(model_class, seed):
    # Fixed context and output initialization across embedding variants for each seed.
    torch.manual_seed(seed)
    reference = load(ROOT / "candidate.py")()
    common_context = reference.context.state_dict()
    common_head = reference.head.state_dict()
    torch.manual_seed(seed)
    model = model_class()
    model.context.load_state_dict(common_context)
    model.head.load_state_dict(common_head)
    budget(model)
    common = {"context": state_hash(common_context), "head": state_hash(common_head)}
    assert state_hash(model.context.state_dict()) == common["context"]
    assert state_hash(model.head.state_dict()) == common["head"]
    return model, common


def metric(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(*data[:4])
        if logits.shape != (len(data[4]), 29) or not torch.isfinite(logits).all():
            raise RuntimeError("invalid evaluation logits")
        return {
            "cross_entropy": float(torch.nn.functional.cross_entropy(logits, data[4])),
            "character_mismatch": float((logits.argmax(-1) != data[4]).float().mean()),
            "examples": len(data[4]),
        }


def train(model, train_data, validation_data, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    curve = [
        {
            "step": 0,
            "train": metric(model, train_data),
            "validation": metric(model, validation_data),
        }
    ]
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(*train_data[:4])
        if logits.shape != (len(train_data[4]), 29):
            raise RuntimeError("invalid training logits")
        loss = torch.nn.functional.cross_entropy(logits, train_data[4])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        if not all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        ):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        if step % 20 == 0 or step == steps:
            curve.append(
                {
                    "step": step,
                    "train": metric(model, train_data),
                    "validation": metric(model, validation_data),
                }
            )
    return curve


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "per_seed": values,
    }


def run(output, protocol, seeds, steps, names):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = datasets(protocol)
    manifest = {
        "protocol": protocol,
        "scope": "synthetic amplitude-coded developer experiment, not EEG accuracy",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "seeds": seeds,
        "steps": steps,
        "cpu_threads": 2,
        "optimizer": {"name": "AdamW", "lr": 0.003, "weight_decay": 0.01},
        "context_width": 32,
        "fused_width": 32,
        "split_hashes": {name: tensors_hash(split) for name, split in data.items()},
        "split_sizes": {name: len(split[4]) for name, split in data.items()},
        "initialization": "context and head match the reference baseline for each seed",
        "selection": "lowest mean final validation character mismatch; baseline wins exact ties; test evaluated only after selection",
        "variants": {name: VARIANTS[name] for name in names},
    }
    write(output / "manifest.json", manifest)
    records = {}
    model_classes = {}
    for name in names:
        folder = output / name
        folder.mkdir()
        path = folder / "candidate.py"
        path.write_text(candidate_source(name), encoding="utf-8")
        model_class = load(path)
        model_classes[name] = model_class
        checks = contracts(model_class)
        write(folder / "contracts.json", checks)
        if len(checks) != 16 or any(c["status"] != "PASS" for c in checks):
            raise RuntimeError(
                f"contract failure in {name}; preserved {folder / 'contracts.json'}"
            )
        records[name] = []
        for seed in seeds:
            model, common = initialized(model_class, seed)
            curve = train(model, data["train"], data["validation"], steps)
            row = {
                "seed": seed,
                "parameters": sum(p.numel() for p in model.parameters()),
                "common_initialization": common,
                "curve": curve,
                "train": curve[-1]["train"],
                "validation": curve[-1]["validation"],
            }
            records[name].append(row)
            torch.save(model.state_dict(), folder / f"seed-{seed}.pt")
            write(folder / f"seed-{seed}.json", row)
            print(
                f"{protocol} {name} seed={seed} val={row['validation']['character_mismatch']:.4f}",
                flush=True,
            )
    # Freeze the selection before accessing the final test split through any model.
    winner = min(
        names,
        key=lambda name: statistics.mean(
            r["validation"]["character_mismatch"] for r in records[name]
        ),
    )
    write(
        output / "selection.json",
        {"winner": winner, "basis": "validation only", "test_not_evaluated_yet": True},
    )
    for name in names:
        for row in records[name]:
            model, _ = initialized(model_classes[name], row["seed"])
            model.load_state_dict(
                torch.load(output / name / f"seed-{row['seed']}.pt", weights_only=True)
            )
            row["test"] = metric(model, data["test"])
            write(output / name / f"seed-{row['seed']}.json", row)
    baseline = records["baseline32"]
    summary = {}
    for name in names:
        rows = records[name]
        deltas = [
            r["test"]["character_mismatch"] - b["test"]["character_mismatch"]
            for r, b in zip(rows, baseline)
        ]
        summary[name] = {
            "spec": VARIANTS[name],
            "parameters": rows[0]["parameters"],
            "contracts_passed": 16,
            "validation_error": summarize(
                [r["validation"]["character_mismatch"] for r in rows]
            ),
            "test_error": summarize([r["test"]["character_mismatch"] for r in rows]),
            "test_cross_entropy": summarize([r["test"]["cross_entropy"] for r in rows]),
            "paired_test_delta": summarize(deltas),
            "test_seeds_better_than_baseline": sum(value < 0 for value in deltas),
        }
    report = {
        "manifest": manifest,
        "validation_selected_variant": winner,
        "summary": summary,
        "runs": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write(output / "comparison.json", report)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "variant",
                "local_width",
                "parameters",
                "validation_mismatch_mean",
                "test_mismatch_mean",
                "test_mismatch_sd",
                "paired_delta_mean",
                "paired_delta_sd",
                "test_cross_entropy_mean",
                "test_seeds_better",
            ]
        )
        for name, item in summary.items():
            writer.writerow(
                [
                    name,
                    item["spec"]["width"],
                    item["parameters"],
                    item["validation_error"]["mean"],
                    item["test_error"]["mean"],
                    item["test_error"]["sd"],
                    item["paired_test_delta"]["mean"],
                    item["paired_test_delta"]["sd"],
                    item["test_cross_entropy"]["mean"],
                    item["test_seeds_better_than_baseline"],
                ]
            )
    print(
        f"Finished {protocol}; validation-selected={winner}; {output / 'comparison.json'}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Controlled local embedding comparison; synthetic CPU only."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=["legacy", "balanced"], default="balanced")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    args = parser.parse_args()
    if (
        args.steps <= 0
        or len(set(args.seeds)) != len(args.seeds)
        or args.variants[0] != "baseline32"
        or len(set(args.variants)) != len(args.variants)
    ):
        parser.error(
            "positive steps, unique seeds/variants, and baseline32 first are required"
        )
    run(args.output, args.protocol, args.seeds, args.steps, args.variants)


if __name__ == "__main__":
    main()
