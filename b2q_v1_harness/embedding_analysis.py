import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from embedding_study import datasets, initialized, tensors_hash
from lab import load, write


def spectrum(value):
    values = value.detach().double()
    centered = values - values.mean(0)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    if variance.sum() == 0:
        return {
            "width": values.shape[1],
            "numerical_rank": 0,
            "participation_ratio": 0.0,
            "dimensions_for_95_percent_variance": 0,
        }
    cumulative = variance.cumsum(0) / variance.sum()
    return {
        "width": values.shape[1],
        "numerical_rank": int((singular > singular[0] * 1e-5).sum()),
        "participation_ratio": float(variance.sum().square() / variance.square().sum()),
        "dimensions_for_95_percent_variance": int((cumulative < 0.95).sum()) + 1,
        "mean_vector_norm": float(values.norm(dim=1).mean()),
    }


def geometry(model, data):
    captured = {}
    handles = []
    if hasattr(model, "adapter"):
        handles.append(
            model.adapter.register_forward_pre_hook(
                lambda module, inputs: captured.update(local=inputs[0].detach())
            )
        )
        handles.append(
            model.adapter.register_forward_hook(
                lambda module, inputs, output: captured.update(projected=output.detach())
            )
        )
    else:
        handles.append(
            model.sensor.register_forward_hook(
                lambda module, inputs, output: captured.update(sensor=output.detach())
            )
        )
    try:
        model.eval()
        with torch.no_grad():
            model(*data[:4])
        if "sensor" in captured:
            weights = torch.softmax(data[1][..., 0] + data[1][..., 1], -1)
            captured["local"] = (captured.pop("sensor") * weights.unsqueeze(-1)).sum(1)
            captured["projected"] = captured["local"]
        return {name: spectrum(value) for name, value in captured.items()}
    finally:
        for handle in handles:
            handle.remove()


def analyze_geometry(folder):
    folder = Path(folder)
    comparison = json.loads((folder / "comparison.json").read_text())
    manifest = comparison["manifest"]
    data = datasets(manifest["protocol"])
    assert {name: tensors_hash(split) for name, split in data.items()} == manifest[
        "split_hashes"
    ]
    result = {
        "scope": "post-selection, label-free embedding geometry on synthetic test inputs",
        "relative_singular_value_rank_tolerance": 1e-5,
        "variants": {},
    }
    for name in comparison["summary"]:
        path = folder / name / "candidate.py"
        model_class = load(path)
        rows = []
        for seed in manifest["seeds"]:
            model, _ = initialized(model_class, seed)
            initial = geometry(model, data["test"])
            model.load_state_dict(
                torch.load(folder / name / f"seed-{seed}.pt", weights_only=True)
            )
            rows.append(
                {
                    "seed": seed,
                    "initial": initial,
                    "trained": geometry(model, data["test"]),
                }
            )
        result["variants"][name] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "per_seed": rows,
        }
    write(folder / "embedding_geometry.json", result)
    return result


def plots(balanced, legacy, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    reports = {
        "Balanced independent splits": json.loads(
            (Path(balanced) / "comparison.json").read_text()
        ),
        "Original noise-replicate fixture": json.loads(
            (Path(legacy) / "comparison.json").read_text()
        ),
    }
    names = list(next(iter(reports.values()))["summary"])
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    for axis, (title, report) in zip(axes, reports.items()):
        winner = report["validation_selected_variant"]
        colors = [
            "#475569"
            if name == "baseline32"
            else "#0f766e"
            if name == winner
            else "#0284c7"
            for name in names
        ]
        means = [report["summary"][name]["test_error"]["mean"] * 100 for name in names]
        sd = [report["summary"][name]["test_error"]["sd"] * 100 for name in names]
        positions = np.arange(len(names))
        axis.barh(positions, means, xerr=sd, color=colors, alpha=0.8, capsize=3)
        for position, name in zip(positions, names):
            points = np.array(report["summary"][name]["test_error"]["per_seed"]) * 100
            axis.scatter(
                points, np.full(len(points), position), color="black", s=13, zorder=4
            )
        axis.set_title(f"{title}\nSelected on validation: {winner}")
        axis.set_yticks(positions, names)
        axis.set_xlabel("Final test character mismatch (%) — lower is better")
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Changing the local embedding changes the result\nSynthetic proxy: five seeds; bars show mean ± sample SD",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(destination / "embedding_errors.png", dpi=180)
    fig.savefig(destination / "embedding_errors.pdf")
    plt.close(fig)

    report = reports["Balanced independent splits"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    selected = [
        "baseline32",
        "linear32_control",
        "linear48",
        "linear64",
        "gelu48",
        "mlp48",
    ]
    for name in selected:
        rows = report["runs"][name]
        steps = [point["step"] for point in rows[0]["curve"]]
        for axis, split in zip(axes, ["train", "validation"]):
            curves = np.array(
                [
                    [point[split]["cross_entropy"] for point in row["curve"]]
                    for row in rows
                ]
            )
            mean = curves.mean(0)
            deviation = curves.std(0, ddof=1)
            axis.plot(steps, mean, label=name)
            axis.fill_between(steps, mean - deviation, mean + deviation, alpha=0.1)
            axis.set_title(f"{split.capitalize()} cross-entropy")
            axis.set_xlabel("Optimizer updates")
            axis.set_ylabel("Cross-entropy — lower is better")
            axis.grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    fig.suptitle(
        "Same budget and matched context/head initialization\nBalanced synthetic split; lines mean, shading sample SD"
    )
    fig.tight_layout()
    fig.savefig(destination / "embedding_learning_curves.png", dpi=180)
    fig.savefig(destination / "embedding_learning_curves.pdf")
    plt.close(fig)

    geometry_report = json.loads((Path(balanced) / "embedding_geometry.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    positions = np.arange(len(names))
    for axis, stage in zip(axes, ["local", "projected"]):
        widths = [
            geometry_report["variants"][name]["per_seed"][0]["trained"][stage]["width"]
            for name in names
        ]
        ranks = [
            np.mean(
                [
                    row["trained"][stage]["numerical_rank"]
                    for row in geometry_report["variants"][name]["per_seed"]
                ]
            )
            for name in names
        ]
        axis.barh(
            positions - 0.17, widths, height=0.3, color="#cbd5e1", label="Vector width"
        )
        axis.barh(
            positions + 0.17,
            ranks,
            height=0.3,
            color="#0f766e",
            label="Mean numerical rank",
        )
        axis.set_yticks(positions, names)
        axis.set_title(f"Trained {stage} embedding")
        axis.set_xlabel("Dimensions")
        axis.legend(fontsize=9)
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Wider vectors do not automatically provide a richer basis\nCentered synthetic test matrix; rank tolerance = 1e-5 × largest singular value"
    )
    fig.tight_layout()
    fig.savefig(destination / "embedding_geometry.png", dpi=180)
    fig.savefig(destination / "embedding_geometry.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze completed synthetic embedding comparisons."
    )
    parser.add_argument("--balanced", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    analyze_geometry(args.balanced)
    analyze_geometry(args.legacy)
    plots(args.balanced, args.legacy, args.output)
    print(f"Geometry and plots saved to {args.output}")


if __name__ == "__main__":
    main()
