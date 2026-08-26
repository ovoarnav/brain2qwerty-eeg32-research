"""Create honest benchmark/result summaries from Lightning CSV logs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def train_seconds(log: Path) -> float | None:
    text = log.read_text(errors="replace")
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)


def write_benchmark(results: Path, hourly: float) -> None:
    rows = ["# CUDA benchmark and cost projection", "", f"Hourly Pod price: `${hourly:.4f}` USD.", ""]
    rows += ["| Condition | Wall seconds | Seconds/train batch (200-batch bound) | 300-epoch lower-bound cost |", "|---|---:|---:|---:|"]
    for condition in ("full", "eeg32"):
        seconds = train_seconds(results / f"benchmark_{condition}.log")
        if seconds is None:
            rows.append(f"| {condition} | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |")
            continue
        per_batch = seconds / 200
        projected = per_batch * 1832 * 300 / 3600 * hourly
        rows.append(f"| {condition} | {seconds:.1f} | {per_batch:.4f} | ${projected:.2f} |")
    rows.append("")
    rows.append("Projection excludes validation, test, cache misses, and checkpoints; use it as a lower bound.")
    (results / "BENCHMARK.md").write_text("\n".join(rows) + "\n")
    (results / "COST_PROJECTION.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark",))
    parser.add_argument("--hourly-usd", required=True, type=float)
    parser.add_argument("--results-dir", type=Path, default=Path("/workspace/cloud_results"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark(args.results_dir, args.hourly_usd)


if __name__ == "__main__":
    main()
