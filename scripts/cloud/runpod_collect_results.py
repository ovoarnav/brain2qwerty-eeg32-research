"""Create honest benchmark/result summaries from Lightning CSV logs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def wall_seconds(log: Path) -> float | None:
    text = log.read_text(errors="replace")
    portable = re.findall(r"B2Q_ELAPSED_SECONDS:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if portable:
        return float(portable[-1])
    match = re.search(
        r"Elapsed \(wall clock\) time .*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)


def steady_state_train_batch_seconds(log: Path) -> float | None:
    """Read Lightning's final train-loop throughput before validation begins."""
    text = log.read_text(errors="replace")
    train_text = text.split("Validation:", maxsplit=1)[0]
    rates = re.findall(r"\b200/200\s+\[[^\]]*,\s*([0-9]+(?:\.[0-9]+)?)it/s", train_text)
    if not rates:
        return None
    return 1 / float(rates[-1])


def write_benchmark(results: Path, hourly: float) -> None:
    rows = ["# CUDA benchmark and cost projection", "", f"Hourly Pod price: `${hourly:.4f}` USD.", ""]
    rows += ["| Condition | Full benchmark wall seconds | Steady-state train seconds/batch | 300-epoch train-only cost |", "|---|---:|---:|---:|"]
    for condition in ("full", "eeg32"):
        log = results / f"benchmark_{condition}.log"
        wall = wall_seconds(log)
        per_batch = steady_state_train_batch_seconds(log)
        if wall is None or per_batch is None:
            rows.append(f"| {condition} | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |")
            continue
        projected = per_batch * 1832 * 300 / 3600 * hourly
        rows.append(f"| {condition} | {wall:.1f} | {per_batch:.4f} | ${projected:.2f} |")
    rows.append("")
    rows.append("Cost uses the final Lightning train-loop rate before validation. Full wall time is retained to show startup, validation, test, and checkpoint overhead.")
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
