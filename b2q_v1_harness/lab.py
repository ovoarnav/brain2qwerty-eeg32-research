#!/usr/bin/env python3
"""Bounded developer harness. Synthetic results are not EEG accuracy."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from checks import FAULTS, budget, contracts

ROOT = Path(__file__).resolve().parent


def bounded_run(command, timeout, stdout, stderr):
    """Kill the invocation's POSIX process group on timeout, not unrelated tasks."""
    process = subprocess.Popen(
        command, stdout=stdout, stderr=stderr, start_new_session=os.name != "nt"
    )
    try:
        process.wait(timeout=timeout)
    except BaseException:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
        raise
    return process


def reported_tokens(path):
    total = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage", {})
            total += int(usage.get("input_tokens", 0)) + int(
                usage.get("output_tokens", 0)
            )
    return total


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load(path):
    spec = importlib.util.spec_from_file_location("harness_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def training(model_class):
    errors = []
    for seed in (7, 17):
        torch.manual_seed(seed)
        model = model_class()
        budget(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        generator = torch.Generator().manual_seed(140)
        # Fixed 61-character developer fixture, with separate held-out signals.
        labels = torch.arange(61) % 29
        subjects = torch.arange(61) // 31
        sentences = torch.arange(61) // 7
        coords = torch.rand(61, 8, 2, generator=generator)
        signal = labels.float()[:, None, None] / 14 - 1
        x = signal + 0.2 * torch.randn(61, 8, 12, generator=generator)
        heldout = signal + 0.2 * torch.randn(61, 8, 12, generator=generator)
        for _ in range(160):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, coords, subjects, sentences)
            assert logits.shape == (61, 29)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            assert torch.isfinite(loss), "nonfinite training loss"
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(heldout, coords, subjects, sentences).argmax(-1)
        errors.append(float((predictions != labels).float().mean()))
    return {
        "seeds": [7, 17],
        "updates_per_seed": 160,
        "raw_character_error": errors,
        "mean_raw_character_error": sum(errors) / len(errors),
        "scope": "synthetic developer smoke; not paper CER or EEG results",
    }


def evaluate(workspace, output, run_id):
    report = {"run_id": run_id, "status": "FAIL"}
    try:
        model_class = load(Path(workspace) / "candidate.py")
        report["checks"] = contracts(model_class)
        if all(c["status"] == "PASS" for c in report["checks"]):
            report["training"] = training(model_class)
            report["status"] = "PASS"
        model = model_class().eval()
        from checks import fixture

        model(*fixture())
        report["representations"] = model.representation_shapes
        write(Path(workspace) / "math_diff.json", report["representations"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write(output, report)
    return 0 if report["status"] == "PASS" else 1


def prepare(workspace, context="structured"):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "candidate.py", workspace / "candidate.py")
    (workspace / "AGENTS.md").write_text(
        "Edit only candidate.py. Never alter the evaluator, labels, data, budget or acceptance rule. Do not read credentials or use network. Keep the Model interface and coordinate, subject and sentence semantics. This is a synthetic proxy, not EEG science.\n",
        encoding="utf-8",
    )
    (workspace / "TASK.md").write_text(
        "Make one useful representation experiment: migrate local width from 32 to 48 while keeping contextual and fused widths 32 using a trainable adapter. Preserve upstream gradients, sentence isolation, row ordering, ragged padding, checkpoint loading, and parameter count <=500000. Model() must work with no arguments. Keep labels out of the forward computation. Declare actual local/contextual/fused shapes.\n",
        encoding="utf-8",
    )
    if context == "structured":
        (workspace / "MATH_CONTEXT.md").write_text(
            "Inputs: x [N,C,T], coordinates [N,C,2], subjects [N], sentences [N]. Output: logits [N,29]. Pool sensors with joint signal/coordinate permutation invariance. Group by composite (subject,sentence), preserve within-group order, pad with a bool mask, restore original interleaved row order. Local width can differ from context width only through a trainable adapter. Labels are canaries, never model input.\n",
            encoding="utf-8",
        )
        from tools.inspect_candidate import inspect

        write(workspace / "INSPECTION.json", inspect(workspace / "candidate.py"))
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=Harness",
            "-c",
            "user.email=harness@localhost",
            "commit",
            "-qm",
            "Baseline candidate",
        ],
        check=True,
    )
    return workspace


def checked_result(output, run_id, exit_code):
    if exit_code != 0 or not Path(output).is_file():
        return {
            "status": "FAIL",
            "error": f"evaluator exit {exit_code} or missing output",
        }
    try:
        data = json.loads(Path(output).read_text())
        if data.get("run_id") != run_id or data.get("status") != "PASS":
            return {"status": "FAIL", "error": "stale or unsuccessful evaluation"}
        return data
    except (ValueError, OSError) as error:
        return {"status": "FAIL", "error": str(error)}


def report(path, data):
    text = (
        "# Developer harness report\n\nSynthetic CPU developer checks only; no EEG results or deployment.\n\n```json\n"
        + json.dumps(data, indent=2)
        + "\n```\n"
    )
    Path(path).write_text(text, encoding="utf-8")


def selftest():
    model_class = load(ROOT / "candidate.py")
    baseline = contracts(model_class)
    faults = {fault: contracts(model_class, fault) for fault in FAULTS}
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation.json"
        write(path, {"status": "PASS", "run_id": "old"})
        regressions = {
            "stale_output_rejected": checked_result(path, "new", 0)["status"] == "FAIL",
            "unsuccessful_exit_rejected": checked_result(path, "old", 1)["status"]
            == "FAIL",
        }

        class Broken(model_class):
            def forward(self, *args, **kwargs):
                raise RuntimeError("deliberate training exception")

        try:
            training(Broken)
            regressions["training_exception_rejected"] = False
        except RuntimeError:
            regressions["training_exception_rejected"] = True
    passed = (
        all(c["status"] == "PASS" for c in baseline)
        and all(any(c["status"] == "FAIL" for c in checks) for checks in faults.values())
        and all(regressions.values())
    )
    data = {
        "status": "PASS" if passed else "FAIL",
        "baseline": baseline,
        "fault_injections": faults,
        "controller_regressions": regressions,
    }
    write(ROOT / "results/selftest.json", data)
    report(ROOT / "results/DEV_DAY_REPORT.md", data)
    print(
        json.dumps(
            {
                "status": data["status"],
                "baseline_passed": sum(c["status"] == "PASS" for c in baseline),
                "faults_caught": sum(
                    any(c["status"] == "FAIL" for c in checks)
                    for checks in faults.values()
                ),
                "controller_regressions": regressions,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


def attach(repo):
    repo = Path(repo).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    files = [
        "brain2qwerty_v1/pl_module.py",
        "brain2qwerty_v1/config/model_config.py",
        "brain2qwerty_v1/eeg.py",
        "brain2qwerty_v1/utils.py",
    ]
    manifest = {
        "head": head,
        "files": {},
        "scope": "read-only allowlist source snapshot; native smoke separately executes real classes",
    }
    for relative in files:
        source = repo / relative
        if source.is_file():
            data = source.read_bytes()
            dest = ROOT / "results/repo_audit/source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            manifest["files"][relative] = hashlib.sha256(data).hexdigest()
    write(ROOT / "results/repo_audit/manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def agent(args):
    run = (
        ROOT
        / "results"
        / ("codex_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    )
    run.mkdir(parents=True)
    started = time.monotonic()
    baseline = training(load(ROOT / "candidate.py"))
    summary = {
        "baseline": baseline,
        "attempts": [],
        "promoted": False,
        "reported_tokens": 0,
        "security": "workspace-write CLI requested; evaluator outside candidate workspace. Not a hardened adversarial security boundary.",
    }
    previous = None
    try:
        for round_id in range(1, args.rounds + 1):
            workspace = prepare(run / f"round_{round_id}" / "workspace", args.context)
            if previous:
                shutil.copy2(previous, workspace / "candidate.py")
                if args.context == "structured":
                    from tools.inspect_candidate import inspect

                    write(
                        workspace / "INSPECTION.json", inspect(workspace / "candidate.py")
                    )
            for attempt in range(2):
                remaining = 2400 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("2400-second experiment budget exhausted")
                if summary["reported_tokens"] >= args.token_budget:
                    raise RuntimeError("reported-token budget reached between calls")
                prefix = workspace.parent / f"attempt_{attempt}"
                prompt = "Read AGENTS.md, TASK.md, MATH_CONTEXT.md and INSPECTION.json if present. INSPECTION.json describes the starting candidate only; its source hash must match before relying on it. Implement the experiment in candidate.py only."
                if attempt:
                    prompt += "\nRepair these external failures: " + json.dumps(
                        summary["attempts"][-1]["evaluation"]
                    )
                command = [
                    "codex",
                    "exec",
                    "--sandbox",
                    "workspace-write",
                    "--ephemeral",
                    "--json",
                    "-C",
                    str(workspace),
                ]
                if args.model:
                    command += ["--model", args.model]
                command += [prompt]
                with (
                    prefix.with_suffix(".events.jsonl").open("w") as out,
                    prefix.with_suffix(".stderr.log").open("w") as err,
                ):
                    process = bounded_run(
                        command, stdout=out, stderr=err, timeout=min(600, remaining)
                    )
                summary["reported_tokens"] += reported_tokens(
                    prefix.with_suffix(".events.jsonl")
                )
                patch = subprocess.check_output(
                    ["git", "-C", str(workspace), "diff", "--", "candidate.py"], text=True
                )
                prefix.with_suffix(".patch").write_text(patch, encoding="utf-8")
                evaluation_path = prefix.with_suffix(".evaluation.json")
                run_id = uuid.uuid4().hex
                if process.returncode:
                    evaluation = {
                        "status": "BLOCKED",
                        "error": f"Codex exit {process.returncode}; inspect stderr/events",
                    }
                    write(evaluation_path, evaluation)
                else:
                    with prefix.with_suffix(".evaluator.log").open("w") as log:
                        proc = bounded_run(
                            [
                                sys.executable,
                                str(ROOT / "lab.py"),
                                "_evaluate",
                                str(workspace),
                                "--output",
                                str(evaluation_path),
                                "--run-id",
                                run_id,
                            ],
                            stdout=log,
                            stderr=log,
                            timeout=min(600, max(1, 2400 - (time.monotonic() - started))),
                        )
                    evaluation = checked_result(evaluation_path, run_id, proc.returncode)
                    if evaluation["status"] == "FAIL" and evaluation_path.exists():
                        evaluation["detail"] = json.loads(evaluation_path.read_text())
                entry = {
                    "round": round_id,
                    "attempt": attempt,
                    "codex_exit": process.returncode,
                    "evaluation": evaluation,
                }
                summary["attempts"].append(entry)
                if evaluation["status"] == "BLOCKED":
                    raise RuntimeError(evaluation["error"])
                if evaluation["status"] == "PASS":
                    promoted = (
                        evaluation["training"]["mean_raw_character_error"]
                        < baseline["mean_raw_character_error"]
                    )
                    entry["promoted"] = promoted
                    if promoted:
                        previous = workspace / "candidate.py"
                        baseline = evaluation["training"]
                        summary["promoted"] = True
                    break
    except (subprocess.TimeoutExpired, RuntimeError, TimeoutError, OSError) as error:
        summary["blocker"] = str(error)
    write(run / "summary.json", summary)
    report(run / "DEV_DAY_REPORT.md", summary)
    print(str(run / "DEV_DAY_REPORT.md"))
    return (
        1
        if "blocker" in summary
        or any(
            a["evaluation"]["status"] != "PASS"
            for a in summary["attempts"]
            if a["attempt"] == 1
        )
        else 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("selftest")
    p = sub.add_parser("attach-repo")
    p.add_argument("repo")
    p = sub.add_parser("prepare")
    p.add_argument("workspace")
    for name in ["evaluate", "_evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("workspace")
        p.add_argument("--output")
        p.add_argument("--run-id", default=None)
    p = sub.add_parser("agent")
    p.add_argument("--rounds", type=int, choices=range(1, 4), default=3)
    p.add_argument("--model")
    p.add_argument("--context", choices=["minimal", "structured"], default="structured")
    p.add_argument("--token-budget", type=int, default=100000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "cuda": torch.cuda.is_available(),
                    "codex": shutil.which("codex"),
                    "git": shutil.which("git"),
                    "scope": "synthetic CPU development",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "selftest":
        return selftest()
    if args.command == "attach-repo":
        return attach(args.repo)
    if args.command == "prepare":
        print(prepare(args.workspace))
        return 0
    if args.command in ["evaluate", "_evaluate"]:
        return evaluate(
            args.workspace,
            args.output or str(Path(args.workspace) / "evaluation.json"),
            args.run_id or uuid.uuid4().hex,
        )
    return agent(args)


if __name__ == "__main__":
    raise SystemExit(main())
