"""Full V1 forward/backward and tiny-overfit gate on cached real EEG."""

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import studies  # noqa: F401  (registers Pinet2024Eeg)
from brain2qwerty_v1.config.xp_config import eeg_debug_config
from brain2qwerty_v1.main import Experiment
from brain2qwerty_v1.metrics import CER
from brain2qwerty_v1.pl_module import BrainModule
from brain2qwerty_v1.utils import materialize_lazy_params


def run(condition: str, steps: int, batch_size: int) -> dict:
    started = time.perf_counter()
    torch.manual_seed(33)
    cfg = eeg_debug_config(condition)
    cfg["data"].update(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    experiment = Experiment(**cfg)
    loaders = experiment.data.build()
    loader = loaders["train"]
    batch = next(iter(loader))
    brain, transformer = experiment._build_modules(loader)
    module = BrainModule(
        model=brain,
        transformer=transformer,
        loss=experiment.loss.build(),
        metrics={"CER": CER()},
        optimizer=experiment.optimizer,
    )
    materialize_lazy_params(module, loader)
    optimizer = torch.optim.AdamW(module.parameters(), lr=5e-5, weight_decay=1e-4)
    setup_seconds = time.perf_counter() - started

    labels = batch.data["feature"].squeeze(1)
    losses = []
    step_timings = []
    finite_gradients = True
    module.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Hold stochastic masks fixed so a decrease measures optimization rather
        # than dropout noise; dropout and BatchNorm remain in training mode.
        torch.manual_seed(33)
        forward_started = time.perf_counter()
        embeddings = module.forward(batch)
        logits = module._transformer_forward(batch, embeddings)
        loss = module.loss(logits, labels)
        forward_seconds = time.perf_counter() - forward_started
        losses.append(float(loss.detach()))
        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_started
        check_started = time.perf_counter()
        finite_gradients &= all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        gradient_check_seconds = time.perf_counter() - check_started
        optimizer_started = time.perf_counter()
        optimizer.step()
        optimizer_seconds = time.perf_counter() - optimizer_started
        step_timings.append(
            {
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "gradient_check_seconds": gradient_check_seconds,
                "training_seconds_excluding_gate_check": forward_seconds
                + backward_seconds
                + optimizer_seconds,
            }
        )
        print(
            f"[real_model_gate] condition={condition} step={step + 1}/{steps} "
            f"loss={losses[-1]:.6f}",
            flush=True,
        )

    module.train()
    final_started = time.perf_counter()
    with torch.no_grad():
        torch.manual_seed(33)
        embeddings = module.forward(batch)
        logits = module._transformer_forward(batch, embeddings)
        final_loss = float(module.loss(logits, labels))
        accuracy = float((logits.argmax(dim=1) == labels).float().mean())
    final_forward_seconds = time.perf_counter() - final_started

    overfit_pass = final_loss < losses[0] if steps > 1 else True
    return {
        "status": "pass" if finite_gradients and overfit_pass else "fail",
        "condition": condition,
        "backend": "cpu",
        "torch": torch.__version__,
        "input_shape": list(batch.data["neuro"].shape),
        "position_shape": list(batch.data["channel_positions"].shape),
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "steps": steps,
        "losses_before_updates": losses,
        "step_timings": step_timings,
        "setup_seconds": setup_seconds,
        "final_forward_seconds": final_forward_seconds,
        "final_loss": final_loss,
        "loss_delta": final_loss - losses[0],
        "batch_accuracy": accuracy,
        "finite_gradients": finite_gradients,
        "overfit_pass": overfit_pass,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("full", "eeg32"), required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.condition, args.steps, args.batch_size)
    except Exception as error:
        report = {
            "status": "fail",
            "condition": args.condition,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
