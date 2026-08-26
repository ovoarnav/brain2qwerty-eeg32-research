"""Run the unchanged Brain2Qwerty V1 graph on a candidate local backend.

This is deliberately a synthetic gate: it proves model/operator compatibility
without mixing a hardware failure with dataset parsing or preprocessing.
"""

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import torch

from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
from neuraltrain.models import BaseModelConfig


def _device(backend: str) -> torch.device:
    if backend == "directml":
        import torch_directml

        return torch_directml.device()
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def _stage(name: str) -> None:
    print(f"[backend_gate] {name}", flush=True)


def run(backend: str, channels: int, backward: bool, optimizer_step: bool) -> dict:
    started = time.perf_counter()
    device = _device(backend)
    torch.manual_seed(33)

    encoder_cfg = BaseModelConfig(**ENCODER)
    transformer_cfg = BaseModelConfig(**TRANSFORMER)
    encoder = encoder_cfg.build(n_in_channels=channels, n_outputs=ENCODER["hidden"])
    transformer = transformer_cfg.build(dim=ENCODER["hidden"])
    head = torch.nn.Linear(ENCODER["hidden"], 29)
    module = torch.nn.ModuleDict(
        {"encoder": encoder, "transformer": transformer, "head": head}
    )

    # Match Experiment.run(): materialize lazy channel-merger and attention
    # parameters on CPU before Lightning transfers the initialized module.
    _stage("materialize_cpu")
    module.eval()
    with torch.no_grad():
        module["encoder"](
            torch.randn(1, channels, 25),
            torch.zeros(1, 1, dtype=torch.long),
            torch.rand(1, channels, 2),
        )
    # This is the same cleanup performed by materialize_lazy_params() in the
    # production entrypoint for branches that are intentionally unused.
    for submodule in module.modules():
        for name, parameter in list(getattr(submodule, "_parameters", {}).items()):
            if isinstance(parameter, torch.nn.UninitializedParameter):
                submodule._parameters[name] = torch.nn.Parameter(torch.empty(1))
    module.train()
    _stage(f"transfer_{backend}")
    module.to(device)

    x = torch.randn(1, channels, 25, device=device)
    subject = torch.zeros(1, 1, dtype=torch.long, device=device)
    positions = torch.rand(1, channels, 2, device=device)
    target = torch.tensor([0], dtype=torch.long, device=device)

    _stage("encoder_forward")
    encoded = module["encoder"](x, subject, positions)
    if isinstance(encoded, dict):
        encoded = encoded.get("c_out", next(iter(encoded.values())))
    encoded = encoded.reshape(1, -1)
    _stage("transformer_forward")
    transformed = module["transformer"](
        encoded.unsqueeze(0), mask=torch.ones(1, 1, dtype=torch.bool, device=device)
    )
    logits = module["head"](transformed.reshape(1, ENCODER["hidden"]))
    loss = torch.nn.functional.cross_entropy(logits, target)
    if backward:
        _stage("backward")
        loss.backward()

    optimizer_seconds = None
    if optimizer_step:
        if not backward:
            raise ValueError("optimizer step requires backward")
        _stage("optimizer_step")
        step_started = time.perf_counter()
        optimizer = torch.optim.AdamW(module.parameters(), lr=5e-5, weight_decay=1e-4)
        optimizer.step()
        optimizer_seconds = time.perf_counter() - step_started

    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    return {
        "status": "pass",
        "backend": backend,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "channels": channels,
        "time_points": 25,
        "architecture": {"encoder": ENCODER, "transformer": TRANSFORMER},
        "parameters": sum(p.numel() for p in module.parameters()),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "backward": backward,
        "finite_gradients": (
            bool(gradients)
            and all(bool(torch.isfinite(g).all().cpu()) for g in gradients)
            if backward
            else None
        ),
        "optimizer_step": optimizer_step,
        "optimizer_seconds": optimizer_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda", "directml"), required=True)
    parser.add_argument("--channels", type=int, choices=(32, 61), default=32)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = run(
            args.backend,
            args.channels,
            backward=not args.forward_only,
            optimizer_step=args.optimizer_step,
        )
    except Exception as error:
        report = {
            "status": "fail",
            "backend": args.backend,
            "channels": args.channels,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
