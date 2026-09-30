import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from checks import budget, contracts, fixture
from lab import load, write


def inspect(candidate):
    candidate = Path(candidate).resolve()
    result = {
        "status": "FAIL",
        "scope": "synthetic tensor/gradient inspection; not training or EEG accuracy",
        "candidate_file": candidate.name,
    }
    handles = []
    try:
        result["source_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        model_class = load(candidate)
        torch.manual_seed(3)
        model = model_class().eval()
        budget(model)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        trace = {}

        def shapes(value):
            if isinstance(value, torch.Tensor):
                return list(value.shape)
            if isinstance(value, (tuple, list)):
                return [shapes(v) for v in value]
            if isinstance(value, dict):
                return {k: shapes(v) for k, v in value.items()}
            return None

        for name, module in model.named_children():

            def hook(module, inputs, output, name=name):
                trace[name] = {"input": shapes(inputs), "output": shapes(output)}

            handles.append(module.register_forward_hook(hook))
        x, coordinates, subjects, sentences, labels = fixture()
        x.requires_grad_()
        coordinates.requires_grad_()
        logits = model(x, coordinates, subjects, sentences, labels)
        for handle in handles:
            handle.remove()
        handles.clear()
        result["module_trace"] = trace
        result["representations"] = model.representation_shapes
        assert logits.shape == (7, 29) and torch.isfinite(logits).all()
        torch.nn.functional.cross_entropy(logits, labels).backward()

        def gradient_info(gradient):
            return {
                "present": gradient is not None,
                "finite": bool(torch.isfinite(gradient).all())
                if gradient is not None
                else False,
                "nonzero": bool(gradient.abs().sum() > 0)
                if gradient is not None
                else False,
            }

        result["input_gradients"] = {
            "signals": gradient_info(x.grad),
            "coordinates": gradient_info(coordinates.grad),
        }
        result["parameter_gradients"] = {
            name: gradient_info(p.grad) for name, p in model.named_parameters()
        }
        result["contracts"] = contracts(model_class)
        connected = all(
            g["present"] and g["finite"] and g["nonzero"]
            for g in result["input_gradients"].values()
        )
        result["status"] = (
            "PASS"
            if connected and all(c["status"] == "PASS" for c in result["contracts"])
            else "FAIL"
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        for handle in handles:
            handle.remove()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Operator-side inspector for trusted candidate code; no training."
    )
    parser.add_argument(
        "candidate", help="Path to candidate.py, not a workspace directory"
    )
    parser.add_argument("--output", default="results/inspection.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = inspect(args.candidate)
    write(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "scope", "source_sha256", "representations", "error")
                if k in result
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
