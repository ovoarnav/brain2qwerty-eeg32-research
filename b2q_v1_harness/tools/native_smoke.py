"""Smoke-test real V1 classes with synthetic data in the original environment."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', default='results/native_smoke.json')
    args = parser.parse_args()
    result = {'status': 'BLOCKED', 'scope': 'real V1 classes, reduced configuration, synthetic CPU; not reproduction'}
    try:
        sys.path.insert(0, str(Path(args.repo).resolve()))
        import torch
        from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
        from brain2qwerty_v1.metrics import CER
        from brain2qwerty_v1.pl_module import BrainModule
        from neuraltrain.models.base import BaseModelConfig
        from neuraltrain.optimizers import LightningOptimizer
        from brain2qwerty_v1.utils import SentenceGroupedDistributedSampler
        torch.set_num_threads(2)
        torch.manual_seed(0)
        enc = copy.deepcopy(ENCODER)
        enc.update(hidden=32, depth=2, initial_linear=16)
        enc['merger_config'].update(n_virtual_channels=16)
        enc['merger_config']['fourier_emb_config'].update(total_dim=32)
        tr = copy.deepcopy(TRANSFORMER)
        tr.update(depth=1, heads=1)
        module = BrainModule(model=BaseModelConfig(**enc).build(n_in_channels=32, n_outputs=32), transformer=BaseModelConfig(**tr).build(dim=32), loss=torch.nn.CrossEntropyLoss(), metrics={'CER': CER()}, optimizer=LightningOptimizer(optimizer={'name': 'AdamW', 'lr': 5e-5}, scheduler={'name': 'OneCycleLR', 'kwargs': {'max_lr': 5e-5}}))
        batch = SimpleNamespace(data={'neuro': torch.randn(4, 32, 25), 'subject_id': torch.zeros(4, 1, dtype=torch.long), 'channel_positions': torch.rand(4, 32, 2), 'feature': torch.tensor([[0], [1], [2], [3]])}, segments=[SimpleNamespace(trigger=SimpleNamespace(extra={'sentence_UID': uid})) for uid in ['A', 'B', 'A', 'B']])
        embeddings = module.forward(batch)
        logits = module._transformer_forward(batch, embeddings)
        assert embeddings.shape == (4, 32) and logits.shape == (4, 29)
        loss = module.loss(logits, batch.data['feature'].squeeze(1))
        loss.backward()
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        result.update(status='PASS', embeddings_shape=list(embeddings.shape), logits_shape=list(logits.shape), loss=float(loss.detach()), finite_gradients=True, torch=torch.__version__)
        sampler_order = list(SentenceGroupedDistributedSampler(batch.segments))
        assert sampler_order == [0, 2, 1, 3]
        result['sampler_order'] = sampler_order
        # Stronger fidelity probe: does the native transformer restore row order?
        module.eval()
        with torch.no_grad():
            actual = module._transformer_forward(batch, embeddings.detach())
            expected = torch.empty_like(actual)
            for indices in ([0, 2], [1, 3]):
                part = SimpleNamespace(segments=[batch.segments[i] for i in indices])
                expected[indices] = module._transformer_forward(part, embeddings.detach()[indices])
        result['interleaved_row_order_preserved'] = bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        result['grouped_order_matches'] = bool(torch.allclose(actual, expected[sampler_order], atol=1e-5, rtol=1e-5))
        if not result['interleaved_row_order_preserved']:
            result.update(status='BLOCKED', error='Native V1 transformer emits sentence-grouped rows without restoring original interleaved row order. Forward/backward succeeded; this is a source fidelity limitation, not a dependency failure.')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
