"""Operator-owned contracts: keep outside agent-writable candidate workspace."""
import io
import torch

FAULTS = ['wrong_output_shape', 'detached_upstream', 'padding_mask_removed', 'cross_sentence_mix', 'row_order_lost', 'label_leak', 'width_mismatch']


def fixture(channels=32, time=25):
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(7, channels, time, generator=generator)
    coords = torch.rand(7, channels, 2, generator=generator)
    # Interleaving, ragged groups, and reused sentence IDs across subjects.
    subjects = torch.tensor([0, 0, 1, 0, 1, 0, 1])
    sentences = torch.tensor([0, 1, 0, 0, 0, 1, 2])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    return x, coords, subjects, sentences, labels


def contracts(model_class, fault=None):
    torch.manual_seed(3)
    model = model_class(**({'fault': fault} if fault else {})).eval()
    args = fixture()
    results = []

    def check(name, fn):
        try:
            fn()
            results.append({'name': name, 'status': 'PASS'})
        except Exception as error:
            results.append({'name': name, 'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'})

    def output(inputs=args):
        return model(*inputs)

    def shape():
        y = output()
        assert y.shape == (7, 29) and torch.isfinite(y).all()

    check('finite_one_logit_row_per_keystroke', shape)

    def variants():
        for c, t in [(32, 25), (61, 40), (4, 7)]:
            y = output(fixture(c, t))
            assert y.shape == (7, 29) and torch.isfinite(y).all()
    check('channel_and_window_shape_variants', variants)
    check('single_keystroke', lambda: torch.testing.assert_close(output(tuple(v[:1] for v in args)).shape, torch.Size([1, 29])))

    def isolated():
        full = output()
        keys = list(zip(args[2].tolist(), args[3].tolist()))
        for key in set(keys):
            indices = [i for i, k in enumerate(keys) if k == key]
            torch.testing.assert_close(full[indices], output(tuple(v[indices] for v in args)), atol=1e-5, rtol=1e-5)
    check('sentence_isolation', isolated)
    check('ragged_padding_invariance', isolated)
    check('subject_sentence_composite_identity', isolated)

    def order():
        indices = torch.tensor([0, 3, 1, 5, 2, 4, 6])
        torch.testing.assert_close(output()[indices], output(tuple(v[indices] for v in args)), atol=1e-5, rtol=1e-5)
    check('interleaved_rows_restore_original_order', order)

    def joint_permutation():
        indices = torch.arange(args[0].shape[1] - 1, -1, -1)
        torch.testing.assert_close(output(), output((args[0][:, indices], args[1][:, indices], *args[2:])), atol=1e-5, rtol=1e-5)
    check('joint_sensor_coordinate_permutation', joint_permutation)

    def coordinates():
        altered = args[1].clone()
        altered[:, 0, 0] += 2
        assert not torch.allclose(output(), output((args[0], altered, *args[2:])))
    check('coordinate_path_has_effect', coordinates)

    def canary():
        torch.testing.assert_close(output(), output((*args[:4], (args[4] + 9) % 29)))
    check('label_canary_independence', canary)

    def gradient():
        model.zero_grad(set_to_none=True)
        x = args[0].clone().requires_grad_()
        y = model(x, *args[1:])
        torch.nn.functional.cross_entropy(y, args[4]).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert model.sensor.weight.grad is not None and model.sensor.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    check('upstream_gradient_connectivity', gradient)

    def checkpoint():
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        clone = model_class(**({'fault': fault} if fault else {})).eval()
        clone.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(output(), clone(*args))
    check('checkpoint_roundtrip', checkpoint)

    def representations():
        observed = {}
        handles = [
            model.sensor.register_forward_hook(lambda m, inputs, value: observed.update(local=value.shape[-1])),
            model.context.register_forward_hook(lambda m, inputs, value: observed.update(contextual=value.shape[-1])),
            model.head.register_forward_pre_hook(lambda m, inputs: observed.update(fused=inputs[0].shape[-1])),
        ]
        try:
            output()
        finally:
            for handle in handles:
                handle.remove()
        shapes = model.representation_shapes
        assert set(shapes) == {'local', 'contextual', 'fused'}
        assert all(s[0] == 7 and len(s) == 2 and s[1] > 0 for s in shapes.values())
        assert all(shapes[name][1] == observed[name] for name in shapes)
    check('declared_representation_shapes', representations)
    check('parameter_budget', lambda: budget(model))
    check('repeatable_eval', lambda: torch.testing.assert_close(output(), output()))

    def sensors():
        changed = args[0].clone()
        changed[0] += 3
        assert not torch.allclose(output(), output((changed, *args[1:])))
    check('sensor_path_has_effect', sensors)
    return results


def budget(model):
    count = sum(p.numel() for p in model.parameters())
    assert 0 < count <= 500_000, f'parameter budget exceeded: {count}'
