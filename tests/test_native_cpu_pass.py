import copy
import json

import numpy as np
import pytest
import torch

from b2q_v1_harness.tools import native_cpu_pass as native
from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
from brain2qwerty_v1.utils import materialize_lazy_params


def tiny_module(device="cpu"):
    encoder = copy.deepcopy(ENCODER)
    encoder.update(hidden=32, depth=2, initial_linear=16)
    encoder["merger_config"]["n_virtual_channels"] = 16
    encoder["merger_config"]["fourier_emb_config"]["total_dim"] = 32
    transformer = copy.deepcopy(TRANSFORMER)
    transformer.update(depth=1, heads=1)
    return native.build_module(device, encoder, transformer)


def initialized_native():
    torch.manual_seed(33)
    batch, inverse, _, _ = native.prepare_batch(native.synthetic_arrays(4))
    module = tiny_module()
    materialize_lazy_params(module, [batch])
    return module.eval(), batch, inverse


def test_native_checkpoint_cpu_roundtrip(tmp_path):
    module, batch, inverse = initialized_native()
    with torch.no_grad():
        expected = module._transformer_forward(batch, module(batch))[inverse]
    path = tmp_path / "test-native.ckpt"
    torch.save({"state_dict": module.state_dict(), "epoch": 3, "global_step": 10}, path)
    clone, info = native.restore_checkpoint(
        path, module_factory=lambda: tiny_module("meta")
    )
    with torch.no_grad():
        actual = clone._transformer_forward(batch, clone(batch))[inverse]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert info["epoch"] == 3 and info["global_step"] == 10
    assert info["strict_state_dict_load"] and not info["optimizer_restored"]
    assert all(p.device.type == "cpu" for p in clone.parameters())


def test_incompatible_native_weights_are_rejected(tmp_path):
    module, _, _ = initialized_native()
    state = dict(module.state_dict())
    state["linear.weight"] = state["linear.weight"][:, :-1]
    path = tmp_path / "wrong-width.ckpt"
    torch.save({"state_dict": state}, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        native.restore_checkpoint(path, module_factory=lambda: tiny_module("meta"))


def test_proxy_checkpoint_cannot_be_reported_as_native(tmp_path):
    path = tmp_path / "proxy.pt"
    torch.save({"sensor.weight": torch.zeros(32, 4)}, path)
    with pytest.raises(ValueError, match="not the native BrainModule"):
        native.restore_checkpoint(path)


def test_native_grouping_restores_original_rows_and_labels():
    arrays = native.synthetic_arrays(4)
    arrays["sentence_uids"] = np.asarray(["A", "B", "A", "B"])
    arrays["feature"] = np.asarray([1, 2, 3, 4])
    batch, inverse, labels, info = native.prepare_batch(arrays)
    assert info["grouped_input_indices"] == [0, 2, 1, 3]
    torch.testing.assert_close(
        batch.data["neuro"][inverse], torch.from_numpy(arrays["neuro"])
    )
    torch.testing.assert_close(batch.data["feature"].squeeze(1)[inverse], labels)
    torch.manual_seed(33)
    module = tiny_module()
    materialize_lazy_params(module, [batch])
    module.eval()
    with torch.no_grad():
        actual = module._transformer_forward(batch, module(batch))[inverse]
        # Independent native sentence evaluations agree at the original indices.
        for indices in ([0, 2], [1, 3]):
            part_arrays = {
                name: (value[indices] if name != "channel_names" else value)
                for name, value in arrays.items()
            }
            part, part_inverse, _, _ = native.prepare_batch(part_arrays)
            expected = module._transformer_forward(part, module(part))[part_inverse]
            torch.testing.assert_close(actual[indices], expected, atol=1e-5, rtol=1e-5)


def test_montage_and_cross_subject_sentence_collisions_are_rejected():
    arrays = native.synthetic_arrays(2)
    arrays["channel_names"] = arrays["channel_names"][::-1]
    with pytest.raises(ValueError, match="names/order"):
        native.prepare_batch(arrays)
    arrays = native.synthetic_arrays(2)
    arrays["subject_id"][1] = 1
    with pytest.raises(ValueError, match="uniquely identify"):
        native.prepare_batch(arrays)


def test_existing_dataset_gate_npz_is_supported(tmp_path):
    arrays = native.synthetic_arrays(2)
    contents = {
        f"test_{name}": arrays[name]
        for name in ("neuro", "channel_positions", "subject_id")
    }
    contents["test_feature"] = np.asarray([[1], [2]])
    contents["test_segment_signatures"] = np.asarray(
        [
            json.dumps({"sentence_uid": str(uid), "subject": "not-exported"})
            for uid in arrays["sentence_uids"]
        ]
    )
    path = tmp_path / "dataset-gate.npz"
    np.savez_compressed(path, **contents)
    fixture = native.read_fixture(path)
    _, _, labels, info = native.prepare_batch(fixture)
    assert labels.tolist() == [1, 2]
    assert not info["montage_names_verified"]
    assert "subject" not in fixture


def test_lightning_configuration_pickle_requires_explicit_trusted_mode(tmp_path):
    from neuraltrain.optimizers import LightningOptimizer

    module, _, _ = initialized_native()
    path = tmp_path / "lightning-native.ckpt"
    payload = {
        "state_dict": module.state_dict(),
        "hyper_parameters": {
            "optimizer": LightningOptimizer(optimizer={"name": "AdamW", "lr": 5e-5})
        },
    }
    torch.save(payload, path)
    with pytest.raises(Exception, match="Weights only load failed"):
        native.restore_checkpoint(path, module_factory=lambda: tiny_module("meta"))
    clone, info = native.restore_checkpoint(
        path, trusted=True, module_factory=lambda: tiny_module("meta")
    )
    assert not info["weights_only_loader"] and not clone.training


def test_missing_checkpoint_never_silently_uses_random_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="not on this machine"):
        native.restore_checkpoint(tmp_path / "missing-best.ckpt")
