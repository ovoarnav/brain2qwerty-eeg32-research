import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_native_cpu_pass import initialized_native

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2q_v1_harness/tools"))
import native_embedding_probe as probe  # noqa: E402


def test_native_embedding_probe_identity_zero_and_frozen_weights():
    module, batch, inverse = initialized_native()
    before = probe.parameter_hash(module)
    with torch.no_grad():
        embeddings = module.forward(batch)
    results = probe.measure_variants(module, batch, embeddings, inverse)
    assert results[0]["variant"] == "identity"
    assert results[0]["logit_delta_max_abs"] == 0
    assert results[1]["variant"] == "zero"
    assert results[1]["mean_embedding_l2"] == 0
    assert results[1]["logit_delta_rms"] > 0
    assert all(row["finite_logits"] and row["logits_shape"] == [4, 29] for row in results)
    assert probe.parameter_hash(module) == before


def test_chunked_native_eval_encoder_matches_whole_batch():
    module, batch, _ = initialized_native()
    with torch.no_grad():
        expected = module.forward(batch)
    actual = probe.encode_in_chunks(module, batch, chunk_size=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    module.train()
    with pytest.raises(ValueError, match="eval mode"):
        probe.encode_in_chunks(module, batch, chunk_size=1)


def test_trained_mode_cannot_use_surviving_untrained_only_replay():
    args = SimpleNamespace(
        threads=2, seed=33, handoff=Path("missing-replay"), untrained=False
    )
    with pytest.raises(ValueError, match="requires --untrained"):
        probe.run(args)
