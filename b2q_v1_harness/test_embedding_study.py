import tempfile
import unittest
from pathlib import Path

import torch
from checks import fixture
from embedding_model import Model
from embedding_study import (
    VARIANTS,
    candidate_source,
    datasets,
    initialized,
    run,
    tensors_hash,
)
from lab import ROOT, load


class EmbeddingComparison(unittest.TestCase):
    def test_identity_embedding_matches_reference(self):
        model, _ = initialized(Model, 7)
        reference, _ = initialized(load(ROOT / "candidate.py"), 7)
        torch.testing.assert_close(model.eval()(*fixture()), reference.eval()(*fixture()))

    def test_balanced_splits_are_separate_and_cover_all_classes(self):
        splits = datasets("balanced")
        self.assertEqual(len({tensors_hash(data) for data in splits.values()}), 3)
        for name, count in [("train", 8), ("validation", 4), ("test", 8)]:
            self.assertTrue(
                torch.equal(torch.bincount(splits[name][4]), torch.full((29,), count))
            )
            self.assertEqual(splits[name][0].shape[1:], (32, 25))

    def test_context_and_head_initialization_is_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            for name in VARIANTS:
                path = Path(tmp) / f"{name}.py"
                path.write_text(candidate_source(name))
                model, common = initialized(load(path), 17)
                self.assertEqual(model.head.in_features, 32)
                hashes.append(common)
            self.assertTrue(all(common == hashes[0] for common in hashes))

    def test_comparison_selects_on_validation_and_records_test_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run(
                Path(tmp) / "study", "balanced", [7], 2, ["baseline32", "linear48"]
            )
            expected = min(
                report["summary"],
                key=lambda name: report["summary"][name]["validation_error"]["mean"],
            )
            self.assertEqual(report["validation_selected_variant"], expected)
            self.assertEqual(
                report["summary"]["baseline32"]["paired_test_delta"]["mean"], 0
            )
            for row in report["runs"]["linear48"]:
                self.assertEqual(row["curve"][-1]["step"], 2)
                self.assertEqual(row["test"]["examples"], 232)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
