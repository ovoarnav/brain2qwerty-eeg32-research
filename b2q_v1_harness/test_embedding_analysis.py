import unittest

import torch
from checks import fixture
from embedding_analysis import geometry, spectrum
from embedding_model import Model


class EmbeddingGeometry(unittest.TestCase):
    def test_wide_low_rank_matrix_is_not_reported_as_full_rank(self):
        generator = torch.Generator().manual_seed(91)
        factors = torch.randn(30, 2, generator=generator, dtype=torch.float64)
        projection = torch.randn(2, 48, generator=generator, dtype=torch.float64)
        report = spectrum(factors @ projection)
        self.assertEqual(report["width"], 48)
        self.assertEqual(report["numerical_rank"], 2)

    def test_geometry_captures_local_and_projected_embeddings_without_changes(self):
        model = Model(width=48, adapter="linear").eval()
        data = fixture()
        before = model(*data).detach()
        report = geometry(model, data)
        after = model(*data).detach()
        torch.testing.assert_close(before, after)
        self.assertEqual(report["local"]["width"], 48)
        self.assertEqual(report["projected"]["width"], 32)
        self.assertLessEqual(report["local"]["numerical_rank"], 4)
        self.assertFalse(model.adapter._forward_hooks)
        self.assertFalse(model.adapter._forward_pre_hooks)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
