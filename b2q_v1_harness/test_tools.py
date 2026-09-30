import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from checks import fixture
from lab import ROOT, prepare
from tools.fixture_smoke import smoke
from tools.inspect_candidate import inspect


class ToolWorkflows(unittest.TestCase):
    def test_context_conditions_share_candidate_and_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            structured = prepare(Path(tmp) / "structured", "structured")
            minimal = prepare(Path(tmp) / "minimal", "minimal")
            for name in ("candidate.py", "AGENTS.md", "TASK.md"):
                self.assertEqual(
                    (structured / name).read_bytes(), (minimal / name).read_bytes()
                )
            self.assertTrue((structured / "INSPECTION.json").exists())
            self.assertTrue((structured / "MATH_CONTEXT.md").exists())
            self.assertFalse((minimal / "INSPECTION.json").exists())
            self.assertFalse((minimal / "MATH_CONTEXT.md").exists())

    def test_inspector_reports_actual_widths_and_gradients(self):
        report = inspect(ROOT / "verification/manual_candidate.py")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["module_trace"]["sensor"]["output"][-1], 48)
        self.assertEqual(report["module_trace"]["adapter"]["output"][-1], 32)
        self.assertTrue(report["input_gradients"]["signals"]["nonzero"])
        self.assertTrue(report["parameter_gradients"]["adapter.weight"]["nonzero"])

    def test_inspector_detects_detached_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text(
                (ROOT / "candidate.py")
                .read_text()
                .replace("fault=None", "fault='detached_upstream'")
            )
            report = inspect(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["input_gradients"]["signals"]["present"])

    def test_fixture_accepts_both_montage_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for channels in (32, 61):
                x, coordinates, subjects, sentences, _ = fixture(channels)
                path = Path(tmp) / f"{channels}.npz"
                np.savez(
                    path,
                    x=x.numpy(),
                    coordinates=coordinates.numpy(),
                    subjects=subjects.numpy(),
                    sentences=sentences.numpy(),
                )
                report = smoke(path, source_kind="synthetic")
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["source_kind"], "synthetic")
                self.assertEqual(report["logits_shape"], [7, 29])

    def test_fixture_rejects_misaligned_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            x, coordinates, subjects, sentences, _ = fixture()
            path = Path(tmp) / "bad.npz"
            np.savez(
                path,
                x=x.numpy(),
                coordinates=coordinates[:, :-1].numpy(),
                subjects=subjects.numpy(),
                sentences=sentences.numpy(),
            )
            report = smoke(path)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertIn("coordinates", report["error"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
