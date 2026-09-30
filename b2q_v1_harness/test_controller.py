"""Failure-path tests independent of the developer model's success path."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lab


class ControllerFailures(unittest.TestCase):
    def test_reported_usage_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            path.write_text('not json\n' + json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 4}}) + '\n')
            self.assertEqual(lab.reported_tokens(path), 14)

    def test_timeout_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / 'log').open('w') as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    lab.bounded_run([sys.executable, '-c', 'import time; time.sleep(2)'], timeout=0.05, stdout=log, stderr=log)

    def test_stale_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'out.json'
            lab.write(path, {'status': 'PASS', 'run_id': 'old'})
            self.assertEqual(lab.checked_result(path, 'new', 0)['status'], 'FAIL')

    def test_nonzero_exit_overrides_matching_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'out.json'
            lab.write(path, {'status': 'PASS', 'run_id': 'current'})
            self.assertEqual(lab.checked_result(path, 'current', 1)['status'], 'FAIL')

    def test_training_exception_produces_failed_fresh_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = (lab.ROOT / 'candidate.py').read_text()
            # Baseline contracts use eval mode. Fail only when training starts.
            source = source.replace('        # Sensor pooling', "        if self.training:\n            raise RuntimeError('injected training failure')\n        # Sensor pooling")
            (workspace / 'candidate.py').write_text(source)
            output = workspace / 'report.json'
            result = subprocess.run([sys.executable, str(lab.ROOT / 'lab.py'), '_evaluate', str(workspace), '--output', str(output), '--run-id', 'fresh'], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report['run_id'], 'fresh')
            self.assertEqual(report['status'], 'FAIL')
            self.assertTrue(all(check['status'] == 'PASS' for check in report['checks']))
            self.assertIn('injected training failure', report['error'])


if __name__ == '__main__':
    unittest.main()
