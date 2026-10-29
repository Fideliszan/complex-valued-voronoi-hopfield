"""Regression checks for the manuscript-free reference validation workflow."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cvhnn.config import ROOT, TABLES


class ReferenceValidation(unittest.TestCase):
    def test_reference_checker_rejects_changed_numerical_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "tables"
            shutil.copytree(TABLES, reference)
            values_path = reference / "text_values.json"
            values = json.loads(values_path.read_text())
            values["hsv_initial_SER_range"][0] += 0.01
            values_path.write_text(json.dumps(values))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/verify_reference_values.py"),
                    "--reference",
                    str(reference),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("hsv_initial_SER_range", result.stderr)


if __name__ == "__main__":
    unittest.main()
