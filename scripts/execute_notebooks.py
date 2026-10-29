"""Execute all distributed notebooks and record evidence without storing their outputs.

Run from the installed repository environment. The distributed full-mode code is
executed unchanged; results stay in ignored runs/ folders. Use --record
validation/notebook_execution.json to explicitly refresh the reference evidence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from cvhnn.config import ROOT


def execute(record_path):
    records = {}
    # Use this Python interpreter even if another system Jupyter kernel exists.
    # A temporary kernelspec avoids modifying the user's Jupyter installation.
    with tempfile.TemporaryDirectory(prefix="cvhnn-kernel-") as tmp:
        kernel = Path(tmp) / "cvhnn-validation"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "CVHNN validation",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        manager = KernelSpecManager(kernel_dirs=[tmp])
        for path in sorted((ROOT / "colab").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            hashes = [hashlib.sha256(cell.source.encode()).hexdigest() for cell in code]
            start = time.perf_counter()
            print(f"Executing {path.name} in full mode...", flush=True)
            km = AsyncKernelManager(kernel_name="cvhnn-validation", kernel_spec_manager=manager)
            NotebookClient(notebook, km=km, timeout=1800, resources={"metadata": {"path": str(ROOT)}}).execute(
                cleanup_kc=True
            )
            output_text = "\n".join(output.get("text", "") for cell in code for output in cell.get("outputs", []))
            if "Mode: full |" not in output_text or any(cell.execution_count is None for cell in code):
                raise AssertionError(
                    f"{path.name}: the distributed full-mode code must finish before recording evidence"
                )
            records[path.name] = {
                "status": "passed",
                "mode": "full",
                "environment": "local",
                "seconds": time.perf_counter() - start,
                "utc_finished": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "code_cell_sha256": hashes,
                "rendered_png_outputs": sum(
                    "image/png" in output.get("data", {}) for cell in code for output in cell.get("outputs", [])
                ),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "plotnine", "nbclient")
                },
            }
            print(f"PASS: {path.name} ({records[path.name]['seconds']:.1f} s)", flush=True)
    # Do not replace successful evidence with a partially completed run.
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Execution evidence: {record_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "runs/notebook_execution.json")
    args = parser.parse_args()
    execute(args.record.resolve())


if __name__ == "__main__":
    main()
