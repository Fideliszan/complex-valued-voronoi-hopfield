"""Check reference-data integrity and numerical reproducibility without a manuscript."""

import hashlib
import json

import nbformat
import pandas as pd

from cvhnn.config import ROOT


def main():
    manifest = json.loads((ROOT / "validation/manifest.json").read_text())
    for name, digest in manifest.items():
        p = ROOT / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Checksum mismatch: {name}")
    notebook_checks = json.loads((ROOT / "validation/notebook_execution.json").read_text())
    for p in (ROOT / "colab").glob("*.ipynb"):
        notebook = nbformat.read(p, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        record = notebook_checks[p.name]
        assert record["status"] == "passed" and record["mode"] == "full", p.name
        assert [hashlib.sha256(c.source.encode()).hexdigest() for c in cells] == record["code_cell_sha256"], p.name
        # Execution evidence is stored with source hashes; distributed notebooks keep code only.
        assert all(c.execution_count is None for c in cells), f"{p.name}: clear execution counts"
        assert all(not c.outputs for c in cells), f"{p.name}: clear stored outputs"
    for name, rows in [
        ("experiment1_final_runs.csv", 1600),
        ("experiment2_noise_results.csv", 2400),
        ("experiment2_load_results.csv", 64),
    ]:
        d = pd.read_csv(ROOT / "data/paper" / name)
        assert len(d) == rows, name
        assert d.converged.all(), name
    boundary = pd.read_csv(ROOT / "validation/boundary_validation.csv")
    assert boundary.disagreements.sum() == 0
    costs = pd.read_csv(ROOT / "data/paper/experiment3_raw_timings.csv")
    assert set(costs.cardinality) == {16, 64, 256, 1024}
    assert len(costs) == 600 and costs.agreement_with_distance.min() == 1
    for stage in ("saturation", "recall"):
        audit = json.loads((ROOT / "validation" / f"{stage}_coordinate_audit_summary.json").read_text())
        assert audit["coordinate_energy_increases"] == audit["terminal_rule_violations"] == 0
    from verify_hosted_results import main as verify_colab

    verify_colab()
    from verify_reference_values import verify as verify_values

    verify_values()
    from verify_geometry import verify as verify_geometry

    verify_geometry()
    print(
        f"PASS: {len(manifest)} checksums; regenerated reference tables and values; figure source data; "
        "four notebooks; full protocols; coordinate audits. No manuscript files required."
    )


if __name__ == "__main__":
    main()
