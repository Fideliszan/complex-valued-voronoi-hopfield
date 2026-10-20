"""Verify full hosted measurements against the manuscript and publication data."""

import hashlib
import json

import pandas as pd

from compare_results import compare
from cvhnn.config import FIGURES, PAPER_DATA
from cvhnn.config import HOSTED as ARCHIVE


def main():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == digest, name
    for stage in ["geometry", "saturation", "recall", "activation_cost"]:
        runtime = json.loads((ARCHIVE / stage / "notebook_runtime.json").read_text())
        assert runtime["host"] == "Google Colab" and runtime["mode"] == "full"
    report = compare(ARCHIVE, PAPER_DATA)
    for name, item in report.items():
        if name != "activation_cost":
            assert item["keys_match"] and item["matches_at_1e-9"], name
    assert report["activation_cost"]["rows"] == 600
    assert report["activation_cost"]["minimum_agreement"] == 1
    # The hosted history parsed identically to the publication history, so only
    # the comparison record is archived; the shared table is in data/paper.
    history = json.loads((ARCHIVE / "saturation/hosted_comparison.json").read_text())["experiment1_trajectories.csv"]
    assert history["keys_match"] and history["pass_at_1e-9"]
    assert all(d.get("max_abs", 0) == 0 and d.get("equal", True) for d in history["differences"].values())
    b = pd.read_csv(PAPER_DATA / "experiment1_trajectories.csv.gz")
    assert history["rows"] == history["reference_rows"] == len(b) == 129600
    visual = pd.read_csv(ARCHIVE / "geometry/source_data/paper_visual_metrics.csv")
    original = pd.read_csv(FIGURES / "source_data/paper_visual_metrics.csv")
    pd.testing.assert_frame_equal(visual, original, check_exact=True)
    for source in (ARCHIVE / "geometry/source_data").glob("*.csv"):
        header = [0, 1] if source.name == "saturation_plot.csv" else 0
        hosted = pd.read_csv(source, header=header)
        published = pd.read_csv(FIGURES / "source_data" / source.name, header=header)
        pd.testing.assert_frame_equal(hosted, published, check_exact=False, rtol=1e-12, atol=1e-12)
    boundary = pd.read_csv(ARCHIVE / "activation_cost/boundary_validation.csv")
    assert int(boundary.queries.sum()) == 61380
    assert boundary.disagreements.sum() == 0
    for stage in ["saturation", "recall"]:
        audit = json.loads((ARCHIVE / stage / "coordinate_audit_summary.json").read_text())
        assert (
            audit["coordinate_energy_increases"]
            == audit["nonnegative_energy_changes"]
            == audit["terminal_rule_violations"]
            == 0
        )
    query_report = json.loads((ARCHIVE / "activation_cost/hosted_comparison.json").read_text())
    assert len(query_report["query_array_max_abs_differences"]) == 32
    for arrays in query_report["query_array_max_abs_differences"].values():
        assert arrays["current"] == arrays["alphabet"] == 0
        assert arrays["field"] < 1e-12
    print(
        "PASS: full hosted numerical protocols; raw recall data; 129600 saturation history rows "
        "(recorded comparison); 640000 regenerated queries (recorded comparison); 61380 boundary decisions; "
        "energy audits."
    )


if __name__ == "__main__":
    main()
