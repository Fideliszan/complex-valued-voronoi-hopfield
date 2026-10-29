"""Cross-check numerical reference tables, plotted aggregates, and recorded trial values."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_tables import build
from cvhnn.config import HOSTED, PAPER_DATA, ROOT, SELECTED_HSV_SEED, TABLES


def compare_table_files(generated, reference):
    """Validate every generated numeric and TeX table against the saved reference."""
    for name in ("saturation", "recall", "activation_cost"):
        pd.testing.assert_frame_equal(
            pd.read_csv(generated / f"{name}.csv"),
            pd.read_csv(reference / f"{name}.csv"),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            obj=f"{name}.csv",
        )
    for name in ("saturation.tex", "recall.tex"):
        assert (generated / name).read_text() == (reference / name).read_text(), name
    actual = json.loads((generated / "text_values.json").read_text())
    expected = json.loads((reference / "text_values.json").read_text())
    assert actual.keys() == expected.keys(), "text_values.json keys"
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-15, err_msg=key)


def verify(reference=TABLES):
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        tmp = Path(directory)
        publication_text_values = build(PAPER_DATA, tmp / "publication_tables")
        compare_table_files(tmp / "publication_tables", reference)
        inputs = tmp / "inputs"
        inputs.mkdir()
        for stage, name in [
            ("saturation", "experiment1_final_runs.csv"),
            ("recall", "experiment2_noise_results.csv"),
            ("recall", "experiment2_weight_validation.csv"),
            ("activation_cost", "experiment3_raw_timings.csv"),
        ]:
            shutil.copy2(HOSTED / stage / name, inputs / name)
        hosted_text_values = build(inputs, tmp / "tables")
        for name in ["saturation.tex", "recall.tex"]:
            assert (tmp / "tables" / name).read_text() == (reference / name).read_text(), name
        for name in ("saturation", "recall"):
            pd.testing.assert_frame_equal(
                pd.read_csv(tmp / "tables" / f"{name}.csv"),
                pd.read_csv(reference / f"{name}.csv"),
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
                obj=f"hosted {name}.csv",
            )

    # Figure 3: every mean and sample standard deviation, not only endpoints.
    # Identical to the hosted history (see validation/hosted/saturation/hosted_comparison.json).
    history = pd.read_csv(ROOT / "data/paper/experiment1_trajectories.csv.gz")
    subset = history.query('alphabet == "16-QAM" and init_protocol == "uniform"')
    expected = (
        subset.groupby(["rule", "sweep"])[["E_c", "E_r", "rho", "boundary_fraction"]].agg(["mean", "std"]).reset_index()
    )
    plotted = pd.read_csv(ROOT / "figures/source_data/saturation_plot.csv", header=[0, 1])
    assert list(expected["rule"]) == list(plotted.iloc[:, 0])
    np.testing.assert_allclose(
        expected.iloc[:, 1:].to_numpy(dtype=float), plotted.iloc[:, 1:].to_numpy(dtype=float), rtol=1e-13, atol=1e-13
    )

    # Figure 5: publication timings stay distinct from the hosted timings.
    raw = pd.read_csv(ROOT / "data/paper/experiment3_raw_timings.csv")
    keys = ["ensemble", "family", "cardinality", "method"]
    expected_cost = raw.groupby(keys, as_index=False).agg(
        median=("time_per_query_us", "median"), lower=("time_per_query_us", "min"), upper=("time_per_query_us", "max")
    )
    plotted_cost = pd.read_csv(ROOT / "figures/source_data/activation_scaling.csv")
    matched = expected_cost.merge(plotted_cost, on=keys, validate="one_to_one", suffixes=("_raw", "_plot"))
    assert len(matched) == len(expected_cost) == len(plotted_cost) == 120
    for metric in ["median", "lower", "upper"]:
        np.testing.assert_allclose(matched[metric + "_raw"], matched[metric + "_plot"], rtol=1e-13, atol=1e-13)

    final = pd.read_csv(HOSTED / "saturation/experiment1_final_runs.csv")
    qam = final.query('alphabet == "16-QAM" and init_protocol == "uniform" and rule == "voronoi"')
    assert len(qam) == 100
    rho, boundary = float(qam.rho.mean()), float(qam.boundary_fraction.mean())
    plateau = pd.read_csv(reference / "saturation.csv").set_index("alphabet").loc["16-QAM"]
    np.testing.assert_allclose([rho, boundary], [plateau.rho_v, plateau.b_v], rtol=1e-12, atol=1e-12)
    for key in ["symbol_initial_SER_range", "hsv_initial_SER_range"]:
        np.testing.assert_allclose(hosted_text_values[key], publication_text_values[key], rtol=1e-12, atol=1e-12)
    publication_residual = publication_text_values["max_projection_residual"]
    for value in [publication_residual, hosted_text_values["max_projection_residual"]]:
        assert np.isfinite(value) and 0 <= value < 1e-12

    # Figure 4 is the disclosed selected trial, distinct from the default image run.
    visual = pd.read_csv(HOSTED / "geometry/source_data/paper_visual_metrics.csv").set_index("rule")
    assert visual.trial_seed.eq(SELECTED_HSV_SEED).all()
    publication_visual = pd.read_csv(ROOT / "figures/source_data/paper_visual_metrics.csv").set_index("rule")
    pd.testing.assert_frame_equal(visual, publication_visual, check_exact=True)
    assert visual.loc["voronoi", "SER"] == visual.loc["voronoi", "radial_error"] == 0
    assert visual.loc["voronoi", "rgb_psnr"] == 100
    recall = pd.read_csv(HOSTED / "recall/experiment2_noise_results.csv")
    qam_hsv = recall.query(
        'alphabet == "64-QAM" and corruption_kind == "hsv" and noise_level == 0.1 and rule == "voronoi"'
    )
    assert len(qam_hsv) == 30 and int(qam_hsv.exact_recovery.sum()) == 27
    recall_reference = pd.read_csv(reference / "recall.csv").set_index(["kind", "alphabet"])
    np.testing.assert_allclose(qam_hsv.exact_recovery.mean(), recall_reference.loc[("hsv", "64-QAM"), "exact_v"])

    # Check the recorded rankings separately for each benchmark host.
    for data in [raw, pd.read_csv(HOSTED / "activation_cost/experiment3_raw_timings.csv")]:
        median = data.groupby(keys).time_per_query_us.median().unstack("method")
        small = median.xs(16, level="cardinality")
        large = median.xs(1024, level="cardinality")
        assert (small.distance < small.kdtree).all()
        for (_, family), row in large.iterrows():
            assert row.idxmin() == ("kdtree" if family == "Hexagonal" else "structured")

    return {
        "status": "passed",
        "numerical_tables": (
            "Publication CSV/TeX/JSON tables regenerated and compared with saved references; "
            "hosted saturation and recall tables independently agree"
        ),
        "figure_3": "Every plotted mean and sample standard deviation matches reference trajectories",
        "figure_4": "Selected seed and reference image metrics verified",
        "figure_5": "All 120 plotted timing aggregates match publication raw data; hosted timings remain separate",
        "qam_plateaus": {"rho": rho, "boundary_fraction": boundary},
        "initial_SER_ranges_hosted": {k: v for k, v in hosted_text_values.items() if "range" in k},
        "projection_residual": {
            "publication": publication_residual,
            "hosted": hosted_text_values["max_projection_residual"],
            "interpretation": ("Each host retains its own residual; both satisfy the numerical projection check"),
        },
        "qam_hsv_exact_recovery": "27/30 = 90%",
        "timing_extreme_cardinality_rankings": "Verified independently for both hosts",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, default=TABLES)
    args = parser.parse_args()
    report = verify(args.reference)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
