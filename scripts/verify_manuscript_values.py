"""Cross-check manuscript tables, plotted aggregates, and quoted numerical values."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_tables import build
from cvhnn.config import HOSTED, ROOT, SELECTED_HSV_SEED


def verify():
    tex = (ROOT / "paper/main.tex").read_text(encoding="utf8")
    (ROOT / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
        tmp = Path(directory)
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
            assert (tmp / "tables" / name).read_text() == (ROOT / "paper/Tables" / name).read_text(), name

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
    assert f"\\rho\\approx{rho:.3f}" in tex and f"b\\approx{boundary:.3f}" in tex
    assert f"{100 * boundary:.1f}\\%" in tex
    for key in ["symbol_initial_SER_range", "hsv_initial_SER_range"]:
        low, high = hosted_text_values[key]
        assert f"${low:.3f}$ to ${high:.3f}$" in tex
    publication_residual = json.loads((ROOT / "tables/text_values.json").read_text())["max_projection_residual"]
    for value in [publication_residual, hosted_text_values["max_projection_residual"]]:
        mantissa, exponent = f"{value:.2e}".split("e")
        assert f"${mantissa}\\times10^{{{int(exponent)}}}$" in tex

    # Figure 4 is the disclosed selected trial, distinct from the default image run.
    visual = pd.read_csv(HOSTED / "geometry/source_data/paper_visual_metrics.csv").set_index("rule")
    assert visual.trial_seed.eq(SELECTED_HSV_SEED).all()
    assert str(SELECTED_HSV_SEED) in tex
    for column, digits in [("initial_SER", 3), ("SER", 3), ("rgb_psnr", 2)]:
        assert f"{visual.loc['classic', column]:.{digits}f}" in tex
    assert visual.loc["voronoi", "SER"] == visual.loc["voronoi", "radial_error"] == 0
    assert visual.loc["voronoi", "rgb_psnr"] == 100
    recall = pd.read_csv(HOSTED / "recall/experiment2_noise_results.csv")
    qam_hsv = recall.query(
        'alphabet == "64-QAM" and corruption_kind == "hsv" and noise_level == 0.1 and rule == "voronoi"'
    )
    assert len(qam_hsv) == 30 and int(qam_hsv.exact_recovery.sum()) == 27
    assert "90\\%" in tex

    # Check both environments against the specific rankings stated in the text.
    for data in [raw, pd.read_csv(HOSTED / "activation_cost/experiment3_raw_timings.csv")]:
        median = data.groupby(keys).time_per_query_us.median().unstack("method")
        small = median.xs(16, level="cardinality")
        large = median.xs(1024, level="cardinality")
        assert (small.distance < small.kdtree).all()
        for (_, family), row in large.iterrows():
            assert row.idxmin() == ("kdtree" if family == "Hexagonal" else "structured")

    return {
        "status": "passed",
        "numerical_tables": "Both generated directly from hosted raw trials and exactly match manuscript TeX",
        "figure_3": "Every plotted mean and sample standard deviation matches hosted trajectories",
        "figure_4": "Selected seed and all quoted image metrics verified",
        "figure_5": "All 120 plotted timing aggregates match publication raw data; hosted timings remain separate",
        "table_III": "Analytical operation counts, not measured timings; excluded from empirical table regeneration",
        "qam_plateaus": {"rho": rho, "boundary_fraction": boundary},
        "initial_SER_ranges_hosted": {k: v for k, v in hosted_text_values.items() if "range" in k},
        "projection_residual": {
            "publication": publication_residual,
            "hosted": hosted_text_values["max_projection_residual"],
            "interpretation": (
                "The manuscript explicitly quotes both host measurements; the difference is floating-point roundoff"
            ),
        },
        "qam_hsv_exact_recovery": "27/30 = 90%",
        "timing_extreme_cardinality_rankings": "Verified independently for both hosts",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
