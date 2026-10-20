"""Compare new trial measurements, excluding timings and roundoff diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cvhnn.config import COMPARISON_TOLERANCE


def compare(run, reference):
    report = {}
    for stage, name, keys, metrics in [
        (
            "saturation",
            "experiment1_final_runs.csv",
            ["alphabet", "init_protocol", "run", "rule"],
            ["rho", "boundary_fraction", "E_c", "E_r", "steps_to_convergence", "converged"],
        ),
        (
            "recall",
            "experiment2_noise_results.csv",
            ["alphabet", "corruption_kind", "noise_level", "trial_seed", "rule"],
            [
                "initial_SER",
                "SER",
                "exact_recovery",
                "radial_error",
                "sweeps",
                "rgb_psnr",
                "complex_similarity",
                "angular_error_cycles",
                "initial_E_c",
                "initial_E_r",
                "final_E_c",
                "final_E_r",
                "changed_updates",
                "attempted_updates",
                "converged",
                "lyapunov_sweep_violations",
            ],
        ),
        (
            "recall",
            "experiment2_load_results.csv",
            ["alphabet", "P", "trial_seed", "rule"],
            [
                "SER",
                "exact_recovery",
                "radial_error",
                "rgb_psnr",
                "sweeps",
                "changed_updates",
                "final_E_c",
                "final_E_r",
            ],
        ),
    ]:
        a = pd.read_csv(reference / name).sort_values(keys).reset_index(drop=True)
        b = pd.read_csv(run / stage / name).sort_values(keys).reset_index(drop=True)
        item = {
            "reference_rows": len(a),
            "new_rows": len(b),
            "keys_match": len(a) == len(b) and a[keys].equals(b[keys]),
        }
        if item["keys_match"]:
            item["max_absolute_difference"] = {
                c: float(np.max(np.abs(a[c].astype(float) - b[c].astype(float)))) for c in metrics
            }
            item["matches_at_1e-9"] = all(x <= COMPARISON_TOLERANCE for x in item["max_absolute_difference"].values())
        report[name] = item
    p = run / "activation_cost/experiment3_raw_timings.csv"
    if p.exists():
        d = pd.read_csv(p)
        report["activation_cost"] = {
            "rows": len(d),
            "minimum_agreement": float(d.agreement_with_distance.min()),
            "timings_expected_to_differ": True,
        }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--reference", type=Path, default=Path("data/paper"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true", help="Allow the deliberately smaller smoke protocol.")
    a = p.parse_args()
    r = compare(a.run, a.reference)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
    # Partial smoke runs intentionally contain different trial counts.
    failed = (
        any(
            (v.get("keys_match") is False and not a.allow_partial)
            or (v.get("keys_match") and not v.get("matches_at_1e-9", True))
            for v in r.values()
        )
        or r.get("activation_cost", {}).get("minimum_agreement", 1) < 1
    )
    if failed:
        raise SystemExit(1)
