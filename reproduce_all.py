"""Run all three experiments sequentially, then compare their trial results."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cvhnn import config
from cvhnn.config import PAPER_DATA, ROOT, STAGES

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--no-save-data",
        dest="save_data",
        action="store_false",
        default=config.SAVE_DATA,
        help=(
            "Discard trial data. Keep stage run records when figures are retained; "
            "with both save switches off, keep only the top-level comparison.json."
        ),
    )
    p.add_argument(
        "--no-save-figures",
        dest="save_figures",
        action="store_false",
        default=config.SAVE_FIGURES,
        help="Discard generated figures.",
    )
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory.")
    # Stages keep everything until the comparison has read it; the switches apply afterwards.
    env = {**os.environ, "CVHNN_SAVE_DATA": "1", "CVHNN_SAVE_FIGURES": "1"}
    for stage in STAGES:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_experiment.py"),
                stage,
                "--mode",
                a.mode,
                "--output",
                str(out / stage),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )
    comparison = [
        sys.executable,
        str(ROOT / "scripts/compare_results.py"),
        str(out),
        "--reference",
        str(PAPER_DATA),
        "--output",
        str(out / "comparison.json"),
    ]
    if a.mode == "smoke":
        comparison.append("--allow-partial")
    subprocess.run(comparison, check=True, cwd=ROOT)
    for stage in STAGES:
        config.retain(out / stage, a.save_data, a.save_figures)
