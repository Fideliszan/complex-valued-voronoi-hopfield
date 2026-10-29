"""Run-level configuration shared by the launchers, scripts, notebooks, and checks.

Scientific protocol constants (seeds, network sizes, trial counts, dynamics
tolerances) deliberately stay at the top of their experiment modules
(``saturation``, ``recall``, ``activation_cost``): every run records those
module constants in its ``execution.json``. This module holds what is common
to all runs: repository paths, stage names, smoke-mode sizes, the comparison
tolerance, and the global save switches.
"""

import os
import shutil
from pathlib import Path

# Paths (the package is installed in editable mode from the repository).
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
PAPER_DATA = ROOT / "data" / "paper"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
PAPER = ROOT / "paper"
VALIDATION = ROOT / "validation"
HOSTED = VALIDATION / "hosted"
RUNS = ROOT / "runs"

# Experiment stages, in execution order.
STAGES = ("saturation", "recall", "activation_cost")

# Reduced sizes for `--mode smoke`; full mode uses the module defaults.
SMOKE_OVERRIDES = {
    "saturation": {"N_RUNS": 2, "N": 20},
    "recall": {"N_TRIALS_NOISE": 1, "N_TRIALS_LOAD": 1},
    "activation_cost": {"N_QUERY_UNIFORM": 200, "N_QUERY_FIELDS": 200, "N_REPEATS": 2, "MIN_FIELD_TRIALS": 1},
}

# Absolute tolerance for comparing new scientific measurements with data/paper.
COMPARISON_TOLERANCE = 1e-9

# The disclosed successful HSV trial shown in Figure 4.
SELECTED_HSV_SEED = 120044901


# Global save switches. SAVE_DATA keeps tabular and array results (CSV, NPZ,
# JSON, TXT); SAVE_FIGURES keeps images (PNG, PDF, SVG). Both default to on and
# are read from CVHNN_SAVE_DATA / CVHNN_SAVE_FIGURES ("0" disables). Runs always
# write a complete output first, so checks can read it; `retain` then removes
# what the switches do not keep. The notebooks carry an inline copy of `retain`
# so that they also work with the tagged source cloned on Colab.
def flag(name, default=True):
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


SAVE_DATA = flag("CVHNN_SAVE_DATA")
SAVE_FIGURES = flag("CVHNN_SAVE_FIGURES")
FIGURE_SUFFIXES = {".png", ".pdf", ".svg"}
# Provenance of whatever is kept: configuration, environment, comparison verdict.
RECORDS = {"execution.json", "notebook_runtime.json", "comparison.json"}


def retain(output, save_data=None, save_figures=None):
    """Delete the parts of a finished run directory that are not to be saved."""
    output = Path(output)
    save_data = SAVE_DATA if save_data is None else save_data
    save_figures = SAVE_FIGURES if save_figures is None else save_figures
    # Some protocols also write a complete <output>.zip next to the directory.
    sibling = output.with_name(output.name + ".zip")
    if not (save_data and save_figures) and sibling.exists():
        sibling.unlink()
    if not (save_data or save_figures):
        shutil.rmtree(output, ignore_errors=True)
        return
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file() and path.name not in RECORDS:
            figure = path.suffix.lower() in FIGURE_SUFFIXES
            if not (save_figures if figure else save_data):
                path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
