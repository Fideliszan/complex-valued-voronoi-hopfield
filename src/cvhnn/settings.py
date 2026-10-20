"""Global output switches shared by the launchers.

SAVE_DATA keeps tabular and array results (CSV, NPZ, JSON, TXT); SAVE_FIGURES
keeps images (PNG, PDF, SVG). Both default to on and are read from the
environment variables CVHNN_SAVE_DATA and CVHNN_SAVE_FIGURES ("0" disables).
Experiments always write a complete run first, so that comparisons and checks
can read it; ``retain`` then removes what the switches do not keep.
"""
from pathlib import Path
import os
import shutil

FIGURE_SUFFIXES = {'.png', '.pdf', '.svg'}
# Provenance of whatever is kept: configuration, environment, comparison verdict.
RECORDS = {'execution.json', 'notebook_runtime.json', 'comparison.json'}


def flag(name, default=True):
    return os.getenv(name, '1' if default else '0').strip().lower() not in ('0', 'false', 'no', '')


SAVE_DATA = flag('CVHNN_SAVE_DATA')
SAVE_FIGURES = flag('CVHNN_SAVE_FIGURES')


def retain(output, save_data=None, save_figures=None):
    """Delete the parts of a finished run directory that are not to be saved."""
    output = Path(output)
    save_data = SAVE_DATA if save_data is None else save_data
    save_figures = SAVE_FIGURES if save_figures is None else save_figures
    # Some protocols also write a complete <output>.zip next to the directory.
    sibling = output.with_name(output.name + '.zip')
    if not (save_data and save_figures) and sibling.exists():
        sibling.unlink()
    if not (save_data or save_figures):
        shutil.rmtree(output, ignore_errors=True)
        return
    for path in sorted(output.rglob('*'), reverse=True):
        if path.is_file() and path.name not in RECORDS:
            figure = path.suffix.lower() in FIGURE_SUFFIXES
            if not (save_figures if figure else save_data):
                path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
