"""Build four public, reproducible notebook companions."""

import nbformat as nb

from cvhnn.config import ROOT

# Tag cloned on Colab; update when a release containing these notebooks is tagged.
REVISION = "v1.0.2"
REPOSITORY = "https://github.com/Fideliszan/complex-valued-voronoi-hopfield.git"
PAPER_TITLE = "Voronoi-Based Activation Functions and Inertial Energy for Complex-Valued Hopfield Networks"
NOTEBOOKS = [
    ("00_geometry", "Geometry and publication figures", None),
    ("01_saturation", "Magnitude saturation and energy", "saturation"),
    ("02_hsv_recall", "HSV associative recall", "recall"),
    ("03_activation_cost", "Exact activation cost", "activation_cost"),
]

INTRODUCTION = (
    "# {title}\n\n"
    f"Companion to *{PAPER_TITLE}*.\n\n"
    "## Goal\n"
    "Run the same implementation as the paper with explicit inputs and outputs. "
    "The public setup downloads the tagged source without GitHub credentials. "
    "Full mode follows the paper protocol; smoke mode is an execution check. "
    "The final cell keeps or discards this run according to the settings below."
)
SETTINGS_TEXT = (
    "## Run settings\n"
    "Smoke mode checks execution with fewer trials; full mode uses the paper protocol. "
    "`SAVE_DATA` and `SAVE_FIGURES` decide what remains in `runs/` after the notebook finishes. "
    "The checks below always run on the complete output; with both switches off, results are only shown "
    "in the notebook."
)
SETTINGS = """\
# Global run settings: edit here only.
MODE = "full"          # "full" = paper protocol, "smoke" = short execution check
SAVE_DATA = True       # keep CSV / NPZ / JSON results after the run
SAVE_FIGURES = True    # keep PNG / PDF figures after the run
print("Mode:", MODE, "| save data:", SAVE_DATA, "| save figures:", SAVE_FIGURES)
"""
SETUP_TEXT = (
    "## Setup\n"
    "Select a CPU runtime and run all cells in order. On Google Colab, setup clones the public release "
    "and installs its requirements. Locally, install requirements.txt and open the notebook from the "
    "extracted repository."
)
SETUP = f"""\
from pathlib import Path
import os, sys, subprocess, uuid, json

REPOSITORY = "{REPOSITORY}"
REVISION = "{REVISION}"
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    ROOT = Path("/content") / ("complex_voronoi_" + uuid.uuid4().hex[:8])
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", REVISION,
                    REPOSITORY, str(ROOT)], check=True, capture_output=True, text=True)
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next((p for p in candidates if (p / "src/cvhnn").is_dir()), None)
    if ROOT is None:
        raise RuntimeError("Open this notebook from an extracted repository directory.")
os.environ["MPLCONFIGDIR"] = str(ROOT / "runs/matplotlib")
print("Project:", ROOT.name)
print("Runtime:", "Google Colab" if IN_COLAB else "local")
print("Source:", REVISION if IN_COLAB else "local checkout")
"""
# Requirements are installed from the repository root, where `-e .` resolves.
INSTALL = """\
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True, cwd=ROOT)
print("Setup complete.")
"""
OUTPUT_TEXT = (
    "## Output folder\nEach run gets a new folder under `runs/`; existing output directories are never overwritten."
)
OUTPUT = """\
OUTPUT = ROOT / "runs" / ("{name}_" + uuid.uuid4().hex[:8])
# The run writes everything so the checks can read it; the save switches apply in the last cell.
RUN_ENV = {{**os.environ, "CVHNN_SAVE_DATA": "1", "CVHNN_SAVE_FIGURES": "1"}}
print("Output:", OUTPUT.relative_to(ROOT))"""
RUN_EXPERIMENT = """\
command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "{stage}", "--mode", MODE, "--output", str(OUTPUT)]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=RUN_ENV)
print("Experiment completed successfully." if completed.returncode == 0 else completed.stdout[-1200:])
if completed.returncode:
    print(completed.stderr[-3000:])
    raise RuntimeError("Experiment failed; inspect the message above.")
print("Execution record:", json.loads((OUTPUT / "execution.json").read_text())["seconds"], "seconds")
"""
RENDER_FIGURES = """\
command = [sys.executable, str(ROOT / "scripts/render_figures.py"),
           "--data", str(ROOT / "data/paper"), "--output", str(OUTPUT)]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=RUN_ENV)
if completed.returncode:
    print(completed.stderr[-3000:])
    raise RuntimeError("Rendering failed.")
print("Five paper figures and the recall overview rendered.")
"""
RUNTIME_RECORD = """
import platform, hashlib, datetime, importlib.metadata
runtime_record = {
    "mode": MODE, "host": "Google Colab" if IN_COLAB else "local",
    "platform": platform.platform(), "python": sys.version,
    "utc_finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_files_sha256": {str(p.relative_to(ROOT)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / "src/cvhnn").glob("*.py"))},
    "packages": {p: importlib.metadata.version(p) for p in
                 ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "plotnine"]},
}
(OUTPUT / "notebook_runtime.json").write_text(json.dumps(runtime_record, indent=2))
print("Environment:", runtime_record["packages"])
"""
RESULTS = {
    "saturation": """\
import pandas as pd
data = pd.read_csv(OUTPUT / "experiment1_final_runs.csv")
assert data.converged.all()
assert data.monotonicity_violations.max() == 0
display(data.groupby(["alphabet", "init_protocol", "rule"])[["rho", "boundary_fraction"]].mean().round(4))
import seaborn as sns
import matplotlib.pyplot as plt
history = pd.read_csv(OUTPUT / "experiment1_trajectories.csv")
subset = history[(history.alphabet == "16-QAM") & (history.init_protocol == "uniform")]
sns.set_theme(style="whitegrid", context="notebook")
ax = sns.lineplot(data=subset, x="sweep", y="rho", hue="rule", style="rule", markers=True, markevery=8,
                  palette={"classic": "#CB701B", "voronoi": "#2864A0"}, errorbar="sd")
ax.set(xlabel="Asynchronous sweeps", ylabel="Normalized squared norm")
plt.show()
""",
    "recall": """\
import pandas as pd
data = pd.read_csv(OUTPUT / "experiment2_noise_results.csv")
assert data.converged.all()
assert data.lyapunov_sweep_violations.max() == 0
moderate = data[data.noise_level == 0.10]
display(moderate.groupby(["alphabet", "corruption_kind", "rule"])[["SER", "exact_recovery"]].mean().round(4))
from IPython.display import display, Image
display(Image(filename=str(OUTPUT / "figures/experiment2_noise_SER.png"), width=900))
""",
    "activation_cost": """\
import pandas as pd
data = pd.read_csv(OUTPUT / "experiment3_summary.csv")
assert data.agreement_min.min() == 1
boundaries = pd.read_csv(OUTPUT / "boundary_validation.csv")
assert boundaries.disagreements.sum() == 0
columns = ["family", "cardinality", "method", "time_per_query_us_median", "agreement_min"]
display(data[data.ensemble == "local_fields"][columns].round(4))
subprocess.run([sys.executable, str(ROOT / "scripts/render_cost.py"),
                "--data", str(OUTPUT), "--output", str(OUTPUT / "figures")], check=True, cwd=ROOT)
from IPython.display import display, Image
display(Image(filename=str(OUTPUT / "figures/activation_scaling.png"), width=1000))
""",
    None: """\
from IPython.display import display, Image
for name in ["generic_voronoi", "constellations", "saturation_diagnostics", "hsv_recall", "activation_scaling"]:
    display(Image(filename=str(OUTPUT / (name + ".png")), width=750))
print("Vector PDF and 600 dpi PNG files are in the output folder until the save settings are applied.")
""",
}
INTERPRETATION = (
    "## Interpretation\n"
    "Energy descent and target recovery answer different questions. Smoke results are execution checks, "
    "not paper estimates. Means and timings must be interpreted with the stated protocol; timing changes "
    "between machines are expected. For the full paper comparison, follow `docs/VALIDATION.md` and "
    "`docs/USAGE.md`."
)
SAVE_TEXT = (
    "## Save or discard the outputs\n"
    "Applies `SAVE_DATA` and `SAVE_FIGURES`. Run records (`execution.json`, `notebook_runtime.json`) "
    "are kept with whatever is saved."
)
SAVE = """\
import shutil
# Same rule as cvhnn.config.retain, inlined so the notebook also works with the tagged source.
FIGURE_SUFFIXES = {".png", ".pdf", ".svg"}
RECORDS = {"execution.json", "notebook_runtime.json"}
sibling = OUTPUT.with_name(OUTPUT.name + ".zip")
if not (SAVE_DATA and SAVE_FIGURES) and sibling.exists():
    sibling.unlink()
if not (SAVE_DATA or SAVE_FIGURES):
    shutil.rmtree(OUTPUT, ignore_errors=True)
    print("Nothing saved (SAVE_DATA = SAVE_FIGURES = False); the results are shown above.")
else:
    for path in sorted(OUTPUT.rglob("*"), reverse=True):
        if path.is_file() and path.name not in RECORDS:
            figure = path.suffix.lower() in FIGURE_SUFFIXES
            if not (SAVE_FIGURES if figure else SAVE_DATA):
                path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    archive = shutil.make_archive(str(OUTPUT) + "_download", "zip", OUTPUT)
    if IN_COLAB:
        from google.colab import files
        shutil.copy2(archive, Path("/content") / Path(archive).name)
        print("Output ZIP in the Files sidebar:", Path(archive).name)
        files.download(archive)
    else:
        print("Saved:", OUTPUT.relative_to(ROOT), "and", Path(archive).name)
"""


def build_notebook(name, title, stage):
    execute = RUN_EXPERIMENT.format(stage=stage) if stage else RENDER_FIGURES
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    cells = [
        md(INTRODUCTION.format(title=title)),
        md(SETTINGS_TEXT),
        code(SETTINGS),
        md(SETUP_TEXT),
        code(SETUP),
        code(INSTALL),
        md(OUTPUT_TEXT),
        code(OUTPUT.format(name=name)),
        md("## Execute"),
        code(execute + RUNTIME_RECORD),
        md("## Results and checks"),
        code(RESULTS[stage]),
        md(INTERPRETATION),
        md(SAVE_TEXT),
        code(SAVE),
    ]
    for i, cell in enumerate(cells):
        cell.id = f"{name}-{i:02d}"
    notebook = nb.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": name + ".ipynb"},
    }
    nb.validate(notebook)
    return notebook


def main():
    for name, title, stage in NOTEBOOKS:
        nb.write(build_notebook(name, title, stage), ROOT / "colab" / f"{name}.ipynb")
    print("Four notebooks created.")


if __name__ == "__main__":
    main()
