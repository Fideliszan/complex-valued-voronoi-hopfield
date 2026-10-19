"""Build four public, reproducible notebook companions."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
setup='from pathlib import Path\nimport os, sys, subprocess, uuid, json\n\nREPOSITORY = "https://github.com/Fideliszan/complex-valued-voronoi-hopfield.git"\nREVISION = "v1.0.2"\ntry:\n    import google.colab\n    IN_COLAB = True\nexcept ImportError:\n    IN_COLAB = False\n\nif IN_COLAB:\n    ROOT = Path("/content") / ("complex_voronoi_" + uuid.uuid4().hex[:8])\n    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", REVISION,\n                    REPOSITORY, str(ROOT)], check=True, capture_output=True, text=True)\nelse:\n    candidates = [Path.cwd(), *Path.cwd().parents]\n    ROOT = next((p for p in candidates if (p / "src/cvhnn").is_dir()), None)\n    if ROOT is None:\n        raise RuntimeError("Open this notebook from an extracted repository directory.")\nos.environ["MPLCONFIGDIR"] = str(ROOT / "runs/matplotlib")\nprint("Project:", ROOT.name)\nprint("Runtime:", "Google Colab" if IN_COLAB else "local")\nprint("Source:", REVISION if IN_COLAB else "local checkout")\n'
install='''if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)
print("Setup complete.")
'''
stages=[('00_geometry','Geometry and publication figures',None),('01_saturation','Magnitude saturation and energy','saturation'),('02_hsv_recall','HSV associative recall','recall'),('03_activation_cost','Exact activation cost','activation_cost')]
for name,title,stage in stages:
    n=nb.v4.new_notebook();cells=[nb.v4.new_markdown_cell(f'# {title}\n\nCompanion to *Voronoi-Based Activation Functions and Inertial Energy for Complex-Valued Hopfield Networks*.\n\n## Goal\nRun the same implementation as the paper with explicit inputs and outputs. The public setup downloads the tagged source without GitHub credentials. Full mode follows the paper protocol; smoke mode is an execution check. The final cell exports this run.'),nb.v4.new_markdown_cell('## Setup\nSelect a CPU runtime and run all cells in order. On Google Colab, setup clones the public release and installs its requirements. Locally, install requirements.txt and open the notebook from the extracted repository.'),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(install),nb.v4.new_markdown_cell('## Configuration\nSmoke mode checks execution with fewer trials. Full mode uses the paper protocol. Existing output directories are never overwritten.'),nb.v4.new_code_cell('MODE = "full"\nOUTPUT = ROOT / "runs" / ("'+name+'_" + uuid.uuid4().hex[:8])\nprint("Mode:", MODE)')]
    if stage:
        code=f'''command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "{stage}", "--mode", MODE, "--output", str(OUTPUT)]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
print("Experiment completed successfully." if completed.returncode == 0 else completed.stdout[-1200:])
if completed.returncode:
    print(completed.stderr[-3000:])
    raise RuntimeError("Experiment failed; inspect the message above.")
print("Execution record:", json.loads((OUTPUT / "execution.json").read_text())["seconds"], "seconds")
'''
    else:
        code='''command = [sys.executable, str(ROOT / "scripts/render_figures.py"), "--data", str(ROOT / "data/paper"), "--output", str(OUTPUT)]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
if completed.returncode:
    print(completed.stderr[-3000:])
    raise RuntimeError("Rendering failed.")
print("Five paper figures and the recall overview rendered.")
'''
    code += '''
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
'''
    cells += [nb.v4.new_markdown_cell('## Execute'),nb.v4.new_code_cell(code),nb.v4.new_markdown_cell('## Results and checks')]
    if stage=='saturation':
        result='''import pandas as pd
data = pd.read_csv(OUTPUT / "experiment1_final_runs.csv")
assert data.converged.all()
assert data.monotonicity_violations.max() == 0
display(data.groupby(["alphabet", "init_protocol", "rule"])[["rho", "boundary_fraction"]].mean().round(4))
import seaborn as sns
import matplotlib.pyplot as plt
history = pd.read_csv(OUTPUT / "experiment1_trajectories.csv")
subset = history[(history.alphabet == "16-QAM") & (history.init_protocol == "uniform")]
sns.set_theme(style="whitegrid", context="notebook")
ax = sns.lineplot(data=subset, x="sweep", y="rho", hue="rule", style="rule", markers=True, markevery=8, palette={"classic":"#CB701B", "voronoi":"#2864A0"}, errorbar="sd")
ax.set(xlabel="Asynchronous sweeps", ylabel="Normalized squared norm")
plt.show()
'''
    elif stage=='recall':
        result='''import pandas as pd
data = pd.read_csv(OUTPUT / "experiment2_noise_results.csv")
assert data.converged.all()
assert data.lyapunov_sweep_violations.max() == 0
display(data[data.noise_level == 0.10].groupby(["alphabet", "corruption_kind", "rule"])[["SER", "exact_recovery"]].mean().round(4))
from IPython.display import display, Image
display(Image(filename=str(OUTPUT / "figures/experiment2_noise_SER.png"), width=900))
'''
    elif stage=='activation_cost':
        result='''import pandas as pd
data = pd.read_csv(OUTPUT / "experiment3_summary.csv")
assert data.agreement_min.min() == 1
boundaries = pd.read_csv(OUTPUT / "boundary_validation.csv")
assert boundaries.disagreements.sum() == 0
display(data[data.ensemble == "local_fields"][["family", "cardinality", "method", "time_per_query_us_median", "agreement_min"]].round(4))
subprocess.run([sys.executable, str(ROOT / "scripts/render_cost.py"), "--data", str(OUTPUT), "--output", str(OUTPUT / "figures")], check=True, cwd=ROOT)
from IPython.display import display, Image
display(Image(filename=str(OUTPUT / "figures/activation_scaling.png"), width=1000))
'''
    else:
        result='''from IPython.display import display, Image
for name in ["generic_voronoi", "constellations", "saturation_diagnostics", "hsv_recall", "activation_scaling"]:
    display(Image(filename=str(OUTPUT / (name + ".png")), width=750))
print("Vector PDF/SVG and 600 dpi PNG files are available in the output folder.")
'''
    cells += [nb.v4.new_code_cell(result),nb.v4.new_markdown_cell('## Interpretation\nEnergy descent and target recovery answer different questions. Smoke results are execution checks, not paper estimates. Means and timings must be interpreted with the stated protocol; timing changes between machines are expected. For the full paper comparison, follow `docs/VALIDATION.md` and `docs/USAGE.md`.'),nb.v4.new_markdown_cell('## Save the outputs'),nb.v4.new_code_cell('''import shutil
archive = shutil.make_archive(str(OUTPUT) + "_download", "zip", OUTPUT)
if IN_COLAB:
    from google.colab import files
    shutil.copy2(archive, Path("/content") / Path(archive).name)
    print("Output ZIP in the Files sidebar:", Path(archive).name)
    files.download(archive)
else:
    print("Saved:", Path(archive).name)
''')]
    
    for i, cell in enumerate(cells):
        cell.id = f'{name}-{i:02d}'
    n.cells=cells;n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':name+'.ipynb'}}
    nb.validate(n);nb.write(n,ROOT/'colab'/f'{name}.ipynb')
print('Four notebooks created.')

