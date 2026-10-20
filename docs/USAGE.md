# Usage

[Home](../README.md) · [Methods](METHODS.md) · [Validation](VALIDATION.md)

1. [Environment](#environment) · 2. [Run the experiments](#run-the-experiments) · 3. [Save switches](#save-switches) · 4. [Google Colab](#google-colab) · 5. [What is stored and what is regenerated](#what-is-stored-and-what-is-regenerated) · 6. [Paper-to-code map](#paper-to-code-map)

## Environment

Use Python 3.12 or 3.13 and run `python -m pip install -r requirements.txt` from the repository root. This installs the `cvhnn` package in editable mode together with the dependencies declared in `pyproject.toml`. Direct package versions from the publication run are recorded in `requirements-lock.txt`; to use exactly those, run `python -m pip install -r requirements-lock.txt` followed by `python -m pip install --no-deps -e .`. Run-level settings (paths, stages, smoke-mode sizes, comparison tolerance, save switches) are in `src/cvhnn/config.py`; protocol constants stay at the top of each experiment module, because every run records them in `execution.json`, together with the actual Python, OS, NumPy, pandas, and SciPy versions.

One BLAS thread is used by the launchers. Matplotlib uses a noninteractive backend for exported figures. No LaTeX installation is required for simulation or plotting; manuscript compilation uses pdfLaTeX/BibTeX-compatible sources and the IEEEtran class.

## Run the experiments

Run commands from the project root. `--mode smoke` reduces repetition counts while retaining all alphabet families; `--mode full` uses the paper protocol. Each launcher requires `--output` and refuses to overwrite an existing directory.

```bash
python reproduce_all.py --mode full --output runs/my_run
python scripts/compare_results.py runs/my_run --reference data/paper --output runs/my_run/comparison.json
```

`reproduce_all.py` already runs the comparison and writes `comparison.json`; the second line repeats it for an existing run. The comparison tests scientific measurements separately from hardware-dependent timings. Inspect discrepancies; a benchmark timing change alone is not a scientific failure. A full run uses one BLAS thread. It needs a few minutes on the authoring machine; runtime and peak memory depend on the host. Dense recall uses 576 neurons; query batches are limited to 4096 entries.

To regenerate only artwork, use `python scripts/render_figures.py --data data/paper --output runs/figures`. This does not rerun the Monte Carlo experiments; it does rerun the documented illustrative image trial. To regenerate only the two Voronoi diagrams (Figures 1 and 2), use `python scripts/render_geometry.py --output runs/geometry`. `python scripts/build_tables.py --data data/paper --output runs/tables` reaggregates raw rows.

Do not replace `data/paper` casually. Save new simulations under `runs/`, compare them, and update the manuscript together with its tables, figures, provenance, and hashes.

## Save switches

Two global switches decide what a run leaves on disk. Experiments always write a complete output first, so checks and comparisons can read it; the switches are applied at the end.

| Switch | Keeps | Notebook | Command line | Environment |
|---|---|---|---|---|
| `SAVE_DATA` | CSV, NPZ, JSON, TXT results | first code cell | `--no-save-data` | `CVHNN_SAVE_DATA=0` |
| `SAVE_FIGURES` | PNG, PDF, SVG figures | first code cell | `--no-save-figures` | `CVHNN_SAVE_FIGURES=0` |

Both default to `True`. Run records (`execution.json`, `notebook_runtime.json`, `comparison.json`) are kept with whatever is saved. With both switches off, a notebook only shows its results inline and its output folder is deleted; `reproduce_all.py` keeps only `comparison.json`. The command-line flags apply to `reproduce_all.py` and to the three `experiments/` entrypoints. The rule is implemented once in `src/cvhnn/config.py` (`retain`); the notebooks carry an inline copy so they also run against the tagged source on Colab.

```bash
# Repeat the full study, keep only the verdict against the publication data:
python reproduce_all.py --mode full --output runs/check --no-save-data --no-save-figures
# Keep the trial data of one experiment but not its figures:
CVHNN_SAVE_FIGURES=0 python experiments/02_hsv_recall.py --mode full --output runs/recall
```

## Google Colab

1. Open a notebook using a badge in the [README](../README.md#run-in-google-colab).
2. Connect to a CPU runtime. No accelerator is required.
3. In the first code cell, keep `MODE = "full"` for the paper protocol or choose `"smoke"` for a short check, and set `SAVE_DATA` / `SAVE_FIGURES`.
4. Select **Runtime → Run all**. Setup downloads the public tagged source into a fresh directory and installs the requirements. No GitHub sign-in or token is required.
5. Read the results and checks. When something is saved, the final cell offers a ZIP of that run's outputs; the same file is available in the Files sidebar.

Notebook 00 renders the publication measurements and recomputes the selected HSV illustration. Notebooks 01–03 rerun their respective experiments. Notebook 03 plots newly measured timings, so its vertical values need not match the publication-host figure. The scientific decisions must agree.

Every run uses a new output directory and records configuration, package versions, and source revision. For local execution, open a notebook from an extracted repository, install `requirements.txt`, and select that environment's Python kernel. Local runs use the local checkout. The repository stores the notebooks without outputs; running them regenerates every table and plot.

Colab availability, package installation, and CPU speed vary. If the runtime disconnects, reconnect and rerun from the first cell. Do not compare partial or smoke results with full-paper estimates. Keep output ZIPs before ending a temporary session.

## What is stored and what is regenerated

Everything an experiment or notebook produces is regenerated by running it. The repository stores only the inputs and the frozen publication reference that new runs are compared against:

| Stored | Why it cannot simply be regenerated |
|---|---|
| `configs/ordered_alphabets.json` | Experimental input; fixes the symbol order that decides ties |
| `data/paper/experiment3_raw_timings.csv` | Timings depend on the machine; Figure 5 and the timing claims rest on this file |
| `data/paper/experiment1_final_runs.csv`, `experiment1_trajectories.csv.gz` | Reference for Table I and Figure 3 and for the comparison of new runs |
| `data/paper/experiment2_noise_results.csv`, `experiment2_load_results.csv`, `experiment2_weight_validation.csv` | Reference for Table II, the quoted residual, and the comparison of new runs |
| `data/paper/experiment3_configuration.json` | Provenance of the published benchmark; not rewritten by the current engine |
| `figures/*.pdf`, `figures/source_data/` | Paper figures (byte-identical to `paper/Figures`) and their plotting data |

Summary tables (`*_summary.csv`, `*_alphabet_summary.csv`, pairwise comparisons, visual-example metrics), the 32 benchmark query files, SVG figures, and notebook outputs are written by every run and are not stored. The SHA-256 hashes of the publication query files remain in `validation/query_manifest.json`; the v1.0.2 release keeps the files themselves.

## Paper-to-code map

| Paper item | Generator | Governing inputs |
|---|---|---|
| Figure 1: generic Voronoi diagram | `scripts/render_geometry.py` | 12 explicit sites in `figures/source_data/irregular_symbols.csv` |
| Figure 2: four alphabets in one row | `scripts/render_geometry.py` | Saturation alphabets in `configs/ordered_alphabets.json` |
| Figure 3: saturation trajectories | `scripts/render_figures.py` | `experiment1_trajectories.csv.gz`, 16-QAM/uniform |
| Figure 4: HSV image trial | `scripts/render_figures.py` | Selected seed 120044901; independent kernels |
| Figure 5: activation cardinality scaling | `scripts/render_cost.py` | `experiment3_raw_timings.csv`, all families and field sets |
| Table I: saturation | `scripts/build_tables.py` | `experiment1_final_runs.csv` |
| Table II: moderate corruption | `scripts/build_tables.py` | `experiment2_noise_results.csv`, level 0.10 |
| Table III: algorithm costs | Analytical table in `paper/main.tex` | Stated arithmetic model; not measured runtimes |

CSV input names refer to `data/paper` unless a different directory is given. Table bodies are generated in `tables/` and copied to `paper/Tables/`. Do not edit numerical values separately in LaTeX. The additional `recall_summary` plot shown in the README is not a numbered paper figure.

The generic diagram is an illustration, not an experimental alphabet. Its symbols have minimum separation approximately 0.420. All regular diagrams use the actual experimental coordinates. The [geometry checker](../scripts/verify_geometry.py) verifies cell inequalities, full window coverage, and shared faces. All figures use Seaborn/Matplotlib or Plotnine, a shared palette, vector PDF, and 600 dpi PNG; geometry is computed as clipped Voronoi half-plane intersections.
