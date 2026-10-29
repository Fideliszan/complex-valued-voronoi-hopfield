# Usage

[Home](../README.md) · [Methods](METHODS.md) · [Validation](VALIDATION.md)

## Environment

Use Python 3.12 or 3.13 and run `python -m pip install -r requirements.txt` from the repository root. This installs `cvhnn` in editable mode with the dependencies declared in `pyproject.toml`.

Direct package versions from the reference run are recorded in `requirements-lock.txt`. To use those versions, run `python -m pip install -r requirements-lock.txt`, then `python -m pip install --no-deps -e .`.

Run-level paths, stages, smoke sizes, comparison tolerance, and save switches live in `src/cvhnn/config.py`. Protocol constants remain in each experiment module and are recorded in `execution.json` with the actual numerical environment. Launchers use one BLAS thread. Plotting uses a noninteractive backend and requires no LaTeX installation.

## Run the experiments

Run commands from the repository root. `--mode smoke` reduces repetition counts while retaining all alphabet families; `--mode full` uses the complete reference protocol. Launchers require a fresh `--output` directory and refuse to overwrite an existing run.

```bash
python reproduce_all.py --mode full --output runs/my_run
python scripts/compare_results.py runs/my_run --reference data/paper --output runs/my_run/comparison.json
```

`reproduce_all.py` already writes `comparison.json`; the second command repeats the comparison for an existing run. Scientific measurements and machine-dependent timings are assessed separately. Dense recall uses 576 neurons; query batches are limited to 4096 entries. Runtime and peak memory depend on the host.

Generation without rerunning all experiments:

```bash
python scripts/render_figures.py --data data/paper --output runs/figures
python scripts/render_geometry.py --output runs/geometry
python scripts/build_tables.py --data data/paper --output runs/tables
```

The first command renders reference measurements and reruns the disclosed illustrative image trial. The second generates only the two geometry diagrams. The third reaggregates raw rows. New results belong in `runs/`; changes to frozen references require deliberate scientific review and refreshed provenance and hashes.

## Save switches

Experiments first produce complete outputs so validation can read them. Retention switches are applied after checks and comparisons.

| Switch | Controls | Notebook | Command line | Environment |
|---|---|---|---|---|
| `SAVE_DATA` | CSV, NPZ, JSON, TXT results | first code cell | `--no-save-data` | `CVHNN_SAVE_DATA=0` |
| `SAVE_FIGURES` | PNG, PDF, SVG figures | first code cell | `--no-save-figures` | `CVHNN_SAVE_FIGURES=0` |

Both default to `True`. If either switch is enabled, execution, notebook-runtime, and comparison records remain with the retained results. With both switches off, a stage or notebook output directory is deleted after its checks; `reproduce_all.py` retains only its aggregate `comparison.json`. The flags apply to `reproduce_all.py` and the three `experiments/` entrypoints. Notebooks and launchers share the retention logic in `src/cvhnn/config.py`.

```bash
# Validate a full run without retaining data tables or figures:
python reproduce_all.py --mode full --output runs/check --no-save-data --no-save-figures
# Keep data but omit figure files:
python experiments/02_hsv_recall.py --mode full --output runs/recall --no-save-figures
```

## Google Colab

1. Open a notebook from the [README](../README.md#run-in-google-colab) and connect to a CPU runtime.
2. In the first code cell, choose `MODE = "full"` or `"smoke"`, and set `SAVE_DATA` and `SAVE_FIGURES`.
3. Select **Runtime → Run all**. Setup downloads the current public `main` source and installs its requirements; no GitHub token is required.
4. Read the results and checks, and save any output ZIP before ending the temporary session.

The badges and setup target `main` so they include the current code and save switches. Record the resolved commit for reproducibility; `main` changes over time. Historical Colab records remain tied to the exact version they tested.

Notebook 00 renders reference measurements and recomputes the selected HSV illustration. Notebooks 01–03 rerun their experiments. Notebook 03 plots newly measured timings, which need not match the reference host's timings; scientific decisions must agree. Each run records its source revision, configuration, and package versions.

For local execution, install the repository requirements and select that environment's Python kernel. Local notebooks use the local checkout. The repository stores notebooks without outputs; running them recreates tables and plots. Smoke or partial results are execution checks and must not be compared with complete protocol estimates.

## What is stored and what is regenerated

The repository stores small inputs and frozen references needed to evaluate future runs:

| Stored reference | Purpose |
|---|---|
| `configs/ordered_alphabets.json` | Experimental coordinates and symbol order for ties |
| `data/paper/experiment3_raw_timings.csv` | Measurements from the original benchmark machine |
| `data/paper/experiment1_final_runs.csv`, `experiment1_trajectories.csv.gz` | Saturation trial and trajectory baselines |
| `data/paper/experiment2_noise_results.csv`, `experiment2_load_results.csv`, `experiment2_weight_validation.csv` | Recall outcomes and weight-validation baselines |
| `data/paper/experiment3_configuration.json` | Provenance of the recorded benchmark |
| `figures/`, `figures/source_data/`, `tables/` | Selected reference images, plotting data, and expected numerical tables |
| `validation/` | Hashes, coordinate audits, environment records, and separate hosted measurements |

The name `data/paper/` identifies numerical reference data; it contains no manuscript text. Keeping a reference is distinct from storing every future run. Do not replace historical timings with measurements from another host.

Summary CSVs, benchmark query arrays, alternative figure exports, notebook execution outputs, and new run reports are regenerated and ignored by Git. The compressed trajectory CSV is read directly by pandas. `validation/query_manifest.json` preserves the hashes of the original query arrays; new runs also record their own arrays and hashes.

Manuscript sources and submission packages are excluded from the current source tree. Removing current files does not remove copies from older Git commits, tags, or release attachments.

## Result generators

| Result | Generator | Inputs |
|---|---|---|
| Generic Voronoi diagram | `scripts/render_geometry.py` | 12 explicit sites in `figures/source_data/irregular_symbols.csv` |
| Four alphabet diagrams | `scripts/render_geometry.py` | Saturation coordinates in `configs/ordered_alphabets.json` |
| Saturation trajectories | `scripts/render_figures.py` | `experiment1_trajectories.csv.gz`, 16-QAM/uniform |
| HSV image illustration | `scripts/render_figures.py` | Selected seed 120044901; independent kernels |
| Activation cardinality scaling | `scripts/render_cost.py` | `experiment3_raw_timings.csv`, all families and field sets |
| Saturation table | `scripts/build_tables.py` | `experiment1_final_runs.csv` |
| Moderate-corruption recall table | `scripts/build_tables.py` | `experiment2_noise_results.csv`, level 0.10 |

CSV names refer to `data/paper/` unless a different path is given. The table generator writes CSV, LaTeX table fragments, and a numerical JSON summary to its output directory; it does not need a manuscript checkout.

The generic diagram is an illustration, not an experimental alphabet. The regular diagrams use the actual experimental coordinates. The [geometry checker](../scripts/verify_geometry.py) verifies cell inequalities, coverage, and shared faces. Figures use the shared palette and typography, vector PDF, and 600 dpi PNG; the Voronoi cells are computed as clipped half-plane intersections.
