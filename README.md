# Voronoi-Based Activation Functions and Inertial Energy for Complex-Valued Hopfield Networks

**Fidelis Zanetti de Castro · Marcos Eduardo Valle · João P. Papa · Rodolfo Anibal Lobo**

[![Tests](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/actions/workflows/tests.yml/badge.svg)](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-2864A0)](docs/USAGE.md#environment)
[![Code MIT](https://img.shields.io/badge/Code-MIT-16877A)](LICENSE)
[![Data CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-8F6593)](LICENSE-DATA.md)

Computational companion for finite-alphabet complex-valued Hopfield networks: Python implementations, runnable notebooks, small experimental inputs, and frozen numerical references. The current source tree does not distribute the manuscript text or its submission packages.

[Methods and metrics](docs/METHODS.md) · [Usage](docs/USAGE.md) · [Validation](docs/VALIDATION.md)

## Overview

The classical activation maximizes the real inner product with the local field. The Voronoi activation chooses a nearest complex symbol and retains the current symbol on ties whenever possible. On multi-amplitude alphabets, the two rules can produce different magnitude behavior. The associated inertial energy adds half the squared state norm to the classical Hopfield energy.

![Four finite complex alphabets and their clipped Voronoi cells.](figures/constellations.png)

Three experiments study magnitude saturation, HSV associative recall, and exact nearest-symbol query cost. Convergence assumes Hermitian weights, nonnegative diagonal, and fair asynchronous updates; it does not guarantee recovery of a specified memory. See [methods and scope](docs/METHODS.md).

## Getting started

Use **Python 3.12 or 3.13**. A CPU is sufficient; synthetic images are generated locally and no external dataset is needed.

```bash
git clone https://github.com/Fideliszan/complex-valued-voronoi-hopfield.git
cd complex-valued-voronoi-hopfield
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/check_consistency.py
python examples/quick_activation.py
```

Installation includes the editable `cvhnn` package. For the recorded dependency versions, see [Environment](docs/USAGE.md#environment).

## Usage

Run commands from the repository root and choose a new output directory.

| Task | Command |
|---|---|
| Short saturation run | `python experiments/01_saturation.py --mode smoke --output runs/small_saturation` |
| Full saturation study | `python experiments/01_saturation.py --mode full --output runs/full/saturation` |
| Full HSV recall | `python experiments/02_hsv_recall.py --mode full --output runs/full/recall` |
| Full activation benchmark | `python experiments/03_activation_cost.py --mode full --output runs/full/activation_cost` |
| Regenerate reference figures | `python scripts/render_figures.py --data data/paper --output runs/figures` |
| Recompute reference tables | `python scripts/build_tables.py --data data/paper --output runs/tables` |
| Reproduce all experiments | `python reproduce_all.py --mode full --output runs/reproduction` |

Use `--no-save-data` and/or `--no-save-figures` to control retained outputs. The equivalent notebook switches are `SAVE_DATA` and `SAVE_FIGURES`; both default to `True`. Environment settings are `CVHNN_SAVE_DATA=0` and `CVHNN_SAVE_FIGURES=0`. [Output retention and examples](docs/USAGE.md#save-switches).

### Run in Google Colab

| Notebook | Task | Launch |
|---|---|---|
| Geometry | Render figures from the reference measurements | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/main/colab/00_geometry.ipynb) |
| Saturation | Run the paired energy and magnitude study | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/main/colab/01_saturation.ipynb) |
| HSV recall | Run recall under symbol and HSV corruption | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/main/colab/02_hsv_recall.ipynb) |
| Activation cost | Benchmark exact nearest-symbol methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/main/colab/03_activation_cost.ipynb) |

These links use the current `main` branch. Set `MODE` and the save switches in the first code cell, then choose **Runtime → Run all**. Runs record the actual source revision. Notebooks are committed without execution outputs. [Colab instructions](docs/USAGE.md#google-colab).

## Repository contents

| Path | Purpose |
|---|---|
| `src/cvhnn/` | Numerical algorithms, search methods, configuration, and experiment runner |
| `experiments/`, `colab/`, `examples/` | Command-line entrypoints, notebooks, and a minimal example |
| `configs/` | Ordered alphabet coordinates; array order determines tie resolution |
| `data/paper/` | Frozen trial-level reference measurements, including machine-specific timings |
| `figures/`, `tables/` | Selected reference figures, plotting data, and expected numerical tables |
| `validation/` | Hash manifests, coordinate audits, and separately recorded hosted measurements |
| `scripts/`, `tests/`, `docs/` | Generation tools, scientific checks, and documentation |
| `runs/` | New execution outputs; ignored by Git |

Regenerable summaries, query arrays, and notebook outputs are produced by code instead of being committed. Frozen reference measurements are retained so a new execution can be compared with the recorded result. Benchmark timings belong to their original machine and should not be replaced by another machine's timings. [Data policy and generators](docs/USAGE.md#what-is-stored-and-what-is-regenerated).

## Reproducibility

`check_consistency.py` validates frozen hashes, regenerated tables, reference numerical values, notebook structure, geometry, and hosted numerical records without requiring manuscript sources. Unit tests compare independent numerical kernels against direct field and energy recomputation. A full run compares scientific quantities with the reference at absolute tolerance `1e-9`; timings are assessed separately because they depend on the host. [Evidence and limitations](docs/VALIDATION.md).

## Citation and license

Use [CITATION.cff](CITATION.cff) or GitHub's **Cite this repository** button, and record the exact commit and execution configuration. Historical validation records identify the version they tested; they do not certify a later checkout.

Code and documentation: [MIT](LICENSE). Synthetic data, numerical tables, and figures: [CC BY 4.0](LICENSE-DATA.md). Figure fonts retain their [GUST Font License](assets/fonts/GUST-FONT-LICENSE.txt); see the [font derivation note](assets/fonts/DERIVATION.txt). The manuscript remains outside these license grants.
