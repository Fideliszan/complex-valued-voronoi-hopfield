# Voronoi-Based Activation Functions and Inertial Energy for Complex-Valued Hopfield Networks

**Fidelis Zanetti de Castro · Marcos Eduardo Valle · João P. Papa · Rodolfo Anibal Lobo**

[![Tests](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/actions/workflows/tests.yml/badge.svg)](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-2864A0)](docs/ENVIRONMENT.md)
[![Code MIT](https://img.shields.io/badge/Code-MIT-16877A)](LICENSE)
[![Data CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-8F6593)](LICENSE-DATA.md)

[Overview](#overview) · [Getting Started](#getting-started) · [Usage](#usage) · [Metrics](#metrics) · [Project Structure](#project-structure) · [Citation](#citation) · [License](#license)

[Read the manuscript](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/releases/download/v1.0.2/Complex_Voronoi_Manuscript_1.0.2.pdf) · [Download the validated Overleaf ZIP](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/releases/download/v1.0.2/Complex_Voronoi_Overleaf_1.0.2_Colab_Validated_6f3dbfa.zip) · [Browse the documentation](docs/README.md)

**For Overleaf, download `Complex_Voronoi_Overleaf_1.0.2_Colab_Validated_6f3dbfa.zip` using the link above.** This recommended package contains the manuscript, figures, complete computational companion, and records of the direct Colab validation of all four v1.0.2 notebooks. In Overleaf, choose **New project → Existing project (.zip)**, upload this ZIP, and compile `main.tex`. The original ZIP remains in the release for archival purposes; the recommended package adds the validation documentation without changing the manuscript or figures.

> **Research companion.** Reproduce the paper from source, inspect the trial-level results, or launch one of the four CPU notebooks below. No external dataset or GitHub credentials are needed.

**Validated in Overleaf (19 October 2026):** the recommended ZIP compiled to 14 pages with **0 errors and 0 warnings**. All five figures were preserved; all 14 rendered pages match the previous validated PDF. [Open the validated project](https://www.overleaf.com/project/6ad60897f64e207c3401375d) · [Verification record](validation/overleaf_colab_validated_v1.0.2/README.md).

## Overview

An associative memory stores patterns and attempts to recover them from corrupted cues. In this work each neuron takes a symbol from a finite subset of the complex plane, encoding both phase and magnitude. The question is how the local update should respect that geometry while retaining an energy descent guarantee.

The classical rule maximizes the real inner product with the local field. On a multi-amplitude alphabet, this can favor large symbols even when a smaller symbol is nearer. The proposed activation selects the Euclidean nearest symbol and retains the current symbol on ties whenever possible. Its associated inertial energy adds half the squared state norm to the classical Hopfield energy.

![Four finite complex alphabets and their clipped Voronoi cells.](figures/constellations.png)

*PSK, QAM, polar, and hexagonal alphabets. Filled points are alphabet symbols; hollow rings identify maximum-modulus symbols. Thin boundaries and muted fills distinguish adjacent Voronoi cells; the dotted circle shows the maximum radius. [Vector PDF](figures/constellations.pdf) · [Geometric interpretation](docs/OVERVIEW.md).*

### Contributions and limits

- Identify the magnitude saturation bias of classical inner-product updates on multi-amplitude alphabets.
- Formulate a finite-alphabet Voronoi activation with conservative deterministic tie handling, and establish finite convergence under Hermitian weights with nonnegative diagonal and a fair asynchronous schedule.
- Test the mechanism, HSV recall, and alternative exact nearest-symbol implementations in three separate experiments.

Convergence does not guarantee recovery of the intended memory. The term “inertial” refers to the quadratic state penalty; the model is not a momentum-based second-order dynamics. The experiments use synthetic patterns and do not establish universal superiority over other associative memories. [Method and scope](docs/OVERVIEW.md).

## Explore the evidence

| I want to… | Start here |
|---|---|
| Understand the equations and assumptions | [Overview](docs/OVERVIEW.md) |
| Browse figures and their source data | [Gallery](docs/GALLERY.md) |
| Find the script behind a paper figure or table | [Paper map](docs/PAPER_MAP.md) |
| Run one experiment or all experiments | [Usage](docs/USAGE.md), [experiment catalog](docs/EXPERIMENTS.md) |
| Interpret errors, energies, and runtime | [Metrics](docs/METRICS.md) |
| Check numerical reproducibility | [Theory-to-code audit](docs/METHODS_AUDIT.md), [validation](docs/VALIDATION.md) |

![Mean symbol error rate under two corruption channels for four alphabets.](figures/recall_summary.png)

*Means include all trials. Squares denote classical updates; circles denote Voronoi updates. Nominal HSV and symbol-noise levels have different meanings and must not be treated as equal input error. [Trial data](data/paper/experiment2_noise_results.csv).*

## Cardinality and computational cost

![Activation costs for four alphabet cardinalities and two input distributions.](figures/activation_scaling.png)

*Each column is an alphabet family; rows separate uniform and network-generated
fields. Markers show the median of five timing repetitions and bars their range.
All methods share conservative tie handling. The four cardinalities are
16, 64, 256, and 1024; N=576 remains fixed. [Vector PDF](figures/activation_scaling.pdf)
· [Protocol and implementation audit](docs/METHODS_AUDIT.md).*

## Getting Started

**Prerequisites:** Python 3.12 or 3.13. A CPU is sufficient. The data are synthetic; no external image dataset is required.

```bash
git clone https://github.com/Fideliszan/complex-valued-voronoi-hopfield.git
cd complex-valued-voronoi-hopfield
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Alternatively, invoke the environment's Python directly.

```bash
python -m pip install -r requirements.txt
python scripts/check_consistency.py
python examples/quick_activation.py
python -m unittest discover -s tests -v
```

The small example shows a local field of 0.4 selecting radius 1 under the classical score and radius 0.3 under Euclidean projection. It also verifies the energy correction identity.

## Usage

Run commands from the repository root. Choose a new output directory; existing runs are never deleted automatically.

| Task | Command |
|---|---|
| Short saturation run | `python experiments/01_saturation.py --mode smoke --output runs/small_saturation` |
| Full saturation study | `python experiments/01_saturation.py --mode full --output runs/full/saturation` |
| Full HSV recall | `python experiments/02_hsv_recall.py --mode full --output runs/full/recall` |
| Full query benchmark | `python experiments/03_activation_cost.py --mode full --output runs/full/activation_cost` |
| Rebuild paper artwork | `python scripts/render_figures.py --data data/paper --output figures` |
| Recompute paper tables | `python scripts/build_tables.py --data data/paper` |
| Run the entire battery | `python reproduce_all.py --mode full --output runs/reproduction` |

### Run in Google Colab

| Notebook | What it does | Launch |
|---|---|---|
| Geometry and paper figures | Rebuilds all five figures from the publication data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/v1.0.2/colab/00_geometry.ipynb) |
| Saturation | Runs the paired energy and magnitude study | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/v1.0.2/colab/01_saturation.ipynb) |
| HSV recall | Runs recall under symbol and HSV corruption | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/v1.0.2/colab/02_hsv_recall.ipynb) |
| Activation cost | Benchmarks four alphabet cardinalities with exact decision checks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fideliszan/complex-valued-voronoi-hopfield/blob/v1.0.2/colab/03_activation_cost.ipynb) |

**Validated directly in Colab (19 October 2026):** all four v1.0.2 notebooks completed full execution from these GitHub links. Scientific results reproduced within 1e-9; all five paper figures were regenerated with matching plotting data and regular Termes typography. PNG rendering and hardware-dependent timings can differ. [Run record and evidence limits](validation/colab_v1.0.2/README.md).

Select **Runtime → Run all**. Each notebook downloads the tagged public source, defaults to `MODE = "full"`, records its environment, and exports its results. Use `"smoke"` for a short execution check. See [Colab instructions](docs/COLAB.md).

### What the data show

At the reported settings, all 1,600 saturation trajectories and all 2,400 main recall trajectories reached a fixed point. In the QAM/uniform saturation condition, Voronoi dynamics ended at mean normalized squared norm 0.598 and maximum-modulus occupation 0.299. For 64-QAM with HSV noise level 0.10, Voronoi dynamics recovered the exact target in 27 of 30 trials. These are condition-specific observations, not universal recovery guarantees.

The companion includes a separate full Google Colab replication of the numerical protocols. Scientific outputs agree with the paper to numerical tolerance; runtime measurements are retained separately because machines differ. [Validation and scope](docs/VALIDATION.md).

## Metrics

| Quantity | Definition and interpretation |
|---|---|
| Classical energy | `E_c = -Re(s* W s)/2`; descent diagnostic for the classical rule |
| Inertial energy | `E_r = E_c + ||s||²/2`; descent diagnostic for Voronoi dynamics |
| Normalized squared norm | `rho = ||s||² / (N Rmax²)` |
| Boundary occupation | Fraction of symbols with modulus `Rmax` |
| Symbol error rate (SER) | Fraction of neurons differing from the intended target |
| Exact recovery | All target symbols correct; averaged across trials |
| Radial error | Mean absolute modulus error, normalized by `Rmax` |
| Query time | Measured microseconds per query; environment dependent |

[Full definitions and interpretation](docs/METRICS.md).

## Project Structure

```text
├── src/cvhnn/          # Numerical algorithms, search methods, and experiment protocols
├── experiments/       # One Python entrypoint per experiment
├── colab/             # Four runnable notebooks
├── configs/           # Explicit ordered alphabet coordinates
├── data/paper/        # Trial-level publication measurements
├── data/queries/      # All benchmark field/current-symbol pairs
├── figures/           # Vector PDF/SVG, 600 dpi PNG, and source data
├── tables/            # Generated CSV and LaTeX tables
├── paper/             # Manuscript, bibliography, and paper assets
├── validation/        # Numerical audits and separate hosted measurements
├── scripts/           # Rendering, checking, notebook, and packaging tools
├── tests/             # Mathematical unit tests
├── examples/          # Minimal working example
├── docs/              # Methods, usage, metrics, and figure gallery
└── .github/           # Continuous integration and contribution templates
```

## Reproducibility

The paper configuration runs 1,600 saturation trajectories, 2,400 noise-recall trajectories, and 64 exploratory load trajectories, plus a separate two-rule illustration. The query benchmark evaluates 120 method/input/alphabet combinations with five repetitions. Every run records its configuration and numerical environment.

The default engine independently implements the activation rules, weight matrices, asynchronous updates, corruptions, and metrics. It records an energy check at every changed coordinate. Unit tests compare small trajectories with full field/energy recomputation; stored full-run audits check energy descent and terminal decisions.

Explicit ordered coordinates in `configs/ordered_alphabets.json` prevent platform-dependent ordering of equal-radius symbols. Conservative tie handling retains the current state when it belongs to the numerical minimizing set. `data/queries` contains all benchmark field/current-symbol pairs. The selected HSV illustration uses the disclosed successful trial with seed 120044901; aggregate tables include all trials.

`scripts/check_consistency.py` verifies dataset hashes, generated tables, five manuscript figures, citation keys, notebook structures, geometry, hosted measurements, and quoted manuscript values. The CI matrix runs these checks on Linux and Windows with Python 3.12 and 3.13. [Complete protocol](docs/EXPERIMENTS.md) · [Numerical conventions](docs/METHODS_AUDIT.md).

## Citation

Use [CITATION.cff](CITATION.cff) or GitHub's **Cite this repository** button. Cite version **v1.0.2** and record the exact commit and execution configuration. The paper explicitly cites this public companion.

## License

Code and repository documentation: [MIT](LICENSE). Original synthetic data, generated tables, and figures: [CC BY 4.0](LICENSE-DATA.md). The manuscript in `paper/` is excluded from those grants and remains copyrighted by its authors. Third-party packages and publisher classes retain their own licenses.

Figure fonts are distributed under their own [GUST Font License](assets/fonts/GUST-FONT-LICENSE.txt); see the [font derivation note](assets/fonts/DERIVATION.txt).
