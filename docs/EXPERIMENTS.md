# Experiment catalog

[Home](../README.md) · [Methods](METHODS_AUDIT.md) · [Metrics](METRICS.md)

| Stage | Python entrypoint | Notebook | Full protocol |
|---|---|---|---|
| Geometry and figures | `scripts/render_figures.py` | [00_geometry](../colab/00_geometry.ipynb) | Five paper figures and a recall overview |
| Saturation | `experiments/01_saturation.py` | [01_saturation](../colab/01_saturation.ipynb) | N=100; 4 alphabets; 2 initializations; 100 paired runs; 80-sweep cap |
| HSV recall | `experiments/02_hsv_recall.py` | [02_hsv_recall](../colab/02_hsv_recall.ipynb) | 24×24 images; P=4; 4 alphabets; 2 channels; 5 noise levels; 30 paired trials |
| Activation cost | `experiments/03_activation_cost.py` | [03_activation_cost](../colab/03_activation_cost.ipynb) | 4 families; cardinalities 16, 64, 256, 1024; 2 field sets; 20,000 queries; 5 repeats |

The noise study contains 2,400 trajectories. The additional load sweep contains 64 trajectories with only two trials per condition and is exploratory. The selected successful HSV illustration uses seed 120044901; it does not estimate average retrieval performance.

The saturation master seed is 20260831. The synthetic image bank uses seed 20260901. Ordered symbol coordinates are supplied in `configs/ordered_alphabets.json`; per-trial seeds are present in the CSV measurements. The full technical protocol is in [Methods](METHODS_AUDIT.md).

The default `independent` engine supplies the paper's numerical kernels. The optional `reference` engine exposes a second implementation for algorithm inspection; it is not the source of the publication benchmark, whose search methods share conservative tie handling. Use the default for reproduction.
