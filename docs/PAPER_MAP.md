# Paper-to-code map

[Home](../README.md) · [Gallery](GALLERY.md) · [Validation](VALIDATION.md)

| Paper item | Generator | Governing inputs |
|---|---|---|
| Figure 1: generic Voronoi diagram | `scripts/render_geometry.py` | 12 explicit sites in `figures/source_data/irregular_symbols.csv` |
| Figure 2: four alphabets in one row | `scripts/render_geometry.py` | Saturation alphabets in `configs/ordered_alphabets.json` |
| Figure 3: saturation trajectories | `scripts/render_figures.py` | `experiment1_trajectories.csv`, 16-QAM/uniform |
| Figure 4: HSV image trial | `scripts/render_figures.py` | Selected seed 120044901; independent kernels |
| Figure 5: activation cardinality scaling | `scripts/render_cost.py` | `experiment3_raw_timings.csv`, all families and field sets |
| Table I: saturation | `scripts/build_tables.py` | `experiment1_final_runs.csv` |
| Table II: moderate corruption | `scripts/build_tables.py` | `experiment2_noise_results.csv`, level 0.10 |
| Table III: algorithm costs | Analytical table in `paper/main.tex` | Stated arithmetic model; not measured runtimes |

CSV input names refer to `data/paper` unless a different directory is given. Table bodies are generated in `tables/` and copied to `paper/Tables/`. Do not edit numerical values separately in LaTeX. The additional `recall_summary` gallery plot is not a numbered paper figure.

The generic diagram is an illustration, not an experimental alphabet. Its symbols have minimum separation approximately 0.420. All regular diagrams use the actual experimental coordinates. The [geometry checker](../scripts/verify_geometry.py) verifies cell inequalities, full window coverage, and shared faces.
