# Usage

Install `requirements.txt` and run commands from the project root. `--mode smoke` reduces repetition counts while retaining all alphabet families; `--mode full` uses the paper protocol. Each launcher requires `--output` and refuses to overwrite an existing directory.

```bash
python reproduce_all.py --mode full --output runs/my_run
python scripts/compare_results.py runs/my_run --reference data/paper --output runs/my_run/comparison.json
```

The comparison tests scientific measurements separately from hardware-dependent timings. Inspect discrepancies; a benchmark timing change alone is not a scientific failure. A full run uses one BLAS thread. It needs a few minutes on the authoring machine; runtime and peak memory depend on the host. Dense recall uses 576 neurons; query batches are limited to 4096 entries.

To regenerate only artwork, use `python scripts/render_figures.py --data data/paper --output figures`. This does not rerun the Monte Carlo experiments; it does rerun the documented illustrative image trial. `python scripts/build_tables.py --data data/paper` reaggregates raw rows.

Do not replace `data/paper` casually. Save new simulations under `runs/`, compare them, and update the manuscript together with its tables, figures, provenance, and hashes.
