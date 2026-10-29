# Contributing

Keep code, experimental inputs, and numerical references traceable. Open an issue describing the problem and a minimal example. Make a focused change on a branch, run `python -m unittest discover -s tests -v` and `python scripts/check_consistency.py`, then submit a pull request.

CI enforces [Ruff](https://docs.astral.sh/ruff/): install `ruff==0.17.1`, run `ruff check .`, then `ruff format .`. Run-level settings belong in `src/cvhnn/config.py`; experimental constants stay with their protocol and are recorded in `execution.json`. Shared figure colors and typography live in `src/cvhnn/style.py`.

Store new results under `runs/`, leave notebook outputs cleared, and avoid committing files the code regenerates. Keep frozen reference measurements, especially machine-specific timings, unless an intentional scientific update reviews their configuration, seeds, environment, comparisons, and hashes. After changing notebook code, run `python scripts/execute_notebooks.py --record validation/notebook_execution.json` to refresh the full-execution record. Keep committed notebook outputs and execution counts cleared.

The public source excludes manuscript text and submission packages. Keep those outside the tracked tree. Do not commit credentials, private paths, notebook tokens, or unlicensed datasets. Discuss protocol changes before presenting their output as reproduction of the reference experiment.
