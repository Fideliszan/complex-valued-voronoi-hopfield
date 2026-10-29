# Contributing

Keep code, raw results, figures, and manuscript claims traceable. Open an issue describing the problem and a minimal reproducible example. Create a branch, make a focused change, run `python -m unittest discover -s tests -v` and `python scripts/check_consistency.py`, and submit a pull request.

Code style is enforced in CI with [Ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`): run `python -m pip install ruff==0.17.1`, then `ruff check .` and `ruff format .` before committing. Run-level settings (paths, stages, smoke sizes, comparison tolerance, save switches) live in `src/cvhnn/config.py`; protocol constants stay at the top of their experiment module, because each run records them in `execution.json`. Shared figure colors and typography live in `src/cvhnn/style.py`.

Never overwrite archived data. New numerical results need explicit configuration, environment, seeds, comparison reports, and review of affected tables and text. Do not commit credentials, private paths, notebook tokens, or unlicensed datasets. Discuss changes to the scientific protocol before presenting them as the paper reproduction.
