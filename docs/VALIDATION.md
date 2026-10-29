# Validation and reproducibility

[Home](../README.md) · [Methods](METHODS.md) · [Usage](USAGE.md)

## Frozen reference evidence

`data/paper/` contains the trial-level numerical baseline. Reference-host configuration and environment records are consolidated in `validation/publication_run.json`; coordinate audits and hash manifests are stored alongside it. The 640,000 benchmark field/current-symbol pairs are regenerated from fixed seeds. `validation/query_manifest.json` records the original array hashes.

The default independent implementation is tested against direct field and energy recomputation. Full runs audit every changed coordinate and every terminal state. The reference protocol records 1,600 saturation trajectories and 2,466 recall trajectories: 2,400 noise trials, 64 load trials, and two image trajectories. Their recorded 1,587,062 coordinate changes have no reported energy increases or terminal-rule violations.

## Hosted measurements

`validation/hosted/` preserves a separate full Google Colab CPU replication completed on 19 October 2026 UTC, with trial data, environments, timings, coordinate audits, and comparison records. These are historical measurements, not a claim that the current checkout has just been run on Colab.

All 129,600 saturation history records agreed exactly after CSV parsing. The identical duplicate history is represented by its comparison record and the shared compressed reference CSV. Recall outcomes, symbol errors, and stopping counts agreed; the largest recorded recall energy difference was below 4×10^-13. All 61,380 tested boundary decisions agreed with exhaustive nearest-symbol search. The benchmark contains 120 configurations and 600 timing records. Machine-dependent timing ratios ranged approximately from 0.63 to 4.98; the two hosts' timings must not be mixed into one curve.

## Historical public Colab validation

All four public GitHub-linked **v1.0.2** notebooks completed full execution directly in Colab on 19 October 2026, including their tagged download and dependency installation. The [run record](../validation/colab_v1.0.2/README.md) and [structured audit](../validation/colab_v1.0.2/validation.json) identify the exact commit, environment, and evidence limits.

That run reproduced 1,600 final saturation rows and 129,600 history rows exactly. The 2,400 noise-recall and 64 load rows matched within `1e-9`. All tested activation decisions agreed. Figure plotting-data CSVs matched after parsing, while PNG rendering and benchmark timings varied by environment. Raw-output ZIP and executed-notebook downloads did not complete; the older `validation/hosted/` measurements are separate evidence.

Current README badges use `main`. Historical validation records apply only to their recorded revisions; a change to notebook code requires a new execution record.

## Run the checks

```bash
python -m unittest discover -s tests -v
python scripts/check_consistency.py
python reproduce_all.py --mode full --output runs/independent_full
```

The consistency check validates frozen hashes, reaggregated reference tables and numerical values, notebook structure and execution records, geometry, and preserved hosted measurements. These checks do not require manuscript text, a bibliography, a LaTeX installation, or manuscript compilation records.

Run `python scripts/execute_notebooks.py` to execute all four notebooks in full mode and write a new record under `runs/`. After an intentional notebook update, refresh the checked-in evidence with `python scripts/execute_notebooks.py --record validation/notebook_execution.json`. Committed notebook outputs and execution counts remain cleared.

`validation/notebook_execution.json` records full local notebook execution and code-cell hashes. It exercises the local setup path; it does not by itself certify a fresh Colab download/install session. The historical local comparison in `validation/publication_run.json` remains tied to the original execution.

A fresh full run compares scientific quantities at absolute tolerance `1e-9`; timestamps and hardware-dependent timings are excluded from numerical equality. Activation benchmarks require exact agreement of tested decisions. Inspect the new run's `comparison.json` and environment records.

## Limits

The convergence result concerns exact minimizers; code uses documented floating-point tie tolerances. Finite test cases and successful retrieval experiments do not prove universal recall. The selected HSV image is a disclosed successful trial, while aggregate tables include all trials. The small load sweep is exploratory, and noise replicates reuse a fixed synthetic memory bank.

CI checks cover the declared operating-system and Python matrix. They do not guarantee compatibility with every future package version or hosted runtime. Frozen timing measurements and historical validation records are preserved as evidence rather than silently regenerated.
