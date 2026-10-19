# Validation and reproducibility

[Home](../README.md) · [Methods](METHODS_AUDIT.md) · [Paper map](PAPER_MAP.md)

## Publication evidence

The manuscript uses `data/paper`. Publication-host environment records and coordinate audits are in `validation/`. The 32 files in `data/queries` store all 640,000 benchmark field/current-symbol pairs. Hash manifests protect these inputs against silent modification.

The default numerical implementation is checked against direct field and energy recomputation in unit tests. Full runs audit every changed coordinate and every terminal state. The full protocol records 1,600 saturation trajectories and 2,466 recall trajectories (2,400 noise trials, 64 load trials, two default image trajectories). Their 1,587,062 changed coordinates have no reported energy increases or terminal-rule violations.

## Hosted measurements

`validation/hosted` contains a full scientific replication on a Google Colab CPU runtime, completed on 19 October 2026 UTC. The measurements, actual numerical environments, and coordinate audits are retained separately from the publication-host data. These records validate the experimental kernels and figure inputs; they are distinct from the newly packaged public-download notebook setup.

All 129,600 saturation history records agree exactly after CSV parsing. Recall outcomes, symbol errors, and stopping counts agree; the largest recall energy difference is below 4×10^-13. All 61,380 tested boundary decisions agree with exhaustive nearest-symbol search. The benchmark has 120 configurations and 600 timing records. Timing ratios across the two machines range from approximately 0.63 to 4.98; do not mix them into one publication curve.

The geometry rendering reproduces the five paper figures, including the selected HSV image. The selected image is a disclosed successful case, while tables include all trials. The small load sweep is exploratory. Noise replicates use a fixed synthetic memory bank and are not independent memory-bank samples.

## Public GitHub-to-Colab validation of v1.0.2

On 19 October 2026, all four public GitHub-linked notebooks of **v1.0.2** completed full execution directly in Google Colab, including the tagged clone and dependency-installation setup. The [run record](../validation/colab_v1.0.2/README.md) and [structured audit](../validation/colab_v1.0.2/validation.json) document the exact source commit and observed environment.

Saturation reproduced 1,600 final rows and 129,600 history rows exactly. The 2,400 noise-recall and 64 load-trial rows matched within 1e-9, with a maximum numerical difference below 4e-13. The benchmark produced 600 timing records, 100% activation agreement, and zero disagreements in 61,380 boundary queries.

All five paper figures were generated. Their six plotting-data CSVs matched after parsing, and regular Termes figure typography was retained. PNGs are not pixel-identical across rendering environments, and newly measured benchmark timings are machine-dependent. Notebook 00 renders publication data and recomputes the selected image; notebooks 01-03 rerun the experimental protocols.

The record preserves live audit outputs and package versions. Raw-output ZIP and executed-notebook downloads did not complete; earlier `validation/hosted` data are separate evidence. The v1.0.2 release assets and tag remain unchanged.

## Run the checks

The four distributed notebooks also completed a full local execution from top to bottom. Their saved outputs are actual execution results. [Notebook records](../validation/notebook_execution.json) identify the executed code cells, and the [numerical comparison](../validation/local_comparison.json) records zero differences in the compared saturation and recall metrics. All six geometry plotting-data files match their publication counterparts. These checks exercise the local setup path; the public Colab setup uses the tagged GitHub clone described in [Colab instructions](COLAB.md).

The manuscript compiled locally with pdfLaTeX and BibTeX to 14 pages without errors or layout warnings; see [Compilation record](../validation/compilation.json).

```bash
python -m unittest discover -s tests -v
python scripts/check_consistency.py
python reproduce_all.py --mode full --output runs/independent_full
```

The consistency check verifies hashes, table regeneration, figure equality, citation keys, notebook structure, geometry, hosted numerical records, and quoted manuscript values. A fresh full run compares scientific quantities at absolute tolerance 1e-9; timestamps and hardware-dependent timings are excluded. The benchmark requires exact agreement of all tested activation decisions.

The theorem concerns exact minimizers; the implementation has explicit floating-point tie tolerances. Experimental success is not a proof of universal recall. CI results indicate the tested software environments, not a guarantee for all future package versions or hosted runtimes.
