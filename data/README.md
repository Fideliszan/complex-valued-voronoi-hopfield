# Research data

- `paper/`: the frozen trial-level measurements used by the manuscript. New runs are compared against them.
- Synthetic images are generated procedurally by `cvhnn.recall.generate_synthetic_hsv_memories`; no external dataset is required.

Everything in this folder was produced by `reproduce_all.py`. Only the trial-level tables are stored: the summary tables are recomputed by every run, and the 640,000 benchmark queries are regenerated from fixed seeds (their publication hashes are in `validation/query_manifest.json`). `experiment1_trajectories.csv.gz` is a compressed CSV that pandas reads directly. `experiment3_raw_timings.csv` cannot be regenerated, because timings belong to their recorded machine. See [what is stored and what is regenerated](../docs/USAGE.md#what-is-stored-and-what-is-regenerated).

The governing schemas are defined by the CSV headers and [Metrics](../docs/METHODS.md#metrics). `ensemble` in benchmark CSVs is a stable field-set key (`uniform` or `local_fields`), not a machine-learning ensemble. Missing analytical quantities are preserved as missing rather than treated as zero.

The data and generated figures are licensed under [CC BY 4.0](../LICENSE-DATA.md). Preserve seeds, configurations, and environment records when sharing derived results.
