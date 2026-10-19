# Research data

- `paper/`: trial-level and aggregate measurements used by the manuscript.
- `queries/`: all benchmark fields, current indices, and ordered alphabets in compressed NumPy files.
- Synthetic images are generated procedurally by `cvhnn.recall.generate_synthetic_hsv_memories`; no external dataset is required.

The governing schemas are defined by the CSV headers and [Metrics](../docs/METRICS.md). `ensemble` in benchmark CSVs is a stable field-set key (`uniform` or `local_fields`), not a machine-learning ensemble. Missing analytical quantities are preserved as missing rather than treated as zero. Timing data belong to their recorded machine.

The data and generated figures are licensed under [CC BY 4.0](../LICENSE-DATA.md). Preserve seeds, configurations, and environment records when sharing derived results.
