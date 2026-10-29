# Research data

`paper/` contains frozen trial-level numerical references for comparison with new runs. It contains no manuscript text. Synthetic images are generated procedurally by `cvhnn.recall.generate_synthetic_hsv_memories`; no external dataset is required.

The recorded measurements were produced by the experimental code. Summary tables are reaggregated from these rows; the 640,000 benchmark queries are regenerated from fixed seeds. `experiment1_trajectories.csv.gz` is a compressed CSV read directly by pandas. Benchmark timings are historical measurements of a specific machine and must be retained as recorded. See [the storage policy](../docs/USAGE.md#what-is-stored-and-what-is-regenerated).

CSV headers and [metric definitions](../docs/METHODS.md#metrics) define the data schema. `ensemble` in benchmark CSVs means the field set (`uniform` or `local_fields`). Missing analytical quantities remain missing rather than being treated as zero.

Data and derived figures use [CC BY 4.0](../LICENSE-DATA.md). Preserve seeds, configurations, environment records, and source revision with derived results.
