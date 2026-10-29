# Direct Colab validation of v1.0.2

On 19 October 2026, all four notebooks were opened from the public GitHub Colab links and executed from top to bottom in `MODE = "full"` on Google Colab CPU runtimes. They cloned tag `v1.0.2`, commit `c3fc72a05ac64c7af029cac7bbe927ca883c03f0`, and installed `requirements.txt` without GitHub credentials. All four completed successfully.

[Structured audit and environment records](validation.json) contain the notebook URLs, package versions, comparisons, and figure font names.

| Notebook | Verified result |
|---|---|
| `00_geometry.ipynb` | All five paper figures generated and visually checked. All six plotting-data CSVs agree after parsing: zero numeric differences, equal shapes and nonnumeric values. PDF figure text uses TermesFigure-Regular, with STIX for mathematical symbols. |
| `01_saturation.ipynb` | 1,600 final-run rows and 129,600 trajectory rows match exactly, including trial keys. |
| `02_hsv_recall.ipynb` | 2,400 noise-trial rows and 64 exploratory load rows match at absolute tolerance 1e-9. Maximum numeric differences are 3.979039320256561e-13 and 2.8421709430404007e-13, respectively. |
| `03_activation_cost.ipynb` | 600 timing records; minimum activation agreement 1.0; zero disagreements over 61,380 boundary queries. Runtime measurements are machine-dependent. |

## Figure equivalence

The paper's figure content and scientific data were reproduced, but the PNG files are **not pixel-identical** to the publication-host images. Three images have slightly different raster dimensions; rendering packages differ between hosts. All six plotting-data CSVs are identical after parsing even where byte-level CSV comparisons differ. Regular Termes typography was preserved, without bold panel titles. The JSON distinguishes `source_data_byte_identical` from parsed-data comparisons under `figure_data_and_fonts`.

Notebook 00 regenerates figures using publication data and recomputes the disclosed HSV illustration. Notebooks 01-03 independently rerun the experimental protocols. Notebook 03 plots newly measured timings, which are not expected to equal the paper's timing values.

## Evidence and limits

Additional comparison cells were appended to the live Colab sessions after the distributed notebooks completed. They compared newly generated outputs against the tagged repository's publication data. The published v1.0.2 notebooks were not edited. Package versions and experiment durations in the JSON come from those live outputs; a null duration for geometry means no experiment duration was printed.

The browser did not complete downloads of the generated raw-output ZIPs or executed notebooks. Therefore this directory preserves the observed audit results and environment records, not a fresh raw-data archive or executed `.ipynb` files. Earlier raw hosted measurements in `../hosted/` belong to a separate validation and are not represented as downloads from this run. Browser UI exports and account information are excluded.

This historical record describes the tested v1.0.2 revision; it does not certify later source revisions.
