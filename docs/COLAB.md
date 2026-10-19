# Run the experiments in Google Colab

[Home](../README.md) · [Experiment catalog](EXPERIMENTS.md) · [Validation](VALIDATION.md)

1. Open a notebook using a badge in the [README](../README.md#run-in-google-colab).
2. Connect to a CPU runtime. No accelerator is required.
3. Keep `MODE = "full"` to use the paper protocol; choose `"smoke"` for a short check.
4. Select **Runtime → Run all**. Setup downloads the public `v1.0.0` source into a fresh directory and installs the requirements. No GitHub sign-in or token is required.
5. Read the results and checks. The final cell offers a ZIP of that run's outputs; the same file is available in the Files sidebar.

Notebook 00 renders the publication measurements and recomputes the selected HSV illustration. Notebooks 01–03 rerun their respective experiments. Notebook 03 plots newly measured timings, so its vertical values need not match the publication-host figure. The scientific decisions must agree.

Every run uses a new output directory and records configuration, package versions, and source revision. For local execution, open a notebook from an extracted repository, install `requirements.txt`, and select that environment's Python kernel. Local runs use the local checkout.

Colab availability, package installation, and CPU speed vary. If the runtime disconnects, reconnect and rerun from the first cell. Do not compare partial or smoke results with full-paper estimates. Keep output ZIPs before ending a temporary session.
