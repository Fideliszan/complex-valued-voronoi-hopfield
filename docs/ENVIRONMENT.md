# Environment

Use Python 3.12 or 3.13 and install `requirements.txt`. Direct package versions from the publication run are recorded in `requirements-lock.txt`; each experiment additionally writes `execution.json` with the actual Python, OS, NumPy, pandas, and SciPy versions.

One BLAS thread is used by the launchers. Timing values depend on CPU, operating system, numerical libraries, and background load. Matplotlib uses a noninteractive backend for exported figures. No LaTeX installation is required for simulation or plotting; manuscript compilation uses pdfLaTeX/BibTeX-compatible sources and the IEEEtran class.

Published alphabet coordinates are loaded from JSON so equal-radius ordering does not depend on platform-specific evaluations of complex exponentials and absolute values. The original constructors remain available for inspection and custom, non-publication experiments.

The full hosted Google Colab measurements and their actual package versions are documented in [Validation](VALIDATION.md). Publication-host and Colab timing measurements are stored separately.
