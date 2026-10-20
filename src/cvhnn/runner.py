"""Execute the paper protocol with independently implemented numerical kernels."""

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

from cvhnn import config


def main(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument("stage", choices=config.STAGES)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["independent", "reference"], default="independent")
    parser.add_argument(
        "--no-save-data",
        dest="save_data",
        action="store_false",
        default=config.SAVE_DATA,
        help="Discard CSV/NPZ/JSON results after the run (also CVHNN_SAVE_DATA=0).",
    )
    parser.add_argument(
        "--no-save-figures",
        dest="save_figures",
        action="store_false",
        default=config.SAVE_FIGURES,
        help="Discard PNG/PDF figures after the run (also CVHNN_SAVE_FIGURES=0).",
    )
    args = parser.parse_args()
    stage = stage or args.stage
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    m = importlib.import_module("cvhnn." + stage)
    m.OUTPUT_DIR = out
    audits = None
    if args.mode == "smoke":
        for name, value in config.SMOKE_OVERRIDES[stage].items():
            setattr(m, name, value)
    if args.engine == "independent" and stage != "activation_cost":
        from cvhnn.independent_adapter import install

        audits = install(m, stage)
    if args.engine == "independent" and stage == "activation_cost":
        from cvhnn.independent_benchmark import install

        install(m)
    # A single BLAS thread avoids oversubscribing the many small matrix operations.
    from threadpoolctl import threadpool_limits

    start = time.perf_counter()
    with threadpool_limits(limits=1):
        if stage == "saturation":
            m.run_experiment()
        else:
            m.MAKE_FIGURES = stage == "recall"
            m.main()
    configuration = {k: v for k, v in vars(m).items() if k.isupper() and isinstance(v, (str, int, float, bool, tuple))}
    import numpy
    import pandas
    import scipy

    record = {
        "stage": stage,
        "mode": args.mode,
        "engine": args.engine,
        "configuration": configuration,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__},
        "blas_threads": 1,
        "seconds": time.perf_counter() - start,
    }
    (out / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    if audits is not None:
        from cvhnn.independent_adapter import save_audits

        save_audits(audits, out)
    config.retain(out, args.save_data, args.save_figures)
    if not (args.save_data and args.save_figures):
        kept = [name for name, keep in [("data", args.save_data), ("figures", args.save_figures)] if keep]
        print(f"Saved: {', '.join(kept) or 'nothing'} ({out})")
