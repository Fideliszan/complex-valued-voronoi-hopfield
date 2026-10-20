"""Plot all four cardinalities without averaging distinct alphabet families."""

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_grid,
    geom_line,
    geom_linerange,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

from cvhnn.config import FIGURES, PAPER_DATA
from cvhnn.style import FONT_FAMILY, GRID, INK, MUTED
from cvhnn.style import METHOD_COLORS as PALETTE


def render(data, out):
    from cvhnn.style import configure

    configure()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data)
    raw = pd.read_csv(data / "experiment3_raw_timings.csv")
    # Named aggregation avoids relying on pandas' multi-index display behavior.
    d = raw.groupby(["ensemble", "family", "cardinality", "method"], as_index=False).agg(
        median=("time_per_query_us", "median"), lower=("time_per_query_us", "min"), upper=("time_per_query_us", "max")
    )
    d["Method"] = pd.Categorical(
        d.method.map(
            {
                "distance": "Distance",
                "inertial_score": "Inertial score",
                "kdtree": "KD-tree",
                "structured": "Structured",
            }
        ),
        categories=list(PALETTE),
        ordered=True,
    )
    d["family"] = pd.Categorical(d.family, categories=["PSK", "QAM", "Polar", "Hexagonal"], ordered=True)
    d["Inputs"] = pd.Categorical(
        d.ensemble.map({"uniform": "Uniform fields", "local_fields": "Network fields"}),
        categories=["Uniform fields", "Network fields"],
        ordered=True,
    )
    (out / "source_data").mkdir(exist_ok=True)
    d.to_csv(out / "source_data/activation_scaling.csv", index=False)
    plot = (
        ggplot(d, aes("cardinality", "median", color="Method", shape="Method", linetype="Method", group="Method"))
        + geom_linerange(aes(ymin="lower", ymax="upper"), size=0.55, alpha=0.85, linetype="solid")
        + geom_line(size=0.75)
        + geom_point(size=2.4, stroke=0.6, fill="white")
        + facet_grid("Inputs ~ family")
        + scale_x_log10(breaks=[16, 64, 256, 1024], labels=["16", "64", "256", "1024"], expand=(0.13, 0))
        + scale_y_log10(breaks=[0.3, 1, 3, 10, 30], labels=lambda xs: [f"{x:g}" for x in xs])
        + scale_color_manual(values=PALETTE)
        + scale_shape_manual(values={"Distance": "o", "Inertial score": "s", "KD-tree": "^", "Structured": "D"})
        + scale_linetype_manual(
            values={"Distance": "solid", "Inertial score": "dashed", "KD-tree": "dashdot", "Structured": "dotted"}
        )
        + theme_minimal(base_size=10)
        + theme(
            figure_size=(7.4, 4.9),
            legend_position="bottom",
            text=element_text(family=FONT_FAMILY, color=INK),
            axis_text_x=element_text(size=8),
            axis_text_y=element_text(size=8),
            axis_line=element_line(color=MUTED, size=0.5),
            axis_ticks=element_line(color=MUTED, size=0.4),
            axis_ticks_minor=element_blank(),
            panel_spacing_x=0.04,
            panel_spacing_y=0.055,
            panel_grid_minor=element_blank(),
            panel_grid_major=element_line(color=GRID, size=0.4),
            strip_text=element_text(weight="normal", size=9),
            legend_title=element_blank(),
        )
        + labs(
            x="Alphabet cardinality |S|", y="Microseconds per query (batch average)", color="", shape="", linetype=""
        )
    )
    for ext in ("pdf", "png"):
        plot.save(out / f"activation_scaling.{ext}", dpi=600, verbose=False)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PAPER_DATA)
    p.add_argument("--output", type=Path, default=FIGURES)
    args = p.parse_args()
    render(args.data, args.output)
