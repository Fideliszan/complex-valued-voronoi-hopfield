"""Regenerate publication artwork from original measurements and explicit geometry."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import (
    aes,
    element_text,
    facet_grid,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_shape_manual,
    theme,
    theme_minimal,
)

from cvhnn import recall
from cvhnn.config import FIGURES, PAPER_DATA, SELECTED_HSV_SEED
from cvhnn.style import RULE_COLORS as COLORS
from cvhnn.style import RULE_MARKERS as MARKERS


def style():
    from cvhnn.style import configure

    configure()
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Termes Figure",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#717984",
            "grid.color": "#E6EAF0",
            "grid.linewidth": 0.55,
            "axes.labelcolor": "#263440",
            "text.color": "#263440",
            "xtick.color": "#52606C",
            "ytick.color": "#52606C",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def save(fig, name, out):
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def clip(poly, a, b):
    """Clip a convex polygon to a.x <= b (exact planar Voronoi halfspace)."""
    result = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        u = float(a @ p - b)
        v = float(a @ q - b)
        if u <= 1e-12:
            result.append(p)
        if (u < 0 < v) or (v < 0 < u):
            result.append(p + (q - p) * u / (u - v))
    return np.asarray(result)


def trajectories(data):
    """Publication data store the large history compressed; fresh runs write plain CSV."""
    path = data / "experiment1_trajectories.csv"
    return pd.read_csv(path if path.exists() else path.with_suffix(".csv.gz"))


def render(data, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_data").mkdir(exist_ok=True)
    style()
    from render_geometry import render as render_geometry

    render_geometry(out)

    h = trajectories(data)
    d = h[(h.alphabet == "16-QAM") & (h.init_protocol == "uniform")]
    # Terminal values are carried forward in the original trajectories.
    g = d.groupby(["rule", "sweep"])[["E_c", "E_r", "rho", "boundary_fraction"]].agg(["mean", "std"]).reset_index()
    g.to_csv(out / "source_data/saturation_plot.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), layout="constrained")
    specs = [("E_c", r"$E_c(t)$"), ("E_r", r"$E_r(t)$"), ("rho", r"$\rho(t)$"), ("boundary_fraction", r"$b(t)$")]
    for panel, (ax, (metric, label)) in enumerate(zip(axes.flat, specs, strict=True)):
        for rule in ["classic", "voronoi"]:
            a = g[g["rule"] == rule]
            x = a["sweep"].to_numpy()
            y = a[(metric, "mean")].to_numpy()
            sd = a[(metric, "std")].to_numpy()
            ax.fill_between(x, y - sd, y + sd, color=COLORS[rule], alpha=0.12, lw=0)
            ax.plot(
                x,
                y,
                color=COLORS[rule],
                marker=MARKERS[rule],
                markevery=8,
                ms=4,
                lw=1.7,
                ls="-" if rule == "classic" else "--",
                label="Classical" if rule == "classic" else "Voronoi",
            )
        ax.set(xlabel="Asynchronous sweeps", ylabel=label, xlim=(0, 80))
        ax.set_title(f"({chr(97 + panel)})", loc="left", weight="normal")
    axes[0, 0].legend(frameon=False, loc="upper right")
    save(fig, "saturation_diagnostics", out)

    # Recompute the representative image from the original seeds; do not recolor HSV data.
    recall.MAX_SWEEPS = 80
    from cvhnn.independent_adapter import install

    install(recall, "recall")
    image_dir = out / "source_data/images"
    image_dir.mkdir(exist_ok=True)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        hue, sigma = recall.generate_synthetic_hsv_memories(6, recall.IMAGE_SHAPE, seed=recall.RANDOM_SEED)
        alphabet = recall.build_alphabets()["64-QAM"]
        indices, values = recall.quantize_hsv_memories(hue, sigma, alphabet, 4)
        weights, _ = recall.build_projection_weights(values)
        seed = SELECTED_HSV_SEED
        target = indices[0]
        initial = recall.make_initial_condition(
            target, hue[0], sigma[0], alphabet, "hsv", 0.10, np.random.default_rng(seed)
        )
        schedule = recall.make_schedule(len(target), 80, np.random.default_rng(seed + 10_000_000))
        images = {"stored_target": target, "corrupted_initial": initial}
        rows = []
        for rule, name in [("classic", "classical_recall"), ("voronoi", "voronoi_recall")]:
            result = recall.run_dynamics(alphabet, weights, initial, schedule, rule, 80)
            images[name] = result.final_indices
            rows.append(
                {
                    "rule": rule,
                    "trial_seed": seed,
                    "initial_SER": np.mean(initial != target),
                    **recall.compute_metrics(result.final_indices, target, alphabet, recall.IMAGE_SHAPE),
                }
            )
        pd.DataFrame(rows).to_csv(out / "source_data/paper_visual_metrics.csv", index=False)
        for name, idx in images.items():
            rgb = recall.state_indices_to_rgb(idx, alphabet, recall.IMAGE_SHAPE, 1.0, recall.V0)
            recall.save_rgb_image(image_dir / f"experiment2_{name}.png", rgb)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.8), layout="constrained")
    names = ["stored_target", "corrupted_initial", "classical_recall", "voronoi_recall"]
    titles = ["Stored target", "Corrupted cue", "Classical recall", "Voronoi recall"]
    for i, (ax, n, t) in enumerate(zip(axes.flat, names, titles, strict=True)):
        ax.imshow(plt.imread(image_dir / f"experiment2_{n}.png"), interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"({chr(97 + i)}) {t}")
    save(fig, "hsv_recall", out)

    # Plotnine figure for the navigable repository: mean SER, all original noise levels.
    d = pd.read_csv(data / "experiment2_noise_results.csv")
    g = d.groupby(["alphabet", "corruption_kind", "noise_level", "rule"], as_index=False).SER.mean()
    g["Rule"] = g.rule.map({"classic": "Classical", "voronoi": "Voronoi"})
    g.to_csv(out / "source_data/recall_plot.csv", index=False)
    p = (
        ggplot(g, aes("noise_level", "SER", color="Rule", shape="Rule", group="Rule"))
        + geom_line(size=0.75)
        + geom_point(size=2.1)
        + facet_grid("corruption_kind ~ alphabet", scales="free_x")
        + scale_color_manual(values={"Classical": COLORS["classic"], "Voronoi": COLORS["voronoi"]})
        + scale_shape_manual(values={"Classical": "s", "Voronoi": "o"})
        + theme_minimal(base_size=10)
        + theme(figure_size=(10, 4.7), legend_position="bottom", text=element_text(family="Termes Figure"))
        + labs(x="Nominal corruption level", y="Mean symbol error rate", color="", shape="")
    )
    for ext in ["pdf", "png"]:
        p.save(out / f"recall_summary.{ext}", dpi=600, verbose=False)
    from render_cost import render as render_cost

    render_cost(data, out)
    print(f"Artwork written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=PAPER_DATA)
    p.add_argument("--output", type=Path, default=FIGURES)
    a = p.parse_args()
    render(a.data, a.output)
