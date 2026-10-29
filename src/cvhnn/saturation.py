"""
Experiment 1: Geometric and energy diagnostics for complex-valued Hopfield
networks with finite complex alphabets.

This script is intended for Google Colab and is deliberately written as a
self-contained experiment. It compares two asynchronous update rules:

1. Classical rule:
       argmax_{s in S} Re(conj(s) v_k)

2. Alphabet-dependent Voronoi rule:
       D_S(v_k; current state), the conservative nearest-neighbor rule defined
       in the paper.

Anti-bias safeguards built into the default experiment:
- The default weight matrix is Hermitian with zero diagonal. This satisfies
  w_kk >= 0 for the Voronoi convergence theorem and avoids artificially adding
  positive self-feedback that would push a neuron along its current state.
- Both rules use exactly the same matrix, initial condition, and asynchronous
  schedule for each Monte Carlo seed.
- Two initialization protocols are reported:
    "uniform": samples all symbols uniformly and is the primary neutral setting.
    "inner": samples non-boundary nonzero symbols when available and is a
             diagnostic stress setting, not a hidden default.
- All Monte Carlo runs are saved. No run is filtered or selected after the fact.
- Ties are handled conservatively and deterministically, matching the paper:
  keep the current state if it is one of the maximizers/minimizers; otherwise
  choose the first symbol under a fixed total order.

Outputs:
- CSV and TXT trajectory tables.
- CSV and TXT final run tables.
- A human-readable numerical report in TXT.
- High-resolution PNG figures at 300 dpi and vector PDF figures.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =========================
# User-facing configuration
# =========================

OUTPUT_DIR = Path("experiment1_outputs")
RANDOM_SEED = 20260831

# Publication-style defaults. For a quick smoke test, reduce N_RUNS, N, and
# MAX_SWEEPS in a notebook cell before calling run_experiment().
N = 100
N_RUNS = 100
MAX_SWEEPS = 80

# Weight matrix W = DIAGONAL_VALUE I + COUPLING_SCALE (A0 + A0*) / 2.
# The default DIAGONAL_VALUE = 0.0 is the least interventionist choice:
# it satisfies w_kk >= 0 without inserting positive self-feedback.
DIAGONAL_VALUE = 0.0
COUPLING_SCALE = 1.0

# The "uniform" protocol is the neutral primary analysis. The "inner" protocol
# is an explicit diagnostic stress test for radial drift.
INITIALIZATION_PROTOCOLS = ("uniform", "inner")

FIG_DPI = 300
TIE_ATOL = 1e-12
BOUNDARY_ATOL = 1e-10
MONOTONICITY_TOL = 1e-10


# =====================
# Alphabet construction
# =====================


def sort_alphabet(points: np.ndarray) -> np.ndarray:
    """Sort by modulus, then principal argument, defining the fixed total order."""
    points = np.asarray(points, dtype=np.complex128)
    keys = np.lexsort((np.angle(points), np.abs(points)))
    return points[keys]


def normalize_alphabet(points: np.ndarray) -> np.ndarray:
    """Normalize a finite alphabet so that max |s| = 1."""
    points = np.asarray(points, dtype=np.complex128)
    rmax = np.max(np.abs(points))
    if rmax <= 0:
        raise ValueError("The alphabet must contain at least one nonzero symbol.")
    return sort_alphabet(points / rmax)


def make_psk(m: int = 16) -> np.ndarray:
    k = np.arange(m)
    return normalize_alphabet(np.exp(2j * np.pi * k / m))


def make_qam(m_side: int = 4) -> np.ndarray:
    levels = np.arange(-(m_side - 1), m_side, 2, dtype=float)
    xx, yy = np.meshgrid(levels, levels)
    return normalize_alphabet(xx.ravel() + 1j * yy.ravel())


def make_polar() -> np.ndarray:
    """A 16-symbol polar alphabet with four rings and four phases per ring."""
    radii = np.array([0.25, 0.50, 0.75, 1.00])
    phases = 2 * np.pi * np.arange(4) / 4 + np.pi / 4
    points = []
    for r in radii:
        points.extend(r * np.exp(1j * phases))
    return normalize_alphabet(np.array(points))


def make_hexagonal(m: int = 19, search_radius: int = 3) -> np.ndarray:
    """Return the m closest points from an Eisenstein/hexagonal lattice patch."""
    omega = np.exp(1j * np.pi / 3)
    points = []
    for a in range(-search_radius, search_radius + 1):
        for b in range(-search_radius, search_radius + 1):
            points.append(a + b * omega)

    rounded = {}
    for z in points:
        key = (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))
        rounded[key] = z

    unique = np.array(list(rounded.values()), dtype=np.complex128)
    order = np.lexsort((np.angle(unique), np.abs(unique)))
    selected = unique[order[:m]]
    return normalize_alphabet(selected)


def build_alphabets() -> dict[str, np.ndarray]:
    """Load published symbol coordinates and order, independent of libm sorting."""
    from .ordered_alphabets import load_group

    return load_group("saturation")


def generate_alphabets() -> dict[str, np.ndarray]:
    return {
        "16-PSK": make_psk(16),
        "16-QAM": make_qam(4),
        "16-Polar": make_polar(),
        "19-Hexagonal": make_hexagonal(19),
    }


def alphabet_summary(alphabet: np.ndarray) -> dict[str, Any]:
    radii = np.abs(alphabet)
    rmax = radii.max()
    rounded_radii = np.unique(np.round(radii, 6))
    boundary_count = int(np.sum(np.isclose(radii, rmax, rtol=0.0, atol=BOUNDARY_ATOL)))
    return {
        "M": len(alphabet),
        "R_max": float(rmax),
        "unique_radii": ", ".join(f"{r:.6g}" for r in rounded_radii),
        "boundary_symbols": boundary_count,
        "boundary_fraction_uniform_start": boundary_count / len(alphabet),
        "inner_pool_size": len(inner_indices(alphabet)),
    }


def inner_indices(alphabet: np.ndarray) -> np.ndarray:
    """Return non-boundary nonzero indices for an explicit radial stress test."""
    radii = np.abs(alphabet)
    rmax = radii.max()

    if np.allclose(radii, rmax, rtol=0.0, atol=BOUNDARY_ATOL):
        return np.arange(len(alphabet))

    candidates = np.flatnonzero((radii > TIE_ATOL) & (radii < rmax - BOUNDARY_ATOL))
    if len(candidates) == 0:
        candidates = np.arange(len(alphabet))
    return candidates


def initial_pool(alphabet: np.ndarray, protocol: str) -> np.ndarray:
    if protocol == "uniform":
        return np.arange(len(alphabet))
    if protocol == "inner":
        return inner_indices(alphabet)
    raise ValueError(f"Unknown initialization protocol: {protocol}")


# ======================
# Network and activations
# ======================


def hermitian_weight_matrix(
    n: int,
    rng: np.random.Generator,
    diagonal_value: float = DIAGONAL_VALUE,
    coupling_scale: float = COUPLING_SCALE,
) -> np.ndarray:
    real = rng.normal(size=(n, n))
    imag = rng.normal(size=(n, n))
    a = (real + 1j * imag) / math.sqrt(2 * n)
    np.fill_diagonal(a, 0.0)
    h = (a + a.conj().T) / 2.0
    w = coupling_scale * h
    np.fill_diagonal(w, diagonal_value)
    return w.astype(np.complex128)


def validate_weight_matrix(w: np.ndarray) -> dict[str, Any]:
    diag = np.diag(w)
    hermitian_error = float(np.max(np.abs(w - w.conj().T)))
    max_diag_imag = float(np.max(np.abs(np.imag(diag))))
    min_diag_real = float(np.min(np.real(diag)))
    return {
        "hermitian_error": hermitian_error,
        "max_diag_imag_abs": max_diag_imag,
        "min_diag_real": min_diag_real,
        "diag_nonnegative": bool(min_diag_real >= -1e-12),
    }


def assert_weight_matrix_matches_theory(w: np.ndarray) -> None:
    checks = validate_weight_matrix(w)
    if checks["hermitian_error"] > 1e-10:
        raise ValueError(f"W is not Hermitian. Error={checks['hermitian_error']:.3e}")
    if checks["max_diag_imag_abs"] > 1e-12:
        raise ValueError("W has a non-real diagonal.")
    if not checks["diag_nonnegative"]:
        raise ValueError("W violates the condition w_kk >= 0.")


def classical_activation_index(
    v: complex,
    current_index: int,
    alphabet: np.ndarray,
    atol: float = TIE_ATOL,
) -> int:
    scores = np.real(np.conj(alphabet) * v)
    best = np.max(scores)
    maximizers = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=atol))
    if current_index in maximizers:
        return current_index
    return int(maximizers[0])


def voronoi_activation_index(
    v: complex,
    current_index: int,
    alphabet: np.ndarray,
    atol: float = TIE_ATOL,
) -> int:
    distances = np.abs(alphabet - v) ** 2
    best = np.min(distances)
    minimizers = np.flatnonzero(np.isclose(distances, best, rtol=0.0, atol=atol))
    if current_index in minimizers:
        return current_index
    return int(minimizers[0])


# ==================
# Metrics and runner
# ==================


@dataclass
class Trajectory:
    history: pd.DataFrame
    final_indices: np.ndarray
    steps_to_convergence: int
    converged: bool
    monotonicity_violations: int
    lyapunov_column: str


def compute_metrics(
    state: np.ndarray,
    field: np.ndarray,
    alphabet: np.ndarray,
) -> tuple[float, float, float, float]:
    n = len(state)
    rmax = float(np.max(np.abs(alphabet)))
    ec = -0.5 * float(np.real(np.vdot(state, field)))
    norm_sq = float(np.sum(np.abs(state) ** 2))
    er = ec + 0.5 * norm_sq
    rho = norm_sq / (n * rmax**2)
    boundary = float(np.mean(np.isclose(np.abs(state), rmax, rtol=0.0, atol=BOUNDARY_ATOL)))
    return ec, er, rho, boundary


def count_monotonicity_violations(values: Iterable[float], tol: float) -> int:
    values = np.asarray(list(values), dtype=float)
    return int(np.sum(np.diff(values) > tol))


def run_dynamics(
    alphabet: np.ndarray,
    w: np.ndarray,
    initial_indices: np.ndarray,
    schedule: Sequence[np.ndarray],
    rule_name: str,
    activation: Callable[[complex, int, np.ndarray], int],
    max_sweeps: int = MAX_SWEEPS,
) -> Trajectory:
    indices = initial_indices.copy()
    state = alphabet[indices].copy()
    field = w @ state
    n = len(state)
    rows = []

    ec, er, rho, boundary = compute_metrics(state, field, alphabet)
    rows.append(
        {
            "sweep": 0,
            "attempted_updates": 0,
            "E_c": ec,
            "E_r": er,
            "rho": rho,
            "boundary_fraction": boundary,
        }
    )

    converged = False
    steps_to_convergence = max_sweeps * n

    for sweep in range(1, max_sweeps + 1):
        changed_this_sweep = False
        attempted_updates = sweep * n

        for k in schedule[sweep - 1]:
            old_index = int(indices[k])
            new_index = activation(field[k], old_index, alphabet)
            if new_index != old_index:
                old_value = state[k]
                new_value = alphabet[new_index]
                delta = new_value - old_value

                indices[k] = new_index
                state[k] = new_value
                field += w[:, k] * delta
                changed_this_sweep = True

        ec, er, rho, boundary = compute_metrics(state, field, alphabet)
        rows.append(
            {
                "sweep": sweep,
                "attempted_updates": attempted_updates,
                "E_c": ec,
                "E_r": er,
                "rho": rho,
                "boundary_fraction": boundary,
            }
        )

        # A full sweep with no changes means every coordinate is fixed under the
        # current state and current rule, so this is a genuine fixed point.
        if not changed_this_sweep:
            converged = True
            steps_to_convergence = attempted_updates
            break

    history = pd.DataFrame(rows)
    lyapunov_column = "E_c" if rule_name == "classic" else "E_r"
    violations = count_monotonicity_violations(history[lyapunov_column], MONOTONICITY_TOL)
    return Trajectory(
        history=history,
        final_indices=indices,
        steps_to_convergence=steps_to_convergence,
        converged=converged,
        monotonicity_violations=violations,
        lyapunov_column=lyapunov_column,
    )


def pad_history(history: pd.DataFrame, n: int, max_sweeps: int = MAX_SWEEPS) -> pd.DataFrame:
    """Pad with the last row so Monte Carlo curves share the same length."""
    if int(history["sweep"].iloc[-1]) >= max_sweeps:
        return history

    rows = [history]
    last = history.iloc[-1].copy()
    for sweep in range(int(last["sweep"]) + 1, max_sweeps + 1):
        padded = last.copy()
        padded["sweep"] = sweep
        padded["attempted_updates"] = sweep * n
        rows.append(pd.DataFrame([padded]))
    return pd.concat(rows, ignore_index=True)


def summarize_final_results(final_df: pd.DataFrame) -> pd.DataFrame:
    return (
        final_df.groupby(["init_protocol", "alphabet", "rule"])
        .agg(
            initial_rho_mean=("initial_rho", "mean"),
            final_rho_mean=("rho", "mean"),
            final_rho_std=("rho", "std"),
            delta_rho_mean=("delta_rho", "mean"),
            initial_boundary_mean=("initial_boundary_fraction", "mean"),
            final_boundary_mean=("boundary_fraction", "mean"),
            final_boundary_std=("boundary_fraction", "std"),
            delta_boundary_mean=("delta_boundary_fraction", "mean"),
            final_E_c_mean=("E_c", "mean"),
            final_E_c_std=("E_c", "std"),
            final_E_r_mean=("E_r", "mean"),
            final_E_r_std=("E_r", "std"),
            steps_mean=("steps_to_convergence", "mean"),
            steps_std=("steps_to_convergence", "std"),
            convergence_rate=("converged", "mean"),
            monotonicity_violations_mean=("monotonicity_violations", "mean"),
            monotonicity_violations_max=("monotonicity_violations", "max"),
        )
        .reset_index()
    )


def build_pairwise_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, alphabet), group in summary_df.groupby(["init_protocol", "alphabet"]):
        by_rule = group.set_index("rule")
        if {"classic", "voronoi"}.issubset(set(by_rule.index)):
            rows.append(
                {
                    "init_protocol": protocol,
                    "alphabet": alphabet,
                    "classic_minus_voronoi_final_rho": float(
                        by_rule.loc["classic", "final_rho_mean"] - by_rule.loc["voronoi", "final_rho_mean"]
                    ),
                    "classic_minus_voronoi_final_boundary": float(
                        by_rule.loc["classic", "final_boundary_mean"] - by_rule.loc["voronoi", "final_boundary_mean"]
                    ),
                    "classic_minus_voronoi_steps": float(
                        by_rule.loc["classic", "steps_mean"] - by_rule.loc["voronoi", "steps_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_text_outputs(
    history_df: pd.DataFrame,
    final_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    alphabet_df: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    history_df.to_csv(OUTPUT_DIR / "experiment1_trajectories.txt", sep="\t", index=False)
    final_df.to_csv(OUTPUT_DIR / "experiment1_final_runs.txt", sep="\t", index=False)
    summary_df.to_csv(OUTPUT_DIR / "experiment1_summary.txt", sep="\t", index=False)
    pairwise_df.to_csv(OUTPUT_DIR / "experiment1_pairwise_comparison.txt", sep="\t", index=False)

    report_lines = [
        "Experiment 1: Geometric and energy diagnostics",
        "=" * 58,
        "",
        "Purpose",
        "-------",
        "Compare the classical inner-product decision rule with the alphabet-dependent",
        "Voronoi activation D_S in finite complex alphabets. The experiment measures",
        "classical energy E_c(t), inertial energy E_r(t), normalized norm rho(t),",
        "boundary fraction b(t), and updates to convergence.",
        "",
        "Configuration",
        "-------------",
        f"Random seed: {RANDOM_SEED}",
        f"N: {N}",
        f"Monte Carlo runs per alphabet/protocol: {N_RUNS}",
        f"Maximum full asynchronous sweeps: {MAX_SWEEPS}",
        f"Diagonal value w_kk: {DIAGONAL_VALUE}",
        f"Coupling scale: {COUPLING_SCALE}",
        f"Initialization protocols: {', '.join(INITIALIZATION_PROTOCOLS)}",
        f"Tie tolerance: {TIE_ATOL}",
        f"Boundary tolerance: {BOUNDARY_ATOL}",
        f"Monotonicity tolerance: {MONOTONICITY_TOL}",
        f"Elapsed seconds: {elapsed_seconds:.3f}",
        "",
        "Anti-bias safeguards",
        "--------------------",
        "1. Default W has zero diagonal, avoiding artificial positive self-feedback.",
        "2. W is explicitly checked to be Hermitian with w_kk >= 0.",
        "3. Each paired run uses the same W, same initial state, and same update schedule.",
        "4. The uniform initialization protocol samples all symbols and is the primary setting.",
        "5. The inner initialization protocol is reported separately as a diagnostic stress test.",
        "6. All runs are saved; no post-hoc filtering or manual selection is applied.",
        "7. Ties follow the conservative deterministic convention used in the paper.",
        "",
        "Alphabet summary",
        "----------------",
        alphabet_df.to_string(index=False),
        "",
        "Weight matrix validation summary",
        "--------------------------------",
        validation_df.describe(include="all").to_string(),
        "",
        "Final summary by initialization protocol, alphabet, and rule",
        "-----------------------------------------------------------",
        summary_df.to_string(index=False),
        "",
        "Pairwise comparison: classical minus D_S",
        "----------------------------------------",
        pairwise_df.to_string(index=False),
        "",
        "Notes for interpretation",
        "------------------------",
        "Positive values in classic_minus_voronoi_final_rho indicate a larger final",
        "normalized norm under the classical rule. Positive values in",
        "classic_minus_voronoi_final_boundary indicate stronger occupation of the",
        "amplitude saturation boundary under the classical rule.",
        "",
    ]
    (OUTPUT_DIR / "experiment1_numerical_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alphabets = build_alphabets()
    master_rng = np.random.default_rng(RANDOM_SEED)

    all_histories = []
    final_rows = []
    validation_rows = []

    alphabet_df = pd.DataFrame(
        [{"alphabet": name, **alphabet_summary(alphabet)} for name, alphabet in alphabets.items()]
    )
    alphabet_df.to_csv(OUTPUT_DIR / "experiment1_alphabet_summary.csv", index=False)
    alphabet_df.to_csv(OUTPUT_DIR / "experiment1_alphabet_summary.txt", sep="\t", index=False)

    start = time.time()
    for alphabet_name, alphabet in alphabets.items():
        for protocol in INITIALIZATION_PROTOCOLS:
            starts = initial_pool(alphabet, protocol)
            print(f"Running {alphabet_name}, protocol={protocol}: |S|={len(alphabet)}, initial pool={len(starts)}")

            for run in range(N_RUNS):
                run_seed = int(master_rng.integers(0, 2**32 - 1))
                rng = np.random.default_rng(run_seed)
                w = hermitian_weight_matrix(N, rng)
                assert_weight_matrix_matches_theory(w)
                validation = validate_weight_matrix(w)
                validation.update(
                    {
                        "alphabet": alphabet_name,
                        "init_protocol": protocol,
                        "run": run,
                        "seed": run_seed,
                    }
                )
                validation_rows.append(validation)

                initial_indices = rng.choice(starts, size=N, replace=True)
                schedule = [rng.permutation(N) for _ in range(MAX_SWEEPS)]

                common = {
                    "alphabet": alphabet_name,
                    "init_protocol": protocol,
                    "run": run,
                    "seed": run_seed,
                }

                for rule_name, activation in [
                    ("classic", classical_activation_index),
                    ("voronoi", voronoi_activation_index),
                ]:
                    traj = run_dynamics(
                        alphabet=alphabet,
                        w=w,
                        initial_indices=initial_indices,
                        schedule=schedule,
                        rule_name=rule_name,
                        activation=activation,
                        max_sweeps=MAX_SWEEPS,
                    )

                    hist = pad_history(traj.history, n=N, max_sweeps=MAX_SWEEPS)
                    for key, value in common.items():
                        hist[key] = value
                    hist["rule"] = rule_name
                    hist["lyapunov_column"] = traj.lyapunov_column
                    all_histories.append(hist)

                    initial = traj.history.iloc[0].to_dict()
                    final = traj.history.iloc[-1].to_dict()
                    final.update(common)
                    final.update(
                        {
                            "rule": rule_name,
                            "initial_E_c": initial["E_c"],
                            "initial_E_r": initial["E_r"],
                            "initial_rho": initial["rho"],
                            "initial_boundary_fraction": initial["boundary_fraction"],
                            "delta_E_c": final["E_c"] - initial["E_c"],
                            "delta_E_r": final["E_r"] - initial["E_r"],
                            "delta_rho": final["rho"] - initial["rho"],
                            "delta_boundary_fraction": final["boundary_fraction"] - initial["boundary_fraction"],
                            "steps_to_convergence": traj.steps_to_convergence,
                            "converged": traj.converged,
                            "monotonicity_violations": traj.monotonicity_violations,
                            "lyapunov_column": traj.lyapunov_column,
                        }
                    )
                    final_rows.append(final)

    elapsed = time.time() - start

    history_df = pd.concat(all_histories, ignore_index=True)
    final_df = pd.DataFrame(final_rows)
    validation_df = pd.DataFrame(validation_rows)
    summary_df = summarize_final_results(final_df)
    pairwise_df = build_pairwise_comparison(summary_df)

    history_df.to_csv(OUTPUT_DIR / "experiment1_trajectories.csv", index=False)
    final_df.to_csv(OUTPUT_DIR / "experiment1_final_runs.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "experiment1_weight_validation.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "experiment1_summary.csv", index=False)
    pairwise_df.to_csv(OUTPUT_DIR / "experiment1_pairwise_comparison.csv", index=False)

    write_text_outputs(
        history_df=history_df,
        final_df=final_df,
        summary_df=summary_df,
        pairwise_df=pairwise_df,
        validation_df=validation_df,
        alphabet_df=alphabet_df,
        elapsed_seconds=elapsed,
    )

    print(f"\nFinished in {elapsed:.1f} seconds.")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nPairwise comparison: classical minus D_S")
    print(pairwise_df.to_string(index=False))
    return history_df, final_df, summary_df


# ========
# Plotting
# ========


def save_figure(fig: Any, filename_stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.png", dpi=FIG_DPI, bbox_inches="tight")


def mean_std_by_sweep(
    history_df: pd.DataFrame,
    alphabet: str,
    init_protocol: str,
    rule: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = history_df[
        (history_df["alphabet"] == alphabet)
        & (history_df["init_protocol"] == init_protocol)
        & (history_df["rule"] == rule)
    ]
    grouped = subset.groupby("sweep")[metric]
    x = grouped.mean().index.to_numpy()
    mean = grouped.mean().to_numpy()
    std = grouped.std().fillna(0.0).to_numpy()
    return x, mean, std


def plot_metric_axis(
    ax: Any,
    history_df: pd.DataFrame,
    alphabet: str,
    init_protocol: str,
    metric: str,
    ylabel: str,
) -> None:
    styles = {
        "classic": {"color": "#C62828", "label": "Classical"},
        "voronoi": {"color": "#1565C0", "label": r"$\mathcal{D}_{\mathcal{S}}$"},
    }
    for rule, style in styles.items():
        x, mean, std = mean_std_by_sweep(history_df, alphabet, init_protocol, rule, metric)
        ax.plot(x, mean, color=style["color"], lw=2.0, label=style["label"])
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.14, lw=0)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def make_qam_dynamics_figure(history_df: pd.DataFrame, init_protocol: str) -> None:
    import matplotlib.pyplot as plt

    alphabet = "16-QAM"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Experiment 1: 16-QAM diagnostics ({init_protocol} initialization)",
        fontsize=14,
    )

    plot_metric_axis(axes[0, 0], history_df, alphabet, init_protocol, "E_c", r"$E_c(t)$")
    plot_metric_axis(axes[0, 1], history_df, alphabet, init_protocol, "E_r", r"$E_r(t)$")
    plot_metric_axis(axes[1, 0], history_df, alphabet, init_protocol, "rho", r"$\rho(t)$")
    plot_metric_axis(
        axes[1, 1],
        history_df,
        alphabet,
        init_protocol,
        "boundary_fraction",
        r"$b(t)$",
    )

    for ax in axes.flat:
        ax.set_xlabel("Full asynchronous sweeps")
    axes[0, 0].legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, f"experiment1_qam_diagnostics_{init_protocol}")
    plt.show()


def make_alphabet_summary_figure(final_df: pd.DataFrame, init_protocol: str) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("rho", "Final normalized norm"),
        ("boundary_fraction", "Final saturation-boundary fraction"),
        ("steps_to_convergence", "Updates until convergence"),
        ("monotonicity_violations", "Lyapunov monotonicity violations"),
    ]
    alphabets = list(build_alphabets().keys())
    colors = {"classic": "#C62828", "voronoi": "#1565C0"}
    labels = {"classic": "Classical", "voronoi": r"$\mathcal{D}_{\mathcal{S}}$"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(
        f"Experiment 1: final diagnostics by alphabet ({init_protocol} initialization)",
        fontsize=14,
    )
    x = np.arange(len(alphabets))
    width = 0.36

    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        for offset, rule in [(-width / 2, "classic"), (width / 2, "voronoi")]:
            means = []
            errs = []
            for alphabet in alphabets:
                vals = final_df[
                    (final_df["alphabet"] == alphabet)
                    & (final_df["init_protocol"] == init_protocol)
                    & (final_df["rule"] == rule)
                ][metric]
                means.append(vals.mean())
                errs.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
            ax.bar(
                x + offset,
                means,
                width,
                yerr=errs,
                capsize=3,
                color=colors[rule],
                alpha=0.85,
                label=labels[rule],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(alphabets, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, f"experiment1_summary_by_alphabet_{init_protocol}")
    plt.show()


def make_energy_monotonicity_figure(history_df: pd.DataFrame, init_protocol: str) -> None:
    """Plot the Lyapunov candidate used by each rule for every alphabet."""
    import matplotlib.pyplot as plt

    alphabets = list(build_alphabets().keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    fig.suptitle(
        f"Experiment 1: rule-specific Lyapunov candidates ({init_protocol} initialization)",
        fontsize=14,
    )

    for ax, alphabet in zip(axes.flat, alphabets, strict=True):
        x, mean, std = mean_std_by_sweep(history_df, alphabet, init_protocol, "classic", "E_c")
        ax.plot(x, mean, color="#C62828", lw=2.0, label=r"Classical: $E_c$")
        ax.fill_between(x, mean - std, mean + std, color="#C62828", alpha=0.14, lw=0)

        x, mean, std = mean_std_by_sweep(history_df, alphabet, init_protocol, "voronoi", "E_r")
        ax.plot(x, mean, color="#1565C0", lw=2.0, label=r"$\mathcal{D}_{\mathcal{S}}$: $E_r$")
        ax.fill_between(x, mean - std, mean + std, color="#1565C0", alpha=0.14, lw=0)

        ax.set_title(alphabet)
        ax.set_xlabel("Full asynchronous sweeps")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, f"experiment1_rule_specific_energies_{init_protocol}")
    plt.show()


def make_all_figures(history_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    for protocol in sorted(history_df["init_protocol"].unique()):
        make_qam_dynamics_figure(history_df, protocol)
        make_alphabet_summary_figure(final_df, protocol)
        make_energy_monotonicity_figure(history_df, protocol)


def zip_outputs() -> Path:
    zip_path = shutil.make_archive(str(OUTPUT_DIR), "zip", OUTPUT_DIR)
    return Path(zip_path)


if __name__ == "__main__":
    history, final, summary = run_experiment()
    make_all_figures(history, final)
    archive = zip_outputs()
    print(f"\nSaved outputs in: {OUTPUT_DIR.resolve()}")
    print(f"ZIP archive: {archive.resolve()}")

    try:
        from google.colab import files  # type: ignore  # noqa: F401  (probe: detects a Colab runtime)

        print("\nColab detected. The ZIP can be downloaded from the file browser.")
        # Uncomment the next line to open a download dialog automatically.
        # files.download(str(archive))
    except Exception:
        pass
