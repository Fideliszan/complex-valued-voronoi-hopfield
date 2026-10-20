"""
Experiment 2: HSV-encoded associative recall with complex-valued Hopfield
networks over finite complex alphabets.

This script is intended for Google Colab and is self-contained: it generates
deterministic synthetic HSV memories, encodes hue as complex phase, encodes HSV
saturation as complex magnitude, stores several memories by the projection
rule, corrupts target memories, and compares two asynchronous dynamics:

1. Classical inner-product rule
       argmax_{s in S} Re(conj(s) v_k)

2. Alphabet-dependent Voronoi rule
       D_S(v_k), the conservative nearest-neighbor rule defined in the paper.

Anti-bias safeguards built into the experiment:
- The same alphabet, stored memories, weight matrix, corrupted initial state,
  and asynchronous update schedule are used for the two rules in every paired
  trial.
- The storage matrix is W = X (X^* X)^dagger X^*, which is Hermitian positive
  semidefinite. Its diagonal is nonnegative naturally; no artificial positive
  self-feedback is inserted.
- Ties are handled deterministically and conservatively: if the current state
  is one of the maximizers/minimizers, the neuron keeps its current value.
- All runs are saved. The script does not filter, select, or tune runs after
  observing the results.

Outputs saved in experiment2_outputs/:
- CSV and TXT numerical tables for all runs and grouped summaries.
- A human-readable numerical report in TXT.
- High-resolution PNG figures at 300 dpi and vector PDF copies.
- High-resolution PNG image panels at 300 dpi.
- A final experiment2_outputs.zip archive.

Recommended Colab usage:
    !python experiment2_hsv_associative_recall_colab.py

For a faster first pass, edit N_TRIALS_NOISE, N_TRIALS_LOAD, IMAGE_SHAPE, or
MAX_SWEEPS in the configuration block below.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =========================
# User-facing configuration
# =========================

OUTPUT_DIR = Path("experiment2_outputs")
RANDOM_SEED = 20260901

# A moderate default size for Colab. Increase to (32, 32) or (40, 40) after a
# successful first run if you want smoother images.
IMAGE_SHAPE = (24, 24)
V0 = 0.8

# Publication-style comparison. The 16-symbol variants are also implemented
# below so that quick smoke tests can override ALPHABETS_TO_RUN.
ALPHABETS_TO_RUN = ("64-PSK", "64-QAM", "64-Polar", "61-Hexagonal")

# Noise experiment: fixed number of stored memories, several corruption levels.
P_NOISE = 4
N_TRIALS_NOISE = 30
SYMBOL_NOISE_LEVELS = (0.00, 0.05, 0.10, 0.20, 0.30)
HSV_NOISE_LEVELS = (0.00, 0.02, 0.05, 0.10, 0.15)

# Load experiment: fixed corruption, increasing number of stored memories.
P_LOAD_VALUES = (1, 2, 4, 6)
N_TRIALS_LOAD = 2
LOAD_CORRUPTION_KIND = "symbol"
LOAD_NOISE_LEVEL = 0.10

# Asynchronous dynamics.
MAX_SWEEPS = 80
TIE_ATOL = 1e-12
MONOTONICITY_TOL = 1e-9
PINV_RCOND = 1e-10

# Visual output.
MAKE_FIGURES = True
SAVE_OPTIONAL_PLOTS = False
FIG_DPI = 300
IMAGE_UPSCALE = 10
VISUAL_ALPHABET = "64-QAM"
VISUAL_P = 4
VISUAL_TARGET_MU = 0
VISUAL_CORRUPTION_KIND = "hsv"
VISUAL_NOISE_LEVEL = 0.10

# Recreate the output folder on each run.
OVERWRITE_OUTPUT_DIR = True


# =====================
# Alphabet construction
# =====================


def sort_alphabet(points: np.ndarray) -> np.ndarray:
    """Sort by modulus and then principal argument, fixing the total order."""
    points = np.asarray(points, dtype=np.complex128)
    keys = np.lexsort((np.angle(points), np.abs(points)))
    return points[keys]


def normalize_alphabet(points: np.ndarray) -> np.ndarray:
    """Normalize a finite alphabet so that max |s| = 1."""
    points = np.asarray(points, dtype=np.complex128)
    rmax = float(np.max(np.abs(points)))
    if rmax <= 0:
        raise ValueError("The alphabet must contain at least one nonzero symbol.")
    return sort_alphabet(points / rmax)


def make_psk(m: int) -> np.ndarray:
    k = np.arange(m)
    return normalize_alphabet(np.exp(2j * np.pi * k / m))


def make_qam(side: int) -> np.ndarray:
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    xx, yy = np.meshgrid(levels, levels)
    return normalize_alphabet(xx.ravel() + 1j * yy.ravel())


def make_polar(n_radii: int, n_phases: int) -> np.ndarray:
    radii = np.linspace(1.0 / n_radii, 1.0, n_radii)
    phases = 2 * np.pi * np.arange(n_phases) / n_phases
    points: list[complex] = []
    for radius in radii:
        points.extend(radius * np.exp(1j * phases))
    return normalize_alphabet(np.array(points, dtype=np.complex128))


def make_hexagonal(m: int, search_radius: int = 6) -> np.ndarray:
    """Return the m closest points from a finite Eisenstein lattice patch."""
    omega = np.exp(1j * np.pi / 3)
    points: list[complex] = []
    for a in range(-search_radius, search_radius + 1):
        for b in range(-search_radius, search_radius + 1):
            points.append(a + b * omega)

    unique: dict[tuple[float, float], complex] = {}
    for z in points:
        key = (round(float(np.real(z)), 12), round(float(np.imag(z)), 12))
        unique[key] = z

    patch = np.array(list(unique.values()), dtype=np.complex128)
    order = np.lexsort((np.angle(patch), np.abs(patch)))
    return normalize_alphabet(patch[order[:m]])


def build_alphabets() -> dict[str, np.ndarray]:
    """Load published symbol coordinates and order, independent of libm sorting."""
    from .ordered_alphabets import load_group

    return load_group("recall")


def generate_alphabets() -> dict[str, np.ndarray]:
    return {
        "16-PSK": make_psk(16),
        "16-QAM": make_qam(4),
        "16-Polar": make_polar(4, 4),
        "19-Hexagonal": make_hexagonal(19, search_radius=3),
        "64-PSK": make_psk(64),
        "64-QAM": make_qam(8),
        "64-Polar": make_polar(8, 8),
        "61-Hexagonal": make_hexagonal(61, search_radius=6),
    }


def alphabet_summary_row(name: str, alphabet: np.ndarray) -> dict[str, Any]:
    radii = np.abs(alphabet)
    rounded_radii = np.unique(np.round(radii, 6))
    rmax = float(np.max(radii))
    boundary_count = int(np.sum(np.isclose(radii, rmax, atol=1e-10, rtol=0.0)))
    zero_count = int(np.sum(np.isclose(radii, 0.0, atol=1e-12, rtol=0.0)))
    return {
        "alphabet": name,
        "cardinality": int(len(alphabet)),
        "R_max": rmax,
        "unique_radius_count": int(len(rounded_radii)),
        "unique_radii": ", ".join(f"{r:.6g}" for r in rounded_radii),
        "boundary_symbols": boundary_count,
        "boundary_fraction": boundary_count / len(alphabet),
        "zero_symbols": zero_count,
    }


# =====================
# HSV/complex conversion
# =====================


def hsv_fields_to_complex(hue: np.ndarray, saturation: np.ndarray, rmax: float = 1.0) -> np.ndarray:
    """Encode HSV hue and saturation as complex phase and magnitude."""
    theta = 2.0 * np.pi * (np.asarray(hue) % 1.0) - np.pi
    sigma = np.clip(np.asarray(saturation, dtype=float), 0.0, 1.0)
    return rmax * sigma * np.exp(1j * theta)


def complex_to_hsv_fields(
    z: np.ndarray, rmax: float = 1.0, v0: float = V0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a complex state into HSV fields for visualization."""
    z = np.asarray(z, dtype=np.complex128)
    saturation = np.clip(np.abs(z) / max(rmax, 1e-12), 0.0, 1.0)
    hue = ((np.angle(z) + np.pi) / (2.0 * np.pi)) % 1.0
    hue = np.where(saturation <= 1e-12, 0.0, hue)
    value = np.full_like(saturation, fill_value=v0, dtype=float)
    return hue, saturation, value


def hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray | float) -> np.ndarray:
    """Vectorized HSV-to-RGB conversion with channels in [0, 1]."""
    h = np.asarray(hue, dtype=float) % 1.0
    s = np.clip(np.asarray(saturation, dtype=float), 0.0, 1.0)
    v = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    if v.shape == ():
        v = np.full_like(h, float(v))

    hp = h * 6.0
    i = np.floor(hp).astype(int)
    f = hp - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    r = np.empty_like(h, dtype=float)
    g = np.empty_like(h, dtype=float)
    b = np.empty_like(h, dtype=float)

    masks = [i_mod == idx for idx in range(6)]
    r[masks[0]], g[masks[0]], b[masks[0]] = v[masks[0]], t[masks[0]], p[masks[0]]
    r[masks[1]], g[masks[1]], b[masks[1]] = q[masks[1]], v[masks[1]], p[masks[1]]
    r[masks[2]], g[masks[2]], b[masks[2]] = p[masks[2]], v[masks[2]], t[masks[2]]
    r[masks[3]], g[masks[3]], b[masks[3]] = p[masks[3]], q[masks[3]], v[masks[3]]
    r[masks[4]], g[masks[4]], b[masks[4]] = t[masks[4]], p[masks[4]], v[masks[4]]
    r[masks[5]], g[masks[5]], b[masks[5]] = v[masks[5]], p[masks[5]], q[masks[5]]

    return np.stack([r, g, b], axis=-1)


def state_indices_to_rgb(
    indices: np.ndarray,
    alphabet: np.ndarray,
    image_shape: tuple[int, int],
    rmax: float = 1.0,
    v0: float = V0,
) -> np.ndarray:
    z = alphabet[np.asarray(indices, dtype=int)].reshape(image_shape)
    hue, saturation, value = complex_to_hsv_fields(z, rmax=rmax, v0=v0)
    return hsv_to_rgb(hue, saturation, value)


# ============================
# Synthetic memory construction
# ============================


def generate_synthetic_hsv_memories(
    num_memories: int,
    image_shape: tuple[int, int],
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic HSV fields that exercise phase and magnitude."""
    height, width = image_shape
    y = np.linspace(-1.0, 1.0, height)
    x = np.linspace(-1.0, 1.0, width)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    radius_norm = np.clip(radius / np.sqrt(2.0), 0.0, 1.0)
    angle01 = ((np.arctan2(yy, xx) + np.pi) / (2.0 * np.pi)) % 1.0

    rng = np.random.default_rng(seed)
    hue_bank = np.empty((num_memories, height, width), dtype=float)
    saturation_bank = np.empty_like(hue_bank)

    for mu in range(num_memories):
        rotation = rng.uniform(0.0, 1.0)
        cx = 0.55 * math.cos(2.0 * math.pi * (mu + 1) / max(num_memories, 1))
        cy = 0.55 * math.sin(2.0 * math.pi * (mu + 1) / max(num_memories, 1))
        blob_width = 0.18 + 0.03 * (mu % 3)

        wave_angle = 2.0 * np.pi * (0.11 * mu + 0.07)
        projection = xx * math.cos(wave_angle) + yy * math.sin(wave_angle)
        angular_texture = 0.08 * np.sin((mu + 2) * np.pi * xx) * np.cos((mu + 3) * np.pi * yy)
        radial_twist = 0.12 * np.sin((mu + 1.5) * np.pi * radius)
        hue = (angle01 + rotation + angular_texture + radial_twist) % 1.0

        waves = 0.5 + 0.5 * np.sin((mu + 2) * np.pi * projection + 0.4 * mu)
        rings = 0.5 + 0.5 * np.cos((mu + 2.5) * np.pi * radius - 0.3 * mu)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * blob_width**2))
        diagonal = 0.5 + 0.5 * np.sin((mu + 1) * np.pi * (xx + yy))

        raw_saturation = 0.32 * waves + 0.25 * rings + 0.25 * (1.0 - radius_norm) + 0.30 * blob + 0.12 * diagonal
        raw_saturation -= raw_saturation.min()
        raw_saturation /= raw_saturation.max() + 1e-12
        saturation = 0.04 + 0.96 * raw_saturation

        hue_bank[mu] = hue
        saturation_bank[mu] = np.clip(saturation, 0.0, 1.0)

    return hue_bank, saturation_bank


def save_memory_bank_overview(hue_bank: np.ndarray, saturation_bank: np.ndarray, out_path: Path) -> None:
    if not MAKE_FIGURES:
        return

    import matplotlib.pyplot as plt

    rgb_images = [hsv_to_rgb(hue_bank[i], saturation_bank[i], V0) for i in range(len(hue_bank))]
    cols = min(4, len(rgb_images))
    rows = int(math.ceil(len(rgb_images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if idx < len(rgb_images):
            ax.imshow(rgb_images[idx], interpolation="nearest")
            ax.set_title(f"Memory {idx + 1}", fontsize=9)
    fig.suptitle("Synthetic HSV Memory Bank", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ========================
# Quantization and storage
# ========================


def quantize_complex_to_indices(z: np.ndarray, alphabet: np.ndarray, chunk_size: int = 20000) -> np.ndarray:
    """Static Euclidean quantizer Q_S(z) with deterministic total-order ties."""
    original_shape = z.shape
    flat = np.asarray(z, dtype=np.complex128).reshape(-1)
    indices = np.empty(flat.shape[0], dtype=np.int64)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        distances = np.abs(flat[start:end, None] - alphabet[None, :]) ** 2
        indices[start:end] = np.argmin(distances, axis=1)
    return indices.reshape(original_shape)


def quantize_hsv_memories(
    hue_bank: np.ndarray,
    saturation_bank: np.ndarray,
    alphabet: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return memory indices and memory values with shapes (P, N)."""
    rmax = float(np.max(np.abs(alphabet)))
    z = hsv_fields_to_complex(hue_bank[:p], saturation_bank[:p], rmax=rmax)
    indices = quantize_complex_to_indices(z, alphabet).reshape(p, -1)
    values = alphabet[indices]
    return indices, values


def build_projection_weights(memory_values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct W = X (X^* X)^dagger X^* from memory values of shape (P, N)."""
    x_matrix = memory_values.T.astype(np.complex128)  # N x P
    gram = x_matrix.conj().T @ x_matrix
    gram_pinv = np.linalg.pinv(gram, rcond=PINV_RCOND)
    w = x_matrix @ gram_pinv @ x_matrix.conj().T
    w = (w + w.conj().T) / 2.0

    diag = np.diag(w)
    residual = np.linalg.norm(w @ x_matrix - x_matrix) / max(np.linalg.norm(x_matrix), 1e-12)
    info = {
        "rank_gram": int(np.linalg.matrix_rank(gram, tol=PINV_RCOND)),
        "hermitian_error": float(np.max(np.abs(w - w.conj().T))),
        "min_diag_real": float(np.min(np.real(diag))),
        "max_diag_imag_abs": float(np.max(np.abs(np.imag(diag)))),
        "projection_residual": float(residual),
        "diag_nonnegative": bool(np.min(np.real(diag)) >= -1e-10),
    }
    return w.astype(np.complex128), info


# ======================
# Network and activations
# ======================


@dataclass
class DynamicsResult:
    final_indices: np.ndarray
    sweeps: int
    attempted_updates: int
    changed_updates: int
    converged: bool
    lyapunov_sweep_violations: int
    initial_E_c: float
    final_E_c: float
    initial_E_r: float
    final_E_r: float


def energies(state: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Return classical energy E_c and inertial energy E_r."""
    quadratic = float(np.real(np.vdot(state, w @ state)))
    e_c = -0.5 * quadratic
    e_r = e_c + 0.5 * float(np.real(np.vdot(state, state)))
    return e_c, e_r


def conservative_argmax(scores: np.ndarray, current_idx: int) -> int:
    best = float(np.max(scores))
    tol = TIE_ATOL * max(1.0, abs(best))
    if scores[current_idx] >= best - tol:
        return int(current_idx)
    return int(np.flatnonzero(scores >= best - tol)[0])


def conservative_argmin(values: np.ndarray, current_idx: int) -> int:
    best = float(np.min(values))
    tol = TIE_ATOL * max(1.0, abs(best))
    if values[current_idx] <= best + tol:
        return int(current_idx)
    return int(np.flatnonzero(values <= best + tol)[0])


def classical_activation_index(local_field: complex, alphabet: np.ndarray, current_idx: int) -> int:
    scores = np.real(np.conj(alphabet) * local_field)
    return conservative_argmax(scores, current_idx)


def voronoi_activation_index(local_field: complex, alphabet: np.ndarray, current_idx: int) -> int:
    distances = np.abs(alphabet - local_field) ** 2
    return conservative_argmin(distances, current_idx)


def make_schedule(n: int, max_sweeps: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.permutation(n) for _ in range(max_sweeps)]


def run_dynamics(
    alphabet: np.ndarray,
    w: np.ndarray,
    initial_indices: np.ndarray,
    schedule: Sequence[np.ndarray],
    rule: str,
    max_sweeps: int = MAX_SWEEPS,
) -> DynamicsResult:
    indices = np.asarray(initial_indices, dtype=np.int64).copy()
    state = alphabet[indices].astype(np.complex128).copy()
    field = w @ state
    initial_e_c, initial_e_r = energies(state, w)
    previous_lyapunov = initial_e_c if rule == "classic" else initial_e_r

    attempted_updates = 0
    changed_updates = 0
    lyapunov_sweep_violations = 0
    converged = False
    completed_sweeps = 0

    for sweep_idx in range(max_sweeps):
        changed_this_sweep = 0
        for k in schedule[sweep_idx]:
            k_int = int(k)
            current_idx = int(indices[k_int])
            local_field = field[k_int]
            if rule == "classic":
                new_idx = classical_activation_index(local_field, alphabet, current_idx)
            elif rule == "voronoi":
                new_idx = voronoi_activation_index(local_field, alphabet, current_idx)
            else:
                raise ValueError(f"Unknown rule: {rule}")

            attempted_updates += 1
            if new_idx != current_idx:
                old_value = state[k_int]
                new_value = alphabet[new_idx]
                delta = new_value - old_value
                indices[k_int] = new_idx
                state[k_int] = new_value
                field += w[:, k_int] * delta
                changed_this_sweep += 1
                changed_updates += 1

        completed_sweeps = sweep_idx + 1
        field = w @ state
        current_e_c, current_e_r = energies(state, w)
        current_lyapunov = current_e_c if rule == "classic" else current_e_r
        if changed_this_sweep > 0 and current_lyapunov > previous_lyapunov + MONOTONICITY_TOL:
            lyapunov_sweep_violations += 1
        previous_lyapunov = current_lyapunov

        if changed_this_sweep == 0:
            converged = True
            break

    final_e_c, final_e_r = energies(state, w)
    return DynamicsResult(
        final_indices=indices,
        sweeps=completed_sweeps,
        attempted_updates=attempted_updates,
        changed_updates=changed_updates,
        converged=converged,
        lyapunov_sweep_violations=lyapunov_sweep_violations,
        initial_E_c=initial_e_c,
        final_E_c=final_e_c,
        initial_E_r=initial_e_r,
        final_E_r=final_e_r,
    )


# ======================
# Corruption and metrics
# ======================


def corrupt_symbol_indices(
    target_indices: np.ndarray,
    alphabet_size: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    corrupted = np.asarray(target_indices, dtype=np.int64).copy()
    mask = rng.random(corrupted.shape) < probability
    if np.any(mask):
        corrupted[mask] = rng.integers(0, alphabet_size, size=int(np.sum(mask)))
    return corrupted


def corrupt_hsv_and_quantize(
    target_hue: np.ndarray,
    target_saturation: np.ndarray,
    alphabet: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbed_hue = (target_hue + rng.normal(0.0, noise_level, size=target_hue.shape)) % 1.0
    perturbed_saturation = np.clip(
        target_saturation + rng.normal(0.0, noise_level, size=target_saturation.shape),
        0.0,
        1.0,
    )
    z = hsv_fields_to_complex(perturbed_hue, perturbed_saturation, rmax=float(np.max(np.abs(alphabet))))
    return quantize_complex_to_indices(z, alphabet).reshape(-1)


def make_initial_condition(
    target_indices: np.ndarray,
    target_hue: np.ndarray,
    target_saturation: np.ndarray,
    alphabet: np.ndarray,
    corruption_kind: str,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if corruption_kind == "symbol":
        return corrupt_symbol_indices(target_indices, len(alphabet), noise_level, rng)
    if corruption_kind == "hsv":
        return corrupt_hsv_and_quantize(target_hue, target_saturation, alphabet, noise_level, rng)
    raise ValueError(f"Unknown corruption kind: {corruption_kind}")


def circular_hue_distance(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    diff = np.abs((h1 - h2 + 0.5) % 1.0 - 0.5)
    return diff


def psnr_rgb(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference_rgb) - np.asarray(test_rgb)) ** 2))
    if mse <= 1e-15:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_metrics(
    final_indices: np.ndarray,
    target_indices: np.ndarray,
    alphabet: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    final_values = alphabet[final_indices]
    target_values = alphabet[target_indices]
    rmax = float(np.max(np.abs(alphabet)))

    ser = float(np.mean(final_indices != target_indices))
    exact_recovery = float(ser == 0.0)
    denominator = max(float(np.linalg.norm(final_values) * np.linalg.norm(target_values)), 1e-12)
    complex_similarity = float(abs(np.vdot(target_values, final_values)) / denominator)
    radial_error = float(np.mean(np.abs(np.abs(final_values) - np.abs(target_values))) / max(rmax, 1e-12))

    target_hue, target_sat, _ = complex_to_hsv_fields(target_values, rmax=rmax, v0=V0)
    final_hue, final_sat, _ = complex_to_hsv_fields(final_values, rmax=rmax, v0=V0)
    visible = (target_sat > 1e-10) & (final_sat > 1e-10)
    if np.any(visible):
        angular_error_cycles = float(np.mean(circular_hue_distance(target_hue[visible], final_hue[visible])))
    else:
        angular_error_cycles = 0.0

    target_rgb = state_indices_to_rgb(target_indices, alphabet, image_shape, rmax=rmax, v0=V0)
    final_rgb = state_indices_to_rgb(final_indices, alphabet, image_shape, rmax=rmax, v0=V0)
    psnr = psnr_rgb(target_rgb, final_rgb)

    return {
        "SER": ser,
        "exact_recovery": exact_recovery,
        "complex_similarity": complex_similarity,
        "radial_error": radial_error,
        "angular_error_cycles": angular_error_cycles,
        "rgb_psnr": psnr,
    }


# ====================
# Numerical experiments
# ====================


def paired_recall_trial(
    alphabet_name: str,
    alphabet: np.ndarray,
    hue_bank: np.ndarray,
    saturation_bank: np.ndarray,
    p: int,
    target_mu: int,
    corruption_kind: str,
    noise_level: float,
    trial_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    memory_indices, memory_values = quantize_hsv_memories(hue_bank, saturation_bank, alphabet, p)
    w, weight_info = build_projection_weights(memory_values)
    n = memory_indices.shape[1]

    rng_initial = np.random.default_rng(trial_seed)
    rng_schedule = np.random.default_rng(trial_seed + 10_000_000)

    target_indices = memory_indices[target_mu]
    initial_indices = make_initial_condition(
        target_indices=target_indices,
        target_hue=hue_bank[target_mu],
        target_saturation=saturation_bank[target_mu],
        alphabet=alphabet,
        corruption_kind=corruption_kind,
        noise_level=noise_level,
        rng=rng_initial,
    )
    initial_ser = float(np.mean(initial_indices != target_indices))
    schedule = make_schedule(n, MAX_SWEEPS, rng_schedule)

    rows: list[dict[str, Any]] = []
    for rule in ("classic", "voronoi"):
        result = run_dynamics(alphabet, w, initial_indices, schedule, rule=rule, max_sweeps=MAX_SWEEPS)
        metrics = compute_metrics(result.final_indices, target_indices, alphabet, IMAGE_SHAPE)
        rows.append(
            {
                "alphabet": alphabet_name,
                "alphabet_size": int(len(alphabet)),
                "N": int(n),
                "P": int(p),
                "P_over_N": float(p / n),
                "target_mu": int(target_mu),
                "corruption_kind": corruption_kind,
                "noise_level": float(noise_level),
                "trial_seed": int(trial_seed),
                "rule": rule,
                "initial_SER": initial_ser,
                "SER": metrics["SER"],
                "exact_recovery": metrics["exact_recovery"],
                "complex_similarity": metrics["complex_similarity"],
                "radial_error": metrics["radial_error"],
                "angular_error_cycles": metrics["angular_error_cycles"],
                "rgb_psnr": metrics["rgb_psnr"],
                "sweeps": int(result.sweeps),
                "attempted_updates": int(result.attempted_updates),
                "changed_updates": int(result.changed_updates),
                "converged": float(result.converged),
                "lyapunov_sweep_violations": int(result.lyapunov_sweep_violations),
                "initial_E_c": result.initial_E_c,
                "final_E_c": result.final_E_c,
                "initial_E_r": result.initial_E_r,
                "final_E_r": result.final_E_r,
                "delta_E_c": result.final_E_c - result.initial_E_c,
                "delta_E_r": result.final_E_r - result.initial_E_r,
                "rank_gram": weight_info["rank_gram"],
                "projection_residual": weight_info["projection_residual"],
                "hermitian_error": weight_info["hermitian_error"],
                "min_diag_real": weight_info["min_diag_real"],
                "diag_nonnegative": float(weight_info["diag_nonnegative"]),
            }
        )

    return rows, weight_info


def run_noise_experiment(
    alphabets: dict[str, np.ndarray],
    hue_bank: np.ndarray,
    saturation_bank: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    print("[Experiment 2] Running noise curves...")
    for alphabet_name in ALPHABETS_TO_RUN:
        alphabet = alphabets[alphabet_name]
        for corruption_kind, noise_levels in (
            ("symbol", SYMBOL_NOISE_LEVELS),
            ("hsv", HSV_NOISE_LEVELS),
        ):
            for noise_level in noise_levels:
                for trial in range(N_TRIALS_NOISE):
                    target_mu = trial % P_NOISE
                    trial_seed = (
                        RANDOM_SEED + 1000 * hash_stable(alphabet_name) + 100 * int(round(1000 * noise_level)) + trial
                    )
                    trial_rows, weight_info = paired_recall_trial(
                        alphabet_name=alphabet_name,
                        alphabet=alphabet,
                        hue_bank=hue_bank,
                        saturation_bank=saturation_bank,
                        p=P_NOISE,
                        target_mu=target_mu,
                        corruption_kind=corruption_kind,
                        noise_level=noise_level,
                        trial_seed=trial_seed,
                    )
                    for row in trial_rows:
                        row["experiment"] = "noise"
                        rows.append(row)
                    validation_rows.append(
                        {
                            "experiment": "noise",
                            "alphabet": alphabet_name,
                            "P": P_NOISE,
                            "corruption_kind": corruption_kind,
                            "noise_level": noise_level,
                            "trial": trial,
                            **weight_info,
                        }
                    )
                    print(
                        f"  {alphabet_name:13s} {corruption_kind:6s} level={noise_level:.3f} "
                        f"trial={trial + 1}/{N_TRIALS_NOISE}"
                    )

    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def run_load_experiment(
    alphabets: dict[str, np.ndarray],
    hue_bank: np.ndarray,
    saturation_bank: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    print("[Experiment 2] Running load curves...")
    for alphabet_name in ALPHABETS_TO_RUN:
        alphabet = alphabets[alphabet_name]
        for p in P_LOAD_VALUES:
            for trial in range(N_TRIALS_LOAD):
                target_mu = trial % p
                trial_seed = RANDOM_SEED + 200_000 + 1000 * hash_stable(alphabet_name) + 17 * p + trial
                trial_rows, weight_info = paired_recall_trial(
                    alphabet_name=alphabet_name,
                    alphabet=alphabet,
                    hue_bank=hue_bank,
                    saturation_bank=saturation_bank,
                    p=p,
                    target_mu=target_mu,
                    corruption_kind=LOAD_CORRUPTION_KIND,
                    noise_level=LOAD_NOISE_LEVEL,
                    trial_seed=trial_seed,
                )
                for row in trial_rows:
                    row["experiment"] = "load"
                    rows.append(row)
                validation_rows.append(
                    {
                        "experiment": "load",
                        "alphabet": alphabet_name,
                        "P": p,
                        "corruption_kind": LOAD_CORRUPTION_KIND,
                        "noise_level": LOAD_NOISE_LEVEL,
                        "trial": trial,
                        **weight_info,
                    }
                )
                print(f"  {alphabet_name:13s} P={p:2d} trial={trial + 1}/{N_TRIALS_LOAD}")

    return pd.DataFrame(rows), pd.DataFrame(validation_rows)


def hash_stable(text: str) -> int:
    """Small deterministic hash independent of Python's randomized hash()."""
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 100_000
    return value


# ====================
# Saving and summaries
# ====================


def prepare_output_dirs() -> tuple[Path, Path]:
    if OUTPUT_DIR.exists() and OVERWRITE_OUTPUT_DIR:
        raise FileExistsError(f"Output already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    images_dir = OUTPUT_DIR / "images"
    figures_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, images_dir


def save_dataframe(df: pd.DataFrame, base_path: Path) -> None:
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_csv(base_path.with_suffix(".txt"), index=False, sep="\t", float_format="%.10g")


def summarize_results(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = [
        "initial_SER",
        "SER",
        "exact_recovery",
        "complex_similarity",
        "radial_error",
        "angular_error_cycles",
        "rgb_psnr",
        "sweeps",
        "changed_updates",
        "converged",
        "lyapunov_sweep_violations",
        "delta_E_c",
        "delta_E_r",
    ]

    grouped = df.groupby(list(group_cols), dropna=False)
    summary = grouped[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part != "") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary = summary.rename(
        columns={
            "exact_recovery_mean": "exact_recovery_rate",
            "converged_mean": "convergence_rate",
        }
    )
    return summary.fillna(0.0)


def write_report(
    report_path: Path,
    alphabet_df: pd.DataFrame,
    noise_summary: pd.DataFrame,
    load_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    lines: list[str] = []
    lines.append("Experiment 2: HSV-encoded associative recall")
    lines.append("=" * 55)
    lines.append("")
    lines.append("Configuration")
    lines.append(f"- random_seed: {RANDOM_SEED}")
    lines.append(f"- image_shape: {IMAGE_SHAPE[0]} x {IMAGE_SHAPE[1]}")
    lines.append(f"- N: {IMAGE_SHAPE[0] * IMAGE_SHAPE[1]}")
    lines.append(f"- V0: {V0}")
    lines.append(f"- alphabets: {', '.join(ALPHABETS_TO_RUN)}")
    lines.append(f"- P_noise: {P_NOISE}")
    lines.append(f"- P_load_values: {P_LOAD_VALUES}")
    lines.append(f"- max_sweeps: {MAX_SWEEPS}")
    lines.append(f"- n_trials_noise: {N_TRIALS_NOISE}")
    lines.append(f"- n_trials_load: {N_TRIALS_LOAD}")
    lines.append(f"- elapsed_seconds: {elapsed_seconds:.2f}")
    lines.append("")

    lines.append("Alphabet summary")
    lines.append(alphabet_df.to_string(index=False))
    lines.append("")

    if not validation_df.empty:
        lines.append("Weight-matrix validation")
        lines.append(f"- maximum Hermitian error: {validation_df['hermitian_error'].max():.3e}")
        lines.append(f"- minimum real diagonal entry: {validation_df['min_diag_real'].min():.3e}")
        lines.append(f"- maximum projection residual: {validation_df['projection_residual'].max():.3e}")
        lines.append(f"- all diagonal entries nonnegative: {bool(validation_df['diag_nonnegative'].min())}")
        lines.append("")

    lines.append("Noise experiment: mean SER and radial error")
    noise_view_cols = [
        "alphabet",
        "corruption_kind",
        "noise_level",
        "rule",
        "SER_mean",
        "radial_error_mean",
        "exact_recovery_rate",
        "rgb_psnr_mean",
        "sweeps_mean",
    ]
    lines.append(noise_summary[noise_view_cols].to_string(index=False))
    lines.append("")

    lines.append("Load experiment: mean SER and exact recovery")
    load_view_cols = [
        "alphabet",
        "P",
        "P_over_N",
        "rule",
        "SER_mean",
        "exact_recovery_rate",
        "radial_error_mean",
        "rgb_psnr_mean",
        "sweeps_mean",
    ]
    lines.append(load_summary[load_view_cols].to_string(index=False))
    lines.append("")

    lines.append("Interpretation note")
    lines.append(
        "The experiment is not designed to force one rule to win. The paired design "
        "only ensures that both dynamics face the same memory matrix, corrupted "
        "state, and asynchronous schedule. The useful comparison is whether the "
        "Voronoi dynamics improves symbol retrieval and radial preservation in "
        "multi-amplitude alphabets, especially QAM, polar, and hexagonal alphabets."
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


# =========
# Plotting
# =========


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIG_DPI,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig: Any, figures_dir: Path, filename_stem: str) -> None:
    png_path = figures_dir / f"{filename_stem}.png"
    pdf_path = figures_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def plot_noise_metric(
    noise_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    figures_dir: Path,
    filename_stem: str,
) -> None:
    if not MAKE_FIGURES:
        return

    import matplotlib.pyplot as plt

    configure_matplotlib()
    corruption_order = ["symbol", "hsv"]
    rule_labels = {"classic": "classical", "voronoi": r"$\mathcal{D}_{\mathcal{S}}$"}
    rule_colors = {"classic": "#CB701B", "voronoi": "#2864A0"}

    fig, axes = plt.subplots(
        len(ALPHABETS_TO_RUN),
        len(corruption_order),
        figsize=(8.2, 2.15 * len(ALPHABETS_TO_RUN)),
        squeeze=False,
        sharey=False,
    )

    for row_idx, alphabet_name in enumerate(ALPHABETS_TO_RUN):
        for col_idx, corruption_kind in enumerate(corruption_order):
            ax = axes[row_idx, col_idx]
            subset = noise_summary[
                (noise_summary["alphabet"] == alphabet_name) & (noise_summary["corruption_kind"] == corruption_kind)
            ]
            for rule in ("classic", "voronoi"):
                rule_df = subset[subset["rule"] == rule].sort_values("noise_level")
                if rule_df.empty:
                    continue
                x = rule_df["noise_level"].to_numpy(dtype=float)
                y = rule_df[f"{metric}_mean"].to_numpy(dtype=float)
                ystd = rule_df.get(f"{metric}_std", pd.Series(np.zeros_like(y))).to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="s" if rule == "classic" else "o",
                    linestyle="-" if rule == "classic" else "--",
                    color=rule_colors[rule],
                    label=rule_labels[rule],
                )
                ax.fill_between(x, y - ystd, y + ystd, color=rule_colors[rule], alpha=0.12, linewidth=0)

            if row_idx == 0:
                ax.set_title("symbol noise" if corruption_kind == "symbol" else "HSV noise")
            if metric in ("SER", "radial_error"):
                ax.set_ylim(-0.03, 1.03)
            if col_idx == 0:
                ax.set_ylabel(f"{alphabet_name}\n{ylabel}")
            if row_idx == len(ALPHABETS_TO_RUN) - 1:
                ax.set_xlabel("noise level")
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    save_figure(fig, figures_dir, filename_stem)
    plt.close(fig)


def plot_load_exact_recovery(load_summary: pd.DataFrame, figures_dir: Path) -> None:
    if not MAKE_FIGURES:
        return

    import matplotlib.pyplot as plt

    configure_matplotlib()
    rule_labels = {"classic": "classical", "voronoi": r"$\mathcal{D}_{\mathcal{S}}$"}
    rule_colors = {"classic": "#CB701B", "voronoi": "#2864A0"}

    cols = 2
    rows = int(math.ceil(len(ALPHABETS_TO_RUN) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.4, 2.6 * rows), squeeze=False, sharey=True)
    for idx, alphabet_name in enumerate(ALPHABETS_TO_RUN):
        ax = axes.ravel()[idx]
        subset = load_summary[load_summary["alphabet"] == alphabet_name]
        for rule in ("classic", "voronoi"):
            rule_df = subset[subset["rule"] == rule].sort_values("P_over_N")
            if rule_df.empty:
                continue
            ax.plot(
                rule_df["P_over_N"],
                rule_df["exact_recovery_rate"],
                marker="s" if rule == "classic" else "o",
                linestyle="-" if rule == "classic" else "--",
                color=rule_colors[rule],
                label=rule_labels[rule],
            )
        ax.set_title(alphabet_name)
        ax.set_xlabel(r"load $P/N$")
        ax.set_ylim(-0.05, 1.05)
        if idx % cols == 0:
            ax.set_ylabel("exact recovery rate")
        if idx == 0:
            ax.legend(loc="best", frameon=False)

    for ax in axes.ravel()[len(ALPHABETS_TO_RUN) :]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, figures_dir, "experiment2_load_exact_recovery")
    plt.close(fig)


def save_rgb_image(path: Path, rgb: np.ndarray, upscale: int = IMAGE_UPSCALE) -> None:
    from PIL import Image

    arr = np.clip(np.asarray(rgb), 0.0, 1.0)
    arr_u8 = np.round(255.0 * arr).astype(np.uint8)
    image = Image.fromarray(arr_u8, mode="RGB")
    if upscale > 1:
        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST
        image = image.resize((image.width * upscale, image.height * upscale), resample=resample)
    image.save(path, dpi=(FIG_DPI, FIG_DPI))


def make_visual_example(
    alphabets: dict[str, np.ndarray],
    hue_bank: np.ndarray,
    saturation_bank: np.ndarray,
    figures_dir: Path,
    images_dir: Path,
) -> None:
    if not MAKE_FIGURES:
        return

    import matplotlib.pyplot as plt

    configure_matplotlib()
    alphabet = alphabets[VISUAL_ALPHABET]
    p = VISUAL_P
    target_mu = VISUAL_TARGET_MU % p
    memory_indices, memory_values = quantize_hsv_memories(hue_bank, saturation_bank, alphabet, p)
    w, weight_info = build_projection_weights(memory_values)

    n = memory_indices.shape[1]
    target_indices = memory_indices[target_mu]
    rng_initial = np.random.default_rng(RANDOM_SEED + 700_001)
    rng_schedule = np.random.default_rng(RANDOM_SEED + 800_001)
    initial_indices = make_initial_condition(
        target_indices=target_indices,
        target_hue=hue_bank[target_mu],
        target_saturation=saturation_bank[target_mu],
        alphabet=alphabet,
        corruption_kind=VISUAL_CORRUPTION_KIND,
        noise_level=VISUAL_NOISE_LEVEL,
        rng=rng_initial,
    )
    schedule = make_schedule(n, MAX_SWEEPS, rng_schedule)

    classic_result = run_dynamics(alphabet, w, initial_indices, schedule, "classic", MAX_SWEEPS)
    voronoi_result = run_dynamics(alphabet, w, initial_indices, schedule, "voronoi", MAX_SWEEPS)

    rmax = float(np.max(np.abs(alphabet)))
    images = {
        "stored_target": state_indices_to_rgb(target_indices, alphabet, IMAGE_SHAPE, rmax, V0),
        "corrupted_initial": state_indices_to_rgb(initial_indices, alphabet, IMAGE_SHAPE, rmax, V0),
        "classical_recall": state_indices_to_rgb(classic_result.final_indices, alphabet, IMAGE_SHAPE, rmax, V0),
        "voronoi_recall": state_indices_to_rgb(voronoi_result.final_indices, alphabet, IMAGE_SHAPE, rmax, V0),
    }

    for name, rgb in images.items():
        save_rgb_image(images_dir / f"experiment2_{name}.png", rgb)

    panel_items = [
        ("stored target", images["stored_target"]),
        ("corrupted", images["corrupted_initial"]),
        ("classical", images["classical_recall"]),
        (r"$\mathcal{D}_{\mathcal{S}}$", images["voronoi_recall"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.4))
    for ax, (title, rgb) in zip(axes, panel_items, strict=True):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"{VISUAL_ALPHABET}, P={p}, {VISUAL_CORRUPTION_KIND} noise={VISUAL_NOISE_LEVEL}",
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "experiment2_visual_panel")
    plt.close(fig)

    visual_rows = []
    for rule, result in (("classic", classic_result), ("voronoi", voronoi_result)):
        metrics = compute_metrics(result.final_indices, target_indices, alphabet, IMAGE_SHAPE)
        visual_rows.append(
            {
                "rule": rule,
                "alphabet": VISUAL_ALPHABET,
                "P": p,
                "target_mu": target_mu,
                "corruption_kind": VISUAL_CORRUPTION_KIND,
                "noise_level": VISUAL_NOISE_LEVEL,
                "initial_SER": float(np.mean(initial_indices != target_indices)),
                **metrics,
                "sweeps": result.sweeps,
                "changed_updates": result.changed_updates,
                "converged": float(result.converged),
                "lyapunov_sweep_violations": result.lyapunov_sweep_violations,
                "projection_residual": weight_info["projection_residual"],
            }
        )
    visual_df = pd.DataFrame(visual_rows)
    save_dataframe(visual_df, OUTPUT_DIR / "experiment2_visual_example_metrics")


def make_figures(
    noise_summary: pd.DataFrame,
    load_summary: pd.DataFrame,
    alphabets: dict[str, np.ndarray],
    hue_bank: np.ndarray,
    saturation_bank: np.ndarray,
    figures_dir: Path,
    images_dir: Path,
) -> None:
    if not MAKE_FIGURES:
        return

    plot_noise_metric(noise_summary, "SER", "SER", figures_dir, "experiment2_noise_SER")
    plot_noise_metric(noise_summary, "radial_error", "radial error", figures_dir, "experiment2_noise_radial_error")
    plot_load_exact_recovery(load_summary, figures_dir)
    make_visual_example(alphabets, hue_bank, saturation_bank, figures_dir, images_dir)

    if SAVE_OPTIONAL_PLOTS:
        plot_noise_metric(noise_summary, "rgb_psnr", "RGB PSNR", figures_dir, "experiment2_noise_rgb_psnr")
        plot_noise_metric(
            noise_summary,
            "complex_similarity",
            "complex similarity",
            figures_dir,
            "experiment2_noise_complex_similarity",
        )


# ====
# Main
# ====


def main() -> None:
    start_time = time.time()
    figures_dir, images_dir = prepare_output_dirs()

    alphabets = build_alphabets()
    selected_alphabets = {name: alphabets[name] for name in ALPHABETS_TO_RUN}
    alphabet_df = pd.DataFrame([alphabet_summary_row(name, alphabet) for name, alphabet in selected_alphabets.items()])
    save_dataframe(alphabet_df, OUTPUT_DIR / "experiment2_alphabet_summary")

    max_p = max(max(P_LOAD_VALUES), P_NOISE, VISUAL_P)
    hue_bank, saturation_bank = generate_synthetic_hsv_memories(max_p, IMAGE_SHAPE, seed=RANDOM_SEED)
    save_memory_bank_overview(hue_bank, saturation_bank, images_dir / "experiment2_memory_bank_overview.png")

    memory_report = OUTPUT_DIR / "experiment2_memory_bank_summary.txt"
    memory_report.write_text(
        "\n".join(
            [
                "Experiment 2 synthetic HSV memory bank",
                f"image_shape: {IMAGE_SHAPE[0]} x {IMAGE_SHAPE[1]}",
                f"N: {IMAGE_SHAPE[0] * IMAGE_SHAPE[1]}",
                f"number_of_generated_memories: {max_p}",
                f"V0: {V0}",
                "encoding: theta = 2*pi*h - pi, z = R_max*sigma*exp(i*theta)",
                "inverse: sigma_hat = min(|z|/R_max, 1), hue_hat = (Arg(z)+pi)/(2*pi) mod 1",
            ]
        ),
        encoding="utf-8",
    )

    noise_df, noise_validation_df = run_noise_experiment(alphabets, hue_bank, saturation_bank)
    load_df, load_validation_df = run_load_experiment(alphabets, hue_bank, saturation_bank)

    noise_summary = summarize_results(
        noise_df,
        ["alphabet", "alphabet_size", "N", "P", "P_over_N", "corruption_kind", "noise_level", "rule"],
    )
    load_summary = summarize_results(
        load_df,
        ["alphabet", "alphabet_size", "N", "P", "P_over_N", "corruption_kind", "noise_level", "rule"],
    )
    validation_df = pd.concat([noise_validation_df, load_validation_df], ignore_index=True)

    save_dataframe(noise_df, OUTPUT_DIR / "experiment2_noise_results")
    save_dataframe(noise_summary, OUTPUT_DIR / "experiment2_noise_summary")
    save_dataframe(load_df, OUTPUT_DIR / "experiment2_load_results")
    save_dataframe(load_summary, OUTPUT_DIR / "experiment2_load_summary")
    save_dataframe(validation_df, OUTPUT_DIR / "experiment2_weight_validation")

    make_figures(
        noise_summary,
        load_summary,
        alphabets,
        hue_bank,
        saturation_bank,
        figures_dir,
        images_dir,
    )

    elapsed_seconds = time.time() - start_time
    write_report(
        OUTPUT_DIR / "experiment2_numerical_report.txt",
        alphabet_df,
        noise_summary,
        load_summary,
        validation_df,
        elapsed_seconds,
    )

    zip_base = OUTPUT_DIR
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_base), "zip", OUTPUT_DIR)
    print("")
    print("[Experiment 2] Finished.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(f"Output archive: {zip_base.with_suffix('.zip').resolve()}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
