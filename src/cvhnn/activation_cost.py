"""
Experiment 3: computational cost of exact Voronoi activation.

This Colab-ready script compares four ways of evaluating

    D_S(z) = argmin_{s in S} |z - s|^2

for finite complex alphabets used in the paper:

1. exhaustive Euclidean distance search;
2. exhaustive inertial-score search;
3. exact KD-tree nearest-neighbor search in R^2;
4. exact structured quantizers when the constellation permits it.

The script uses two input sample sets:
- uniform points in a square containing the normalized constellation;
- local fields v_k = (W x)_k collected from the HSV associative-recall
  protocol of Experiment 2.

Anti-bias safeguards:
- All methods receive exactly the same input values z in each case.
- The exhaustive Euclidean method is the reference for agreement checks.
- Approximate lookup tables are not benchmarked as primary methods.
- Structured quantizers are only used for alphabet families where an exact
  geometry-specific decision rule is available.
- All timings and agreement values are saved; no run is selected or filtered
  after observing the results.

Outputs saved in experiment3_outputs/:
- experiment3_raw_timings.csv
- experiment3_summary.csv
- experiment3_summary.txt
- experiment3_cost_report.txt
- high-resolution 300 dpi figures in PNG and PDF
- experiment3_outputs.zip

Run it through the common launcher, which sets the output folder and the
smoke-mode sizes (cvhnn.config.SMOKE_OVERRIDES):

    python experiments/03_activation_cost.py --mode full --output runs/activation_cost
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
    SCIPY_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on Colab environment.
    cKDTree = None
    SCIPY_AVAILABLE = False
    SCIPY_IMPORT_ERROR = repr(exc)


# =========================
# User-facing configuration
# =========================

OUTPUT_DIR = Path("experiment3_outputs")
RANDOM_SEED = 20260901

FAMILIES = ("PSK", "QAM", "Polar", "Hexagonal")
CARDINALITIES = (16, 64, 256, 1024)

N_QUERY_UNIFORM = 20000
N_QUERY_FIELDS = 20000
N_REPEATS = 5
CHUNK_SIZE = 4096

# Experiment-2-like local-field generation.
IMAGE_SHAPE = (24, 24)
P_FIELDS = 4
FIELD_HSV_NOISE = 0.10
FIELD_SWEEPS = 3
MIN_FIELD_TRIALS = 10
PINV_RCOND = 1e-10

KD_WORKERS = 1

FIG_DPI = 300
MAKE_FIGURES = True
OVERWRITE_OUTPUT_DIR = True


# =====================
# Alphabet construction
# =====================


def sort_alphabet(points: np.ndarray) -> np.ndarray:
    """Sort by modulus and then principal argument, fixing a total order."""
    points = np.asarray(points, dtype=np.complex128)
    keys = np.lexsort((np.angle(points), np.abs(points)))
    return points[keys]


def normalize_alphabet(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.complex128)
    rmax = float(np.max(np.abs(points)))
    if rmax <= 0.0:
        raise ValueError("The alphabet must contain at least one nonzero point.")
    return sort_alphabet(points / rmax)


def make_psk(m: int) -> np.ndarray:
    phases = 2.0 * np.pi * np.arange(m) / m
    return normalize_alphabet(np.exp(1j * phases))


def make_qam(m: int) -> np.ndarray:
    side = int(round(math.sqrt(m)))
    if side * side != m:
        raise ValueError(f"QAM cardinality must be a square, got {m}.")
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    xx, yy = np.meshgrid(levels, levels)
    return normalize_alphabet(xx.ravel() + 1j * yy.ravel())


def make_polar(m: int) -> np.ndarray:
    side = int(round(math.sqrt(m)))
    if side * side != m:
        raise ValueError(f"Polar cardinality must be a square, got {m}.")
    radii = np.linspace(1.0 / side, 1.0, side)
    phases = 2.0 * np.pi * np.arange(side) / side
    points = [radius * np.exp(1j * phase) for radius in radii for phase in phases]
    return normalize_alphabet(np.array(points, dtype=np.complex128))


def make_hexagonal(m: int) -> np.ndarray:
    """Return the m closest points from an Eisenstein-lattice patch."""
    search_radius = max(3, int(math.ceil(math.sqrt(m) / 2.0)) + 3)
    omega = np.exp(1j * np.pi / 3.0)
    while True:
        points: list[complex] = []
        for a in range(-search_radius, search_radius + 1):
            for b in range(-search_radius, search_radius + 1):
                points.append(a + b * omega)
        patch = np.array(points, dtype=np.complex128)
        if len(patch) >= m:
            order = np.lexsort((np.angle(patch), np.abs(patch)))
            return normalize_alphabet(patch[order[:m]])
        search_radius += 2


def make_alphabet(family: str, m: int) -> np.ndarray:
    from .ordered_alphabets import load_group

    key = f"{family}:{m}"
    published = load_group("activation_cost")
    if key in published:
        return published[key]
    return generate_alphabet(family, m)


def generate_alphabet(family: str, m: int) -> np.ndarray:
    if family == "PSK":
        return make_psk(m)
    if family == "QAM":
        return make_qam(m)
    if family == "Polar":
        return make_polar(m)
    if family == "Hexagonal":
        return make_hexagonal(m)
    raise ValueError(f"Unknown family: {family}")


# ======================
# Exact query primitives
# ======================


def as_points(z: np.ndarray) -> np.ndarray:
    flat = np.asarray(z, dtype=np.complex128).reshape(-1)
    return np.column_stack((flat.real, flat.imag))


def circular_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a - b)
    return np.minimum(diff, 2.0 * np.pi - diff)


def nearest_positions_sorted(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Nearest level index for sorted one-dimensional levels."""
    values = np.asarray(values, dtype=float)
    levels = np.asarray(levels, dtype=float)
    pos = np.searchsorted(levels, values, side="left")
    lo = np.clip(pos - 1, 0, len(levels) - 1)
    hi = np.clip(pos, 0, len(levels) - 1)
    lo_dist = np.abs(values - levels[lo])
    hi_dist = np.abs(values - levels[hi])
    return np.where(hi_dist < lo_dist, hi, lo)


@dataclass
class BuiltMethod:
    name: str
    query: Callable[[np.ndarray], np.ndarray]
    build_time_s: float
    aux_memory_bytes: int
    flops_per_query_est: float
    comparisons_per_query_est: float
    transcendental_calls_per_query: float
    exact_by_construction: bool
    notes: str


def build_distance_method(alphabet: np.ndarray) -> BuiltMethod:
    t0 = time.perf_counter()
    sr = np.real(alphabet).astype(np.float64)
    si = np.imag(alphabet).astype(np.float64)
    build_time = time.perf_counter() - t0
    m = len(alphabet)

    def query(z: np.ndarray) -> np.ndarray:
        flat = np.asarray(z, dtype=np.complex128).reshape(-1)
        out = np.empty(len(flat), dtype=np.int64)
        for start in range(0, len(flat), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(flat))
            zr = flat[start:end].real[:, None]
            zi = flat[start:end].imag[:, None]
            distances = (zr - sr[None, :]) ** 2 + (zi - si[None, :]) ** 2
            out[start:end] = np.argmin(distances, axis=1)
        return out

    return BuiltMethod(
        name="distance",
        query=query,
        build_time_s=build_time,
        aux_memory_bytes=sr.nbytes + si.nbytes,
        flops_per_query_est=5.0 * m,
        comparisons_per_query_est=float(m - 1),
        transcendental_calls_per_query=0.0,
        exact_by_construction=True,
        notes="Exhaustive Euclidean nearest-neighbor search.",
    )


def build_score_method(alphabet: np.ndarray) -> BuiltMethod:
    t0 = time.perf_counter()
    sr = np.real(alphabet).astype(np.float64)
    si = np.imag(alphabet).astype(np.float64)
    offsets = 0.5 * (sr**2 + si**2)
    build_time = time.perf_counter() - t0
    m = len(alphabet)

    def query(z: np.ndarray) -> np.ndarray:
        flat = np.asarray(z, dtype=np.complex128).reshape(-1)
        out = np.empty(len(flat), dtype=np.int64)
        for start in range(0, len(flat), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(flat))
            zr = flat[start:end].real[:, None]
            zi = flat[start:end].imag[:, None]
            scores = zr * sr[None, :] + zi * si[None, :] - offsets[None, :]
            out[start:end] = np.argmax(scores, axis=1)
        return out

    return BuiltMethod(
        name="inertial_score",
        query=query,
        build_time_s=build_time,
        aux_memory_bytes=sr.nbytes + si.nbytes + offsets.nbytes,
        flops_per_query_est=4.0 * m,
        comparisons_per_query_est=float(m - 1),
        transcendental_calls_per_query=0.0,
        exact_by_construction=True,
        notes="Exhaustive maximization of Re(conj(s)z)-|s|^2/2.",
    )


def build_kdtree_method(alphabet: np.ndarray) -> BuiltMethod | None:
    if not SCIPY_AVAILABLE:
        return None
    t0 = time.perf_counter()
    points = as_points(alphabet)
    tree = cKDTree(points)
    build_time = time.perf_counter() - t0
    m = len(alphabet)
    proxy_visits = max(1.0, math.ceil(math.log2(m)))

    def query(z: np.ndarray) -> np.ndarray:
        query_points = as_points(z)
        _, indices = tree.query(query_points, k=1, workers=KD_WORKERS)
        return np.asarray(indices, dtype=np.int64)

    return BuiltMethod(
        name="kdtree",
        query=query,
        build_time_s=build_time,
        aux_memory_bytes=int(points.nbytes + 64 * m),
        flops_per_query_est=5.0 * proxy_visits,
        comparisons_per_query_est=proxy_visits,
        transcendental_calls_per_query=0.0,
        exact_by_construction=True,
        notes="Exact cKDTree query in R^2; operation count is a log-size proxy.",
    )


def build_structured_psk(alphabet: np.ndarray) -> BuiltMethod:
    t0 = time.perf_counter()
    m = len(alphabet)
    phases = (np.angle(alphabet) + 2.0 * np.pi) % (2.0 * np.pi)
    order = np.argsort(phases)
    sorted_phases = phases[order]
    build_time = time.perf_counter() - t0

    def query(z: np.ndarray) -> np.ndarray:
        flat = np.asarray(z, dtype=np.complex128).reshape(-1)
        phi = (np.angle(flat) + 2.0 * np.pi) % (2.0 * np.pi)
        pos = np.searchsorted(sorted_phases, phi, side="left")
        cand_a = order[np.mod(pos, m)]
        cand_b = order[np.mod(pos - 1, m)]
        dist_a = np.abs(flat - alphabet[cand_a]) ** 2
        dist_b = np.abs(flat - alphabet[cand_b]) ** 2
        choose_a = (dist_a < dist_b) | ((dist_a == dist_b) & (cand_a < cand_b))
        out = np.where(choose_a, cand_a, cand_b).astype(np.int64)
        out[np.abs(flat) <= 1e-14] = 0
        return out

    return BuiltMethod(
        name="structured",
        query=query,
        build_time_s=build_time,
        aux_memory_bytes=int(phases.nbytes + order.nbytes),
        flops_per_query_est=10.0,
        comparisons_per_query_est=2.0,
        transcendental_calls_per_query=1.0,
        exact_by_construction=True,
        notes="PSK angular quantizer; one atan2-equivalent phase evaluation.",
    )


def build_structured_qam(alphabet: np.ndarray) -> BuiltMethod:
    t0 = time.perf_counter()
    re_levels = np.unique(np.real(alphabet))
    im_levels = np.unique(np.imag(alphabet))
    re_levels.sort()
    im_levels.sort()
    grid = np.empty((len(re_levels), len(im_levels)), dtype=np.int64)
    for idx, s in enumerate(alphabet):
        ri = int(np.argmin(np.abs(re_levels - np.real(s))))
        ii = int(np.argmin(np.abs(im_levels - np.imag(s))))
        grid[ri, ii] = idx
    build_time = time.perf_counter() - t0

    def query(z: np.ndarray) -> np.ndarray:
        flat = np.asarray(z, dtype=np.complex128).reshape(-1)
        ri = nearest_positions_sorted(flat.real, re_levels)
        ii = nearest_positions_sorted(flat.imag, im_levels)
        return grid[ri, ii].astype(np.int64)

    return BuiltMethod(
        name="structured",
        query=query,
        build_time_s=build_time,
        aux_memory_bytes=int(re_levels.nbytes + im_levels.nbytes + grid.nbytes),
        flops_per_query_est=12.0,
        comparisons_per_query_est=4.0,
        transcendental_calls_per_query=0.0,
        exact_by_construction=True,
        notes="Square QAM coordinate quantizer.",
    )


def build_structured_polar(alphabet: np.ndarray) -> BuiltMethod:
    t0 = time.perf_counter()
    radii_all = np.abs(alphabet)
    phases_all = (np.angle(alphabet) + 2.0 * np.pi) % (2.0 * np.pi)
    radius_levels = np.unique(np.round(radii_all, 14))
    phase_levels = np.unique(np.round(phases_all, 14))
    radius_levels.sort()
    phase_levels.sort()
    grid = np.empty((len(radius_levels), len(phase_levels)), dtype=np.int64)
    for idx, s in enumerate(alphabet):
        ri = int(np.argmin(np.abs(radius_levels - np.abs(s))))
        pi = int(np.argmin(circular_abs_diff(phase_levels, (np.angle(s) + 2.0 * np.pi) % (2.0 * np.pi))))
        grid[ri, pi] = idx
    min_radius_idx = int(np.argmin(radii_all))
    build_time = time.perf_counter() - t0

    def query(z: np.ndarray) -> np.ndarray:
        flat = np.asarray(z, dtype=np.complex128).reshape(-1)
        r = np.abs(flat)
        phi = (np.angle(flat) + 2.0 * np.pi) % (2.0 * np.pi)
        phase_pos = np.searchsorted(phase_levels, phi, side="left")
        phase_candidates = np.stack(
            [np.mod(phase_pos - 1, len(phase_levels)), np.mod(phase_pos, len(phase_levels))],
            axis=1,
        )
        best_dist = np.full(len(flat), np.inf, dtype=float)
        best_idx = np.zeros(len(flat), dtype=np.int64)
        for col in range(phase_candidates.shape[1]):
            pi = phase_candidates[:, col]
            phase = phase_levels[pi]
            delta = circular_abs_diff(phi, phase)
            target_radius = r * np.cos(delta)
            radius_pos = np.searchsorted(radius_levels, target_radius, side="left")
            for radius_choice in (radius_pos - 1, radius_pos):
                ri = np.clip(radius_choice, 0, len(radius_levels) - 1)
                cand_idx = grid[ri, pi]
                cand = alphabet[cand_idx]
                dist = np.abs(flat - cand) ** 2
                update = (dist < best_dist) | ((dist == best_dist) & (cand_idx < best_idx))
                best_dist = np.where(update, dist, best_dist)
                best_idx = np.where(update, cand_idx, best_idx)
        best_idx[r <= 1e-14] = min_radius_idx
        return best_idx.astype(np.int64)

    return BuiltMethod(
        name="structured",
        query=query,
        build_time_s=build_time,
        aux_memory_bytes=int(radius_levels.nbytes + phase_levels.nbytes + grid.nbytes),
        flops_per_query_est=32.0,
        comparisons_per_query_est=8.0,
        transcendental_calls_per_query=3.0,
        exact_by_construction=True,
        notes="Polar-grid candidate search using nearest phases and radii.",
    )


def build_structured_method(family: str, alphabet: np.ndarray) -> BuiltMethod | None:
    if family == "PSK":
        return build_structured_psk(alphabet)
    if family == "QAM":
        return build_structured_qam(alphabet)
    if family == "Polar":
        return build_structured_polar(alphabet)
    return None


def build_methods(family: str, alphabet: np.ndarray) -> list[BuiltMethod]:
    methods = [build_distance_method(alphabet), build_score_method(alphabet)]
    kd_method = build_kdtree_method(alphabet)
    if kd_method is not None:
        methods.append(kd_method)
    structured = build_structured_method(family, alphabet)
    if structured is not None:
        methods.append(structured)
    return methods


# ================================
# Experiment-2-like local fields
# ================================


def hsv_fields_to_complex(hue: np.ndarray, saturation: np.ndarray, rmax: float = 1.0) -> np.ndarray:
    theta = 2.0 * np.pi * (np.asarray(hue) % 1.0) - np.pi
    sigma = np.clip(np.asarray(saturation, dtype=float), 0.0, 1.0)
    return rmax * sigma * np.exp(1j * theta)


def generate_synthetic_hsv_memories(
    num_memories: int,
    image_shape: tuple[int, int],
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
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


def quantize_complex_to_indices(z: np.ndarray, alphabet: np.ndarray) -> np.ndarray:
    original_shape = np.asarray(z).shape
    return build_distance_method(alphabet).query(np.asarray(z).reshape(-1)).reshape(original_shape)


def build_projection_weights(memory_values: np.ndarray) -> np.ndarray:
    x_matrix = memory_values.T.astype(np.complex128)
    gram = x_matrix.conj().T @ x_matrix
    gram_pinv = np.linalg.pinv(gram, rcond=PINV_RCOND)
    w = x_matrix @ gram_pinv @ x_matrix.conj().T
    return ((w + w.conj().T) / 2.0).astype(np.complex128)


def make_hsv_corrupted_indices(
    target_hue: np.ndarray,
    target_sat: np.ndarray,
    alphabet: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbed_hue = (target_hue + rng.normal(0.0, noise_level, size=target_hue.shape)) % 1.0
    perturbed_sat = np.clip(target_sat + rng.normal(0.0, noise_level, size=target_sat.shape), 0.0, 1.0)
    z = hsv_fields_to_complex(perturbed_hue, perturbed_sat, rmax=float(np.max(np.abs(alphabet))))
    return quantize_complex_to_indices(z, alphabet).reshape(-1)


def collect_local_fields(
    alphabet: np.ndarray,
    q: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    snapshots_per_trial = FIELD_SWEEPS + 1
    n_trials = max(MIN_FIELD_TRIALS, int(math.ceil(q / (n * snapshots_per_trial))))

    hue_bank, sat_bank = generate_synthetic_hsv_memories(P_FIELDS, IMAGE_SHAPE, seed=RANDOM_SEED)
    z_bank = hsv_fields_to_complex(hue_bank[:P_FIELDS], sat_bank[:P_FIELDS], rmax=float(np.max(np.abs(alphabet))))
    memory_indices = quantize_complex_to_indices(z_bank, alphabet).reshape(P_FIELDS, -1)
    memory_values = alphabet[memory_indices]
    w = build_projection_weights(memory_values)

    fields: list[np.ndarray] = []
    for trial in range(n_trials):
        target_mu = trial % P_FIELDS
        initial_indices = make_hsv_corrupted_indices(
            hue_bank[target_mu],
            sat_bank[target_mu],
            alphabet,
            FIELD_HSV_NOISE,
            rng,
        )
        indices = initial_indices.copy()
        state = alphabet[indices].astype(np.complex128)
        field = w @ state
        fields.append(field.copy())

        for _ in range(FIELD_SWEEPS):
            for k in rng.permutation(n):
                current_idx = int(indices[k])
                local_field = field[k]
                new_idx = int(np.argmin(np.abs(alphabet - local_field) ** 2))
                if new_idx != current_idx:
                    old_value = state[k]
                    new_value = alphabet[new_idx]
                    delta = new_value - old_value
                    indices[k] = new_idx
                    state[k] = new_value
                    field += w[:, k] * delta
            field = w @ state
            fields.append(field.copy())
            if sum(len(arr) for arr in fields) >= q:
                break
        if sum(len(arr) for arr in fields) >= q:
            break

    flat_fields = np.concatenate(fields)
    if len(flat_fields) > q:
        selected = rng.choice(len(flat_fields), size=q, replace=False)
        flat_fields = flat_fields[selected]

    info = {
        "field_trials": n_trials,
        "field_snapshots": len(fields),
        "field_values_collected": int(len(flat_fields)),
        "network_size_N": n,
        "stored_memories_P": P_FIELDS,
        "hsv_noise_level": FIELD_HSV_NOISE,
    }
    return flat_fields.astype(np.complex128), info


def generate_uniform_inputs(alphabet: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    rmax = float(np.max(np.abs(alphabet)))
    limit = 1.25 * rmax
    return rng.uniform(-limit, limit, size=q) + 1j * rng.uniform(-limit, limit, size=q)


# ====================
# Benchmark and output
# ====================


def activation_field_ratio(flops_per_query: float) -> float:
    n = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    return flops_per_query / (8.0 * n)


def time_method(method: BuiltMethod, z: np.ndarray, reference: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    # Warm-up outside the measured repetitions.
    warmup_n = min(512, len(z))
    method.query(z[:warmup_n])

    rows: list[dict[str, Any]] = []
    first_indices: np.ndarray | None = None
    for repeat in range(N_REPEATS):
        t0 = time.perf_counter()
        indices = method.query(z)
        elapsed = time.perf_counter() - t0
        if first_indices is None:
            first_indices = indices.copy()
        agreement = float(np.mean(indices == reference))
        rows.append(
            {
                "method": method.name,
                "repeat": repeat,
                "query_count": int(len(z)),
                "elapsed_s": elapsed,
                "time_per_query_us": 1e6 * elapsed / max(len(z), 1),
                "queries_per_second": len(z) / max(elapsed, 1e-12),
                "agreement_with_distance": agreement,
                "build_time_s": method.build_time_s,
                "aux_memory_bytes": method.aux_memory_bytes,
                "flops_per_query_est": method.flops_per_query_est,
                "comparisons_per_query_est": method.comparisons_per_query_est,
                "transcendental_calls_per_query": method.transcendental_calls_per_query,
                "activation_field_ratio_est": activation_field_ratio(method.flops_per_query_est),
                "exact_by_construction": method.exact_by_construction,
                "notes": method.notes,
            }
        )
    assert first_indices is not None
    return rows, first_indices


def benchmark_case(
    family: str,
    m: int,
    ensemble_name: str,
    z: np.ndarray,
) -> list[dict[str, Any]]:
    alphabet = make_alphabet(family, m)
    methods = build_methods(family, alphabet)
    reference_method = methods[0]
    reference = reference_method.query(z)

    rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows, _ = time_method(method, z, reference)
        for row in method_rows:
            row.update(
                {
                    "family": family,
                    "cardinality": m,
                    "ensemble": ensemble_name,
                    "scipy_available": SCIPY_AVAILABLE,
                }
            )
        rows.extend(method_rows)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ensemble", "family", "cardinality", "method"]
    summary = (
        raw.groupby(group_cols, as_index=False)
        .agg(
            query_count=("query_count", "first"),
            time_per_query_us_mean=("time_per_query_us", "mean"),
            time_per_query_us_std=("time_per_query_us", "std"),
            queries_per_second_mean=("queries_per_second", "mean"),
            agreement_min=("agreement_with_distance", "min"),
            agreement_mean=("agreement_with_distance", "mean"),
            build_time_s=("build_time_s", "first"),
            aux_memory_bytes=("aux_memory_bytes", "first"),
            flops_per_query_est=("flops_per_query_est", "first"),
            comparisons_per_query_est=("comparisons_per_query_est", "first"),
            transcendental_calls_per_query=("transcendental_calls_per_query", "first"),
            activation_field_ratio_est=("activation_field_ratio_est", "first"),
            exact_by_construction=("exact_by_construction", "first"),
            notes=("notes", "first"),
        )
        .sort_values(["ensemble", "family", "cardinality", "method"])
    )

    baseline = summary[summary["method"] == "distance"][
        ["ensemble", "family", "cardinality", "time_per_query_us_mean"]
    ].rename(columns={"time_per_query_us_mean": "distance_time_per_query_us_mean"})
    summary = summary.merge(baseline, on=["ensemble", "family", "cardinality"], how="left")
    summary["speedup_vs_distance"] = summary["distance_time_per_query_us_mean"] / summary["time_per_query_us_mean"]
    return summary


def save_text_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    readable_cols = [
        "ensemble",
        "family",
        "cardinality",
        "method",
        "time_per_query_us_mean",
        "time_per_query_us_std",
        "speedup_vs_distance",
        "agreement_min",
        "build_time_s",
        "aux_memory_bytes",
        "flops_per_query_est",
        "comparisons_per_query_est",
        "transcendental_calls_per_query",
        "activation_field_ratio_est",
    ]
    text = summary[readable_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
    (out_dir / "experiment3_summary.txt").write_text(text, encoding="utf-8")


def write_report(summary: pd.DataFrame, field_infos: list[dict[str, Any]], out_dir: Path) -> None:
    lines: list[str] = []
    n = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    lines.append("Experiment 3: computational cost of exact Voronoi activation")
    lines.append("=" * 68)
    lines.append("")
    lines.append("Configuration")
    lines.append(f"- families: {', '.join(FAMILIES)}")
    lines.append(f"- cardinalities: {', '.join(str(m) for m in CARDINALITIES)}")
    lines.append(f"- uniform query count per case: {N_QUERY_UNIFORM}")
    lines.append(f"- local-field query count per case: {N_QUERY_FIELDS}")
    lines.append(f"- timing repetitions per method: {N_REPEATS}")
    lines.append(f"- Experiment-2-like local-field network size: N={n}")
    lines.append(f"- local-field stored memories: P={P_FIELDS}")
    lines.append(f"- local-field HSV noise level: {FIELD_HSV_NOISE}")
    lines.append(f"- SciPy/cKDTree available: {SCIPY_AVAILABLE}")
    if not SCIPY_AVAILABLE:
        lines.append(f"- SciPy import error: {SCIPY_IMPORT_ERROR}")
    lines.append("")
    lines.append("Method interpretation")
    lines.append("- distance: exact exhaustive Euclidean search, reference method.")
    lines.append("- inertial_score: exact exhaustive score Re(conj(s)z)-|s|^2/2.")
    lines.append("- kdtree: exact nearest-neighbor query in R^2 when SciPy is available.")
    lines.append("- structured: exact geometry-specific quantizer for PSK, QAM, and Polar.")
    lines.append("- approximate lookup tables are intentionally not benchmarked as primary methods.")
    lines.append("")

    low_agreement = summary[summary["agreement_min"] < 1.0]
    lines.append("Agreement check")
    if low_agreement.empty:
        lines.append("- All benchmarked methods agreed with exhaustive Euclidean search in all measured queries.")
    else:
        lines.append("- Some methods disagreed with the exhaustive reference; inspect these rows:")
        lines.append(
            low_agreement[["ensemble", "family", "cardinality", "method", "agreement_min"]].to_string(
                index=False, float_format=lambda x: f"{x:.9g}"
            )
        )
    lines.append("")

    lines.append("Fastest method by case")
    fastest = summary.loc[summary.groupby(["ensemble", "family", "cardinality"])["time_per_query_us_mean"].idxmin()][
        [
            "ensemble",
            "family",
            "cardinality",
            "method",
            "time_per_query_us_mean",
            "speedup_vs_distance",
            "agreement_min",
        ]
    ].sort_values(["ensemble", "family", "cardinality"])
    lines.append(fastest.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    lines.append("")

    if field_infos:
        lines.append("Local-field generation diagnostics")
        field_df = pd.DataFrame(field_infos)
        lines.append(field_df.to_string(index=False))
        lines.append("")

    lines.append("Cost model")
    lines.append("- Exhaustive distance search: approximately 5|S| flops and |S|-1 comparisons per query.")
    lines.append("- Exhaustive inertial-score search: approximately 4|S| flops and |S|-1 comparisons per query.")
    lines.append(
        "- For dense incremental asynchronous updates, maintaining the local field costs "
        "about 8N flops per neuron update."
    )
    lines.append("- The reported activation_field_ratio_est is flops_per_query_est/(8N).")
    lines.append("")
    lines.append("Full numerical tables are saved in CSV and TXT files in this folder.")

    (out_dir / "experiment3_cost_report.txt").write_text("\n".join(lines), encoding="utf-8")


def method_style(method: str) -> tuple[str, str]:
    styles = {
        "distance": ("#b23a48", "o"),
        "inertial_score": ("#8b5a2b", "s"),
        "kdtree": ("#2468ac", "^"),
        "structured": ("#17877d", "D"),
    }
    return styles.get(method, ("#333333", "o"))


def save_time_figure(summary: pd.DataFrame, ensemble: str, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sub = summary[summary["ensemble"] == ensemble]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    methods = ["distance", "inertial_score", "kdtree", "structured"]
    for ax, family in zip(axes.ravel(), FAMILIES, strict=True):
        fam = sub[sub["family"] == family]
        for method in methods:
            data = fam[fam["method"] == method].sort_values("cardinality")
            if data.empty:
                continue
            color, marker = method_style(method)
            ax.plot(
                data["cardinality"],
                data["time_per_query_us_mean"],
                marker=marker,
                linewidth=1.8,
                color=color,
                label=method,
            )
            yerr = data["time_per_query_us_std"].fillna(0.0)
            ax.fill_between(
                data["cardinality"].to_numpy(dtype=float),
                (data["time_per_query_us_mean"] - yerr).clip(lower=1e-12).to_numpy(dtype=float),
                (data["time_per_query_us_mean"] + yerr).to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(family)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xticks(list(CARDINALITIES))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1, 0].set_xlabel(r"$|\mathcal{S}|$")
    axes[1, 1].set_xlabel(r"$|\mathcal{S}|$")
    axes[0, 0].set_ylabel("time per query (us)")
    axes[1, 0].set_ylabel("time per query (us)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    title = "Uniform inputs" if ensemble == "uniform" else "Experiment-2 local fields"
    fig.suptitle(f"Experiment 3: activation time, {title}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    stem = f"experiment3_time_{ensemble}"
    fig.savefig(out_dir / f"{stem}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_ratio_figure(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    m_values = np.array(CARDINALITIES, dtype=float)
    n = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    distance_ratio = 5.0 * m_values / (8.0 * n)
    score_ratio = 4.0 * m_values / (8.0 * n)

    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    ax.plot(m_values, distance_ratio, marker="o", color="#b23a48", label="distance")
    ax.plot(m_values, score_ratio, marker="s", color="#8b5a2b", label="inertial score")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="field cost")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(list(CARDINALITIES))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$|\mathcal{S}|$")
    ax.set_ylabel(r"activation / dense-field cost")
    ax.set_title(r"Estimated cost ratio for one asynchronous sweep, $N=576$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "experiment3_activation_field_ratio.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_dir / "experiment3_activation_field_ratio.pdf", bbox_inches="tight")
    plt.close(fig)


def save_figures(summary: pd.DataFrame, out_dir: Path) -> None:
    if not MAKE_FIGURES:
        return
    save_time_figure(summary, "uniform", out_dir)
    save_time_figure(summary, "local_fields", out_dir)
    save_ratio_figure(out_dir)


def prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists() and OVERWRITE_OUTPUT_DIR:
        raise FileExistsError(f"Output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def save_configuration(out_dir: Path) -> None:
    config = {
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "RANDOM_SEED": RANDOM_SEED,
        "FAMILIES": FAMILIES,
        "CARDINALITIES": CARDINALITIES,
        "N_QUERY_UNIFORM": N_QUERY_UNIFORM,
        "N_QUERY_FIELDS": N_QUERY_FIELDS,
        "N_REPEATS": N_REPEATS,
        "CHUNK_SIZE": CHUNK_SIZE,
        "IMAGE_SHAPE": IMAGE_SHAPE,
        "P_FIELDS": P_FIELDS,
        "FIELD_HSV_NOISE": FIELD_HSV_NOISE,
        "FIELD_SWEEPS": FIELD_SWEEPS,
        "MIN_FIELD_TRIALS": MIN_FIELD_TRIALS,
        "KD_WORKERS": KD_WORKERS,
        "SCIPY_AVAILABLE": SCIPY_AVAILABLE,
        "SCIPY_IMPORT_ERROR": SCIPY_IMPORT_ERROR,
    }
    pd.Series(config).to_csv(out_dir / "experiment3_configuration.csv", header=["value"])
    (out_dir / "experiment3_configuration.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in config.items()),
        encoding="utf-8",
    )


def main() -> None:
    prepare_output_dir(OUTPUT_DIR)
    save_configuration(OUTPUT_DIR)

    raw_rows: list[dict[str, Any]] = []
    field_infos: list[dict[str, Any]] = []

    for family_idx, family in enumerate(FAMILIES):
        for m in CARDINALITIES:
            alphabet = make_alphabet(family, m)
            case_seed = RANDOM_SEED + 100_000 * family_idx + m
            rng = np.random.default_rng(case_seed)

            print(f"[Experiment 3] {family}, |S|={m}: uniform inputs")
            z_uniform = generate_uniform_inputs(alphabet, N_QUERY_UNIFORM, rng)
            raw_rows.extend(benchmark_case(family, m, "uniform", z_uniform))

            print(f"[Experiment 3] {family}, |S|={m}: Experiment-2 local fields")
            z_fields, field_info = collect_local_fields(alphabet, N_QUERY_FIELDS, seed=case_seed + 17_000)
            field_info.update({"family": family, "cardinality": m})
            field_infos.append(field_info)
            raw_rows.extend(benchmark_case(family, m, "local_fields", z_fields))

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)

    raw.to_csv(OUTPUT_DIR / "experiment3_raw_timings.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "experiment3_summary.csv", index=False)
    if field_infos:
        pd.DataFrame(field_infos).to_csv(OUTPUT_DIR / "experiment3_local_field_diagnostics.csv", index=False)
    save_text_tables(summary, OUTPUT_DIR)
    write_report(summary, field_infos, OUTPUT_DIR)
    save_figures(summary, OUTPUT_DIR)

    try:
        script_path = Path(__file__)
    except NameError:
        script_path = None

    if script_path is not None and script_path.exists():
        shutil.copy2(script_path, OUTPUT_DIR / script_path.name)
    else:
        (OUTPUT_DIR / "experiment3_notebook_execution_note.txt").write_text(
            "This run was executed from an interactive notebook cell, so __file__ "
            "was not defined and the script could not copy itself into the output "
            "folder. All numerical outputs and figures were still generated.\n",
            encoding="utf-8",
        )

    zip_base = str(OUTPUT_DIR)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=OUTPUT_DIR)
    print(f"[Experiment 3] Done. Outputs: {OUTPUT_DIR.resolve()}")
    print(f"[Experiment 3] Zip archive: {Path(zip_path).resolve()}")


if __name__ == "__main__":
    main()
