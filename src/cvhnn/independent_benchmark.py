"""Cardinality experiment: independent kernels, conservative ties, shared inputs."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import independent as core
from .independent_search import Search, boundary_validation


def local_fields(m, alphabet, q, seed):
    rng = np.random.default_rng(seed)
    hue, sat = m.generate_synthetic_hsv_memories(max(6, m.P_FIELDS), m.IMAGE_SHAPE, seed=m.RANDOM_SEED)
    hue, sat = hue[: m.P_FIELDS], sat[: m.P_FIELDS]
    memory = core.quantize(core.encode(hue, sat, np.abs(alphabet).max()), alphabet).reshape(m.P_FIELDS, -1)
    w, _ = core.projection_weights(alphabet[memory])
    values = []
    currents = []
    trials = 0
    while sum(map(len, values)) < q:
        target = trials % m.P_FIELDS
        idx = core.corrupt_hsv(hue[target], sat[target], alphabet, m.FIELD_HSV_NOISE, rng)
        state = alphabet[idx].copy()
        field = w @ state
        values.append(field.copy())
        currents.append(idx.copy())
        for _ in range(m.FIELD_SWEEPS):
            for k in rng.permutation(len(idx)):
                new = core.decision(alphabet, field[k], int(idx[k]), relative=False)
                if new != idx[k]:
                    delta = alphabet[new] - state[k]
                    state[k] = alphabet[new]
                    idx[k] = new
                    field += w[:, k] * delta
            field = w @ state
            values.append(field.copy())
            currents.append(idx.copy())
            if sum(map(len, values)) >= q:
                break
        trials += 1
    z = np.concatenate(values)
    current = np.concatenate(currents)
    if len(z) > q:
        select = rng.choice(len(z), q, replace=False)
        z = z[select]
        current = current[select]
    return (
        z,
        current,
        dict(
            field_trials=trials,
            field_snapshots=len(values),
            field_values_collected=len(z),
            network_size_N=len(idx),
            stored_memories_P=m.P_FIELDS,
            hsv_noise_level=m.FIELD_HSV_NOISE,
        ),
    )


def run(m):
    out = Path(m.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "queries").mkdir()
    rows = []
    field_rows = []
    boundaries = []
    query_hashes = {}
    for family_idx, family in enumerate(m.FAMILIES):
        for size in m.CARDINALITIES:
            alphabet = m.make_alphabet(family, size)
            boundaries.extend(boundary_validation(alphabet, family))
            seed = m.RANDOM_SEED + 100000 * family_idx + size
            rng = np.random.default_rng(seed)
            limit = 1.25 * np.abs(alphabet).max()
            uniform = rng.uniform(-limit, limit, m.N_QUERY_UNIFORM) + 1j * rng.uniform(-limit, limit, m.N_QUERY_UNIFORM)
            initial = rng.integers(size, size=m.N_QUERY_UNIFORM)
            fields, current, info = local_fields(m, alphabet, m.N_QUERY_FIELDS, seed + 17000)
            field_rows.append(dict(family=family, cardinality=size, **info))
            for name, z, c in [("uniform", uniform, initial), ("local_fields", fields, current)]:
                print(f"{family} |S|={size}, {name}: {len(z)} queries", flush=True)
                filename = f"{family}_{size}_{name}.npz"
                np.savez_compressed(out / "queries" / filename, field=z, current=c, alphabet=alphabet)
                query_hashes[filename] = hashlib.sha256((out / "queries" / filename).read_bytes()).hexdigest()
                names = ["distance", "inertial_score", "kdtree"] + (["structured"] if family != "Hexagonal" else [])
                methods = {key: Search(alphabet, key, family, m.CHUNK_SIZE) for key in names}
                reference = methods["distance"].query(z, c)
                for key, method in methods.items():
                    if not np.array_equal(reference, method.query(z, c)):
                        raise AssertionError(f"Correctness failure: {family}/{size}/{name}/{key}")
                    method.query(z[:512], c[:512])
                for repeat in range(m.N_REPEATS):
                    # Rotate measured order; input arrays and repetition counts are identical.
                    ordering = names[repeat % len(names) :] + names[: repeat % len(names)]
                    for key in ordering:
                        method = methods[key]
                        start = time.perf_counter()
                        result = method.query(z, c)
                        elapsed = time.perf_counter() - start
                        rows.append(
                            dict(
                                family=family,
                                cardinality=size,
                                ensemble=name,
                                method=key,
                                repeat=repeat,
                                query_count=len(z),
                                elapsed_s=elapsed,
                                time_per_query_us=1e6 * elapsed / len(z),
                                queries_per_second=len(z) / elapsed,
                                agreement_with_distance=float(np.mean(result == reference)),
                                build_time_s=method.build_time,
                                conservative_ties=True,
                            )
                        )
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["ensemble", "family", "cardinality", "method"], as_index=False).agg(
        query_count=("query_count", "first"),
        time_per_query_us_mean=("time_per_query_us", "mean"),
        time_per_query_us_std=("time_per_query_us", "std"),
        time_per_query_us_median=("time_per_query_us", "median"),
        time_per_query_us_min=("time_per_query_us", "min"),
        time_per_query_us_max=("time_per_query_us", "max"),
        agreement_min=("agreement_with_distance", "min"),
        build_time_s=("build_time_s", "first"),
    )
    baseline = summary.query('method=="distance"')[
        ["ensemble", "family", "cardinality", "time_per_query_us_mean"]
    ].rename(columns={"time_per_query_us_mean": "distance_time_per_query_us_mean"})
    summary = summary.merge(baseline, on=["ensemble", "family", "cardinality"])
    summary["speedup_vs_distance"] = summary.distance_time_per_query_us_mean / summary.time_per_query_us_mean
    boundary = pd.DataFrame(boundaries)
    assert boundary.disagreements.sum() == 0
    raw.to_csv(out / "experiment3_raw_timings.csv", index=False)
    summary.to_csv(out / "experiment3_summary.csv", index=False)
    boundary.to_csv(out / "boundary_validation.csv", index=False)
    pd.DataFrame(field_rows).to_csv(out / "experiment3_local_field_diagnostics.csv", index=False)
    (out / "query_manifest.json").write_text(json.dumps(query_hashes, indent=2) + "\n")


def install(module):
    module.main = lambda: run(module)
