"""Four nearest-symbol algorithms with a common conservative boundary rule.

The fast paths find the closest two candidates. Ambiguous gaps use a complete
distance scan; consequently boundary handling never depends on KD-tree order
or angular rounding. Timings include this check and current-state retention.
"""

import time

import numpy as np
from scipy.spatial import cKDTree

TIE_ATOL = 1e-12


class Search:
    def __init__(self, alphabet, method="distance", family=None, chunk_size=4096):
        start = time.perf_counter()
        self.a = np.asarray(alphabet, dtype=complex)
        self.method = method
        self.family = family
        self.chunk = chunk_size
        self.norm = self.a.real**2 + self.a.imag**2
        self.penalty = 0.5 * self.norm
        self.tree = None
        if method == "kdtree":
            self.tree = cKDTree(np.column_stack((self.a.real, self.a.imag)))
        if method == "structured":
            if family == "QAM":
                self.levels = np.unique(self.a.real)
                x = np.searchsorted(self.levels, self.a.real)
                y = np.searchsorted(self.levels, self.a.imag)
                self.lookup = np.empty((len(self.levels), len(self.levels)), dtype=int)
                self.lookup[x, y] = np.arange(len(self.a))
            elif family in ("PSK", "Polar"):
                self.radii = np.unique(np.round(np.abs(self.a), 12))
                self.phases = len(self.a) // len(self.radii)
                phases = np.rint(np.angle(self.a) * self.phases / (2 * np.pi)).astype(int) % self.phases
                radial = np.argmin(abs(np.abs(self.a)[:, None] - self.radii), axis=1)
                self.lookup = np.empty((len(self.radii), self.phases), dtype=int)
                self.lookup[radial, phases] = np.arange(len(self.a))
            else:
                raise ValueError("No structured method is specified for this alphabet.")
        self.build_time = time.perf_counter() - start

    def _structured_candidates(self, z):
        if self.family == "QAM":
            x = np.searchsorted(self.levels, z.real)
            y = np.searchsorted(self.levels, z.imag)
            return np.column_stack(
                [
                    self.lookup[np.clip(x + dx, 0, len(self.levels) - 1), np.clip(y + dy, 0, len(self.levels) - 1)]
                    for dx, dy in ((-1, -1), (-1, 0), (0, -1), (0, 0))
                ]
            )
        phase = z.imag * 0 + np.angle(z) * self.phases / (2 * np.pi)
        floor = np.floor(phase).astype(int)
        candidates = []
        for offset in (0, 1):
            angular = (floor + offset) % self.phases
            theta = angular * 2 * np.pi / self.phases
            projected = z.real * np.cos(theta) + z.imag * np.sin(theta)
            pos = np.searchsorted(self.radii, projected)
            for shift in (-1, 0):
                candidates.append(self.lookup[np.clip(pos + shift, 0, len(self.radii) - 1), angular])
        return np.column_stack(candidates)

    def query(self, queries, current=None):
        z = np.asarray(queries, dtype=complex).reshape(-1)
        c = None if current is None else np.broadcast_to(current, z.shape)
        answer = np.empty(len(z), dtype=np.int64)
        for start in range(0, len(z), self.chunk):
            q = z[start : start + self.chunk]
            if self.method in ("distance", "inertial_score"):
                if self.method == "distance":
                    all_values = (q.real[:, None] - self.a.real) ** 2 + (q.imag[:, None] - self.a.imag) ** 2
                    gap_tolerance = TIE_ATOL
                else:
                    all_values = self.penalty - (q.real[:, None] * self.a.real + q.imag[:, None] * self.a.imag)
                    # A score gap is half the corresponding squared-distance gap.
                    gap_tolerance = 0.5 * TIE_ATOL
                candidate = np.argmin(all_values, axis=1)
                best = all_values[np.arange(len(q)), candidate]
                ambiguous = np.sum(all_values <= best[:, None] + gap_tolerance, axis=1) > 1
            else:
                if self.method == "kdtree":
                    _, choices = self.tree.query(np.column_stack((q.real, q.imag)), k=2, workers=1)
                else:
                    choices = self._structured_candidates(q)
                d = (q.real[:, None] - self.a[choices].real) ** 2 + (q.imag[:, None] - self.a[choices].imag) ** 2
                candidate = choices[np.arange(len(q)), np.argmin(d, axis=1)]
                best = d.min(axis=1)
                # Duplicate clamped candidates do not constitute a tie.
                ambiguous = np.any((d <= best[:, None] + TIE_ATOL) & (choices != candidate[:, None]), axis=1)
                # At the origin all phase directions tie, including duplicate paths.
                if self.method == "structured" and self.family in ("PSK", "Polar"):
                    ambiguous |= abs(q) < 1e-10
            if ambiguous.any():
                ambiguous_q = q[ambiguous]
                d = (ambiguous_q.real[:, None] - self.a.real) ** 2 + (ambiguous_q.imag[:, None] - self.a.imag) ** 2
                candidate[ambiguous] = np.argmax(d <= d.min(axis=1)[:, None] + TIE_ATOL, axis=1)
            if c is not None:
                previous = c[start : start + len(q)]
                dnew = (q.real - self.a[candidate].real) ** 2 + (q.imag - self.a[candidate].imag) ** 2
                dold = (q.real - self.a[previous].real) ** 2 + (q.imag - self.a[previous].imag) ** 2
                candidate = np.where(dold <= dnew + TIE_ATOL, previous, candidate)
                # For ambiguous sets compare current to the actual minimum, not
                # the first near-tied candidate (which could add tolerance twice).
                if ambiguous.any():
                    keep = dold[ambiguous] <= d.min(axis=1) + TIE_ATOL
                    canonical = np.argmax(d <= d.min(axis=1)[:, None] + TIE_ATOL, axis=1)
                    candidate[ambiguous] = np.where(keep, previous[ambiguous], canonical)
            answer[start : start + len(q)] = candidate
        return answer


def boundary_validation(alphabet, family):
    a = alphabet
    tree = cKDTree(np.column_stack((a.real, a.imag)))
    neighbours = tree.query(np.column_stack((a.real, a.imag)), k=2)[1][:, 1]
    fields = np.concatenate(((a + a[neighbours]) / 2, np.array([0j])))
    methods = ["distance", "inertial_score", "kdtree"] + (["structured"] if family != "Hexagonal" else [])
    rows = []
    for current in (None, np.r_[np.arange(len(a)), 0], np.r_[neighbours, len(a) - 1]):
        reference = Search(a).query(fields, current)
        for method in methods:
            result = Search(a, method, family).query(fields, current)
            rows.append(
                dict(
                    family=family,
                    cardinality=len(a),
                    method=method,
                    current_supplied=current is not None,
                    queries=len(fields),
                    disagreements=int(np.sum(result != reference)),
                )
            )
    return rows
