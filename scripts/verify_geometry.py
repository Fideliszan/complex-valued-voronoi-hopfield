"""Check the geometry actually used in Figures 1 and 2 against the definitions."""

import json

import numpy as np
import pandas as pd

from cvhnn import saturation as sat
from cvhnn.config import FIGURES, VALIDATION
from render_geometry import bisector_segment, cell_polygon, illustrative_alphabet


def verify():
    generic = illustrative_alphabet()
    d = pd.read_csv(FIGURES / "source_data/irregular_symbols.csv")
    np.testing.assert_allclose(d.real + 1j * d.imag, generic, rtol=0, atol=1e-15)
    separation = abs(generic[:, None] - generic[None, :]) + np.eye(len(generic)) * 10
    assert separation.min() > 0.35
    exported = pd.read_csv(FIGURES / "source_data/geometry_alphabets.csv")
    records = {}
    for name, z in {"Illustrative": generic, **sat.build_alphabets()}.items():
        if name == "Illustrative":
            i, j = 0, 1
        else:
            rows = exported[exported.alphabet == name].sort_values("index")
            np.testing.assert_allclose(rows.real + 1j * rows.imag, z, rtol=0, atol=1e-15)
            i = int(rows.loc[rows.selected, "index"].iloc[0])
            j = int(rows.loc[rows.neighbor, "index"].iloc[0])
        total_area = 0.0
        maximum_residual = 0.0
        for k in range(len(z)):
            poly = cell_polygon(z, k)
            vertices = poly[:, 0] + 1j * poly[:, 1]
            residual = abs(vertices[:, None] - z[k]) ** 2 - abs(vertices[:, None] - z) ** 2
            maximum_residual = max(maximum_residual, float(residual.max()))
            assert residual.max() < 2e-12, (name, k)
            total_area += (
                abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))) / 2
            )
        assert abs(total_area - (2 * 1.22) ** 2) < 2e-12, name
        face = bisector_segment(z, i, j, active=True)
        assert face is not None
        pts = np.linspace(face[0], face[1], 101)
        pts = pts[:, 0] + 1j * pts[:, 1]
        distances = abs(pts[:, None] - z) ** 2
        np.testing.assert_allclose(distances[:, i], distances[:, j], atol=2e-12, rtol=0)
        assert np.max(distances[:, i] - distances.min(axis=1)) < 2e-12
        records[name] = dict(
            symbols=len(z),
            maximum_modulus=float(abs(z).max()),
            saturation_symbols=int(np.isclose(abs(z), abs(z).max(), atol=1e-12, rtol=0).sum()),
            selected_pair=[i, j],
            cell_area_sum=total_area,
            maximum_cell_inequality_residual=maximum_residual,
            shared_face_samples=101,
        )
    report = dict(
        status="pass",
        illustrative_minimum_separation=float(separation.min()),
        experimental_alphabets_unchanged=True,
        alphabets=records,
    )
    (VALIDATION / "geometry_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: exact cell inequalities, window coverage, shared faces, symbol spacing, "
        "and unchanged experimental alphabets."
    )
    return report


if __name__ == "__main__":
    verify()
