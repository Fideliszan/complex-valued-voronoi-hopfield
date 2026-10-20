"""A local example illustrating magnitude saturation without a large simulation."""

import json

import numpy as np

from cvhnn.independent import decision

alphabet = np.array([0.3, 0.6, 1.0], dtype=complex)
field = 0.4 + 0j
c = decision(alphabet, field, 0, "classic")
v = decision(alphabet, field, 0, "voronoi")
assert alphabet[c] == 1 and alphabet[v] == 0.3
scores = (alphabet.conj() * field).real - np.abs(alphabet) ** 2 / 2
assert np.allclose(abs(alphabet - field) ** 2, abs(field) ** 2 - 2 * scores)
print(
    json.dumps(
        {
            "field": field.real,
            "classical_radius": alphabet[c].real,
            "voronoi_radius": alphabet[v].real,
            "distance_score_identity": True,
        },
        indent=2,
    )
)
