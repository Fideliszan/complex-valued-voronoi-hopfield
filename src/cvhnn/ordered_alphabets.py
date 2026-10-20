"""Published ordered coordinates avoid platform-sensitive equal-radius sorting."""

import json
from functools import lru_cache

import numpy as np

from cvhnn.config import CONFIGS


@lru_cache(maxsize=1)
def _read():
    return json.loads((CONFIGS / "ordered_alphabets.json").read_text())


def load_group(name):
    return {
        k: np.asarray(v, dtype=float)[:, 0] + 1j * np.asarray(v, dtype=float)[:, 1] for k, v in _read()[name].items()
    }
