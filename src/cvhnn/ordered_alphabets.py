"""Published ordered coordinates avoid platform-sensitive equal-radius sorting."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np

@lru_cache(maxsize=1)
def _read():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/ordered_alphabets.json').read_text())

def load_group(name):
    return {k:np.asarray(v,dtype=float)[:,0]+1j*np.asarray(v,dtype=float)[:,1] for k,v in _read()[name].items()}
