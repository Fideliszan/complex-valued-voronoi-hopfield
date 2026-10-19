"""A local example illustrating magnitude saturation without a large simulation."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cvhnn.independent import decision
alphabet=np.array([.3,.6,1.],dtype=complex);field=.4+0j
c=decision(alphabet,field,0,'classic');v=decision(alphabet,field,0,'voronoi')
assert alphabet[c]==1 and alphabet[v]==.3
scores=(alphabet.conj()*field).real-np.abs(alphabet)**2/2
assert np.allclose(abs(alphabet-field)**2,abs(field)**2-2*scores)
print(json.dumps({'field':field.real,'classical_radius':alphabet[c].real,'voronoi_radius':alphabet[v].real,'distance_score_identity':True},indent=2))
