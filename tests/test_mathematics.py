"""Independent identities and boundary cases underlying the convergence argument."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cvhnn import saturation as s, recall, activation_cost as cost

class Mathematics(unittest.TestCase):
    def test_conservative_boundary_and_zero(self):
        a=np.array([-1,1],dtype=complex)
        self.assertEqual(s.voronoi_activation_index(0,1,a),1)
        self.assertEqual(s.classical_activation_index(0,0,a),0)
        b=np.array([0,1,2],dtype=complex)
        self.assertEqual(s.voronoi_activation_index(.5,2,b),0)

    def test_coordinate_energy_identity_and_descent(self):
        rng=np.random.default_rng(914)
        for _ in range(100):
            a=s.build_alphabets()['16-QAM'];n=7
            z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));w=(z+z.conj().T)/2
            np.fill_diagonal(w,rng.random(n))
            idx=rng.integers(len(a),size=n);x=a[idx];k=int(rng.integers(n));v=w@x
            new=s.voronoi_activation_index(v[k],int(idx[k]),a);y=x.copy();y[k]=a[new];delta=y[k]-x[k]
            energy=lambda q:-np.vdot(q,w@q).real/2+np.vdot(q,q).real/2
            formula=-(delta.conjugate()*v[k]).real+(abs(y[k])**2-abs(x[k])**2)/2-w[k,k].real*abs(delta)**2/2
            self.assertAlmostEqual(energy(y)-energy(x),formula,places=11)
            self.assertLessEqual(energy(y)-energy(x),1e-10)

    def test_projector_stores_memories(self):
        rng=np.random.default_rng(15);x=rng.normal(size=(16,3))+1j*rng.normal(size=(16,3))
        w=x@np.linalg.pinv(x.conj().T@x)@x.conj().T
        self.assertTrue(np.allclose(w@x,x));self.assertTrue(np.allclose(w,w.conj().T));self.assertTrue(np.all(w.diagonal().real>=0))

    def test_constant_modulus_equivalence(self):
        a=s.build_alphabets()['16-PSK'];rng=np.random.default_rng(71)
        for _ in range(300):
            v=complex(*rng.normal(size=2));idx=int(rng.integers(len(a)))
            self.assertEqual(s.classical_activation_index(v,idx,a),s.voronoi_activation_index(v,idx,a))

    def test_searches_on_untied_queries(self):
        rng=np.random.default_rng(74);z=rng.normal(size=127)+1j*rng.normal(size=127)
        for family in cost.FAMILIES:
            a=cost.make_alphabet(family,64);methods=cost.build_methods(family,a);ref=methods[0].query(z)
            for method in methods:self.assertTrue(np.array_equal(method.query(z),ref),(family,method.name))

if __name__=='__main__':unittest.main()
