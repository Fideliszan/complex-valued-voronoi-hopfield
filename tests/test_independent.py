"""Independent reference computations and deliberately adversarial boundaries."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cvhnn import independent as core, activation_cost, recall, saturation
from cvhnn.independent_search import Search,boundary_validation


class Independent(unittest.TestCase):
    def test_all_cardinalities_and_boundary_rules(self):
        rng=np.random.default_rng(671)
        for family in activation_cost.FAMILIES:
            for size in (16,64,256,1024):
                alphabet=activation_cost.make_alphabet(family,size)
                self.assertEqual(sum(r['disagreements'] for r in boundary_validation(alphabet,family)),0)
                z=rng.normal(size=99)+1j*rng.normal(size=99)
                currents=rng.integers(size,size=len(z))
                reference=np.array([core.decision(alphabet,v,int(c)) for v,c in zip(z,currents)])
                for method in ('distance','inertial_score','kdtree')+(() if family=='Hexagonal' else ('structured',)):
                    np.testing.assert_array_equal(Search(alphabet,method,family).query(z,currents),reference)

    def test_full_recomputation_trajectory_and_energy(self):
        rng=np.random.default_rng(31415)
        a=np.array([0,.3,.8,1,1j,-1,-1j],complex)
        for rule in ('classic','voronoi'):
            for _ in range(12):
                w=core.random_weights(9,rng,diagonal_value=.2)
                idx=rng.integers(len(a),size=9);orders=[rng.permutation(9) for _ in range(80)]
                ref=idx.copy();old_energy=core.energy(a[ref],w)[rule=='voronoi']
                for order in orders:
                    count=0
                    for k in order:
                        v=np.dot(w[k],a[ref])
                        new=core.decision(a,v,int(ref[k]),rule)
                        if new!=ref[k]:
                            ref[k]=new;count+=1
                            new_energy=core.energy(a[ref],w)[rule=='voronoi']
                            self.assertLess(new_energy,old_energy+1e-13)
                            old_energy=new_energy
                    if not count:break
                result=core.dynamics(a,w,idx,orders,rule)
                np.testing.assert_array_equal(result.indices,ref)
                self.assertTrue(result.converged)
                self.assertEqual(result.audit['coordinate_energy_increases'],0)

    def test_projection_and_metrics_against_definitions(self):
        rng=np.random.default_rng(431)
        memories=rng.normal(size=(3,20))+1j*rng.normal(size=(3,20))
        # Duplicate memories exercise the pseudoinverse rather than an inverse.
        memories=np.vstack((memories,memories[0]))
        w,info=core.projection_weights(memories)
        np.testing.assert_allclose(w@memories.T,memories.T,atol=1e-12)
        np.testing.assert_allclose(w@w,w,atol=1e-12)
        self.assertLess(info['projection_residual'],1e-12)
        a=np.array([0,.5,1j],complex);target=np.array([0,1,2]);final=np.array([0,2,2])
        metric=core.retrieval_metrics(final,target,a)
        self.assertAlmostEqual(metric['SER'],1/3)
        self.assertAlmostEqual(metric['radial_error'],1/6)
        self.assertEqual(core.retrieval_metrics(target,target,a)['rgb_psnr'],100.)

    def test_static_total_order_at_origin(self):
        a=saturation.build_alphabets()['16-PSK']
        self.assertEqual(int(core.quantize(np.array([0j]),a)[0]),0)
        for c in range(len(a)):
            self.assertEqual(core.decision(a,0j,c),c)

    def test_noise_semantics_and_pairing(self):
        target=np.zeros(10000,dtype=int)
        result=core.corrupt_symbols(target,4,1.,np.random.default_rng(52))
        self.assertTrue(.22<np.mean(result==0)<.28)
        # Replacement includes the original symbol; q is not realized SER.
        np.testing.assert_array_equal(core.corrupt_symbols(target,4,0.,np.random.default_rng(52)),target)


if __name__=='__main__':unittest.main()
