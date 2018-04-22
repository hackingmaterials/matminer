from matminer.utils.caching import get_nearest_neighbors, _get_all_nearest_neighbors

from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.lattice import Lattice
from pymatgen.util.testing import PymatgenTest
from pymatgen.core import Structure


class TestCaching(PymatgenTest):

    def test_cache(self):
        x = Structure(Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264],
                     [0, 0, 2.528]]), ["C0+", "C0+"], [[2.554, 1.806, 4.423],
                                                       [0.365, 0.258, 0.632]],
            validate_proximity=False,
            to_unit_cell=False, coords_are_cartesian=True)

        # Reset the cache
        _get_all_nearest_neighbors.cache_clear()

        # Compute the nearest neighbors
        method = VoronoiNN()
        nn_1 = get_nearest_neighbors(method, x, 0)

        # Compute it again and make sure the cache hits
        nn_2 = get_nearest_neighbors(method, x, 0)
        self.assertAlmostEquals(nn_1[0]['weight'], nn_2[0]['weight'])
        self.assertEquals(1, _get_all_nearest_neighbors.cache_info().misses)
        self.assertEquals(1, _get_all_nearest_neighbors.cache_info().hits)

        # Reinstantiate the VoronoiNN class, should not cause a miss
        method = VoronoiNN()
        nn_2 = get_nearest_neighbors(method, x, 0)
        self.assertAlmostEquals(nn_1[0]['weight'], nn_2[0]['weight'])
        self.assertEquals(1, _get_all_nearest_neighbors.cache_info().misses)
        self.assertEquals(2, _get_all_nearest_neighbors.cache_info().hits)

        # Change the NN method, should induce a miss
        method = VoronoiNN(weight='volume')
        get_nearest_neighbors(method, x, 0)
        self.assertEquals(2, _get_all_nearest_neighbors.cache_info().misses)
        self.assertEquals(2, _get_all_nearest_neighbors.cache_info().hits)

        # Perturb the structure, make sure it induces a miss and
        #  a change in the NN weights
        x.perturb(0.1)
        nn_2 = get_nearest_neighbors(method, x, 0)
        self.assertNotAlmostEqual(nn_1[0]['weight'], nn_2[0]['weight'])
        self.assertEquals(3, _get_all_nearest_neighbors.cache_info().misses)
        self.assertEquals(2, _get_all_nearest_neighbors.cache_info().hits)
