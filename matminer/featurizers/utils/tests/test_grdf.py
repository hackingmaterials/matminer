from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.utils.grdf import initialize_pairwise_function, Gaussian, Histogram

import numpy as np


class GRDFTests(PymatgenTest):

    def test_load_class(self):
        g = initialize_pairwise_function('Gaussian', center=4, width=1)
        self.assertIsInstance(g, Gaussian)
        self.assertEqual(g.center, 4)
        self.assertEqual(g.width, 1)

    def test_gaussian(self):
        g = Gaussian(center=4, width=4)
        self.assertArrayAlmostEqual([1, np.exp(-1)], g([4, 0]), decimal=4)
        self.assertAlmostEqual(g.volume(20), 2118, places=0)

        # Make sure the name makes sense
        name = g.name()
        self.assertIn('Gaussian', name)
        self.assertIn('width=4', name)
        self.assertIn('center=4', name)

    def test_histogram(self):
        h = Histogram(1, 4)
        self.assertArrayAlmostEqual([0, 1, 0], h([0.5, 2, 5]))
        self.assertAlmostEqual(h.volume(10), 4 / 3.0 * np.pi * (5 ** 3 - 1 ** 3))
