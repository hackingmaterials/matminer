import unittest

import numpy as np
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN

from matminer.featurizers.site.bonding import (
    AverageBondAngle,
    AverageBondLength,
    BondOrientationalParameter,
)
from matminer.featurizers.site.tests.base import SiteFeaturizerTest


class BondingTest(SiteFeaturizerTest):
    def test_bop(self):
        f = BondOrientationalParameter(max_l=10, compute_w=True, compute_w_hat=True)

        # Check the feature count
        self.assertEqual(30, len(f.feature_labels()))
        self.assertEqual(30, len(f.featurize(self.sc, 0)))

        f.compute_W = False
        self.assertEqual(20, len(f.feature_labels()))
        self.assertEqual(20, len(f.featurize(self.sc, 0)))

        f.compute_What = False
        self.assertEqual(10, len(f.featurize(self.sc, 0)))
        self.assertEqual(10, len(f.featurize(self.sc, 0)))

        f.compute_W = f.compute_What = True

        # Compute it for SC and B1
        sc_features = f.featurize(self.sc, 0)
        b1_features = f.featurize(self.b1, 0)

        # They should be equal
        self.assertArrayAlmostEqual(sc_features, b1_features)

        # Comparing Q's to results from https://aip.scitation.org/doi/10.1063/1.4774084
        self.assertArrayAlmostEqual([0, 0, 0, 0.764, 0, 0.354, 0, 0.718, 0, 0.411], sc_features[:10], decimal=3)

        # Comparing W's to results from https://link.aps.org/doi/10.1103/PhysRevB.28.784
        self.assertArrayAlmostEqual(
            [0, 0, 0, 0.043022, 0, 0.000612, 0, 0.034055, 0, 0.013560],
            sc_features[10:20],
            decimal=3,
        )

        self.assertArrayAlmostEqual(
            [0, 0, 0, 0.159317, 0, 0.013161, 0, 0.058455, 0, 0.090130],
            sc_features[20:],
            decimal=3,
        )

    def test_AverageBondLength(self):
        ft = AverageBondLength(VoronoiNN())
        self.assertAlmostEqual(ft.featurize(self.sc, 0)[0], 3.52)

        for i in range(len(self.cscl.sites)):
            self.assertAlmostEqual(ft.featurize(self.cscl, i)[0], 3.758562645051973)

        for i in range(len(self.b1.sites)):
            self.assertAlmostEqual(ft.featurize(self.b1, i)[0], 1.0)

        ft = AverageBondLength(CrystalNN())
        for i in range(len(self.cscl.sites)):
            self.assertAlmostEqual(ft.featurize(self.cscl, i)[0], 3.649153279231275)

    def test_AverageBondAngle(self):
        ft = AverageBondAngle(VoronoiNN())

        self.assertAlmostEqual(ft.featurize(self.sc, 0)[0], np.pi / 2)

        for i in range(len(self.cscl.sites)):
            self.assertAlmostEqual(ft.featurize(self.cscl, i)[0], 0.9289637531152273)

        for i in range(len(self.b1.sites)):
            self.assertAlmostEqual(ft.featurize(self.b1, i)[0], np.pi / 2)

        ft = AverageBondAngle(CrystalNN())
        for i in range(len(self.b1.sites)):
            self.assertAlmostEqual(ft.featurize(self.b1, i)[0], np.pi / 2)


if __name__ == "__main__":
    unittest.main()
