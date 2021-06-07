import unittest

from matminer.featurizers.structure.composite import JarvisCFID
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class CompositeStructureFeaturesTest(StructureFeaturesTest):
    def test_jarvisCFID(self):
        # default (all descriptors)
        jcf = JarvisCFID()
        self.assertEqual(len(jcf.feature_labels()), 1557)
        fvec = jcf.featurize(self.cscl)
        self.assertEqual(len(fvec), 1557)
        self.assertAlmostEqual(fvec[-1], 0.0, places=3)
        self.assertAlmostEqual(fvec[1], 591.5814, places=3)
        self.assertAlmostEqual(fvec[0], 1346.755, places=3)

        # a combination of descriptors
        jcf = JarvisCFID(use_chem=False, use_chg=False, use_cell=False)
        self.assertEqual(len(jcf.feature_labels()), 737)
        fvec = jcf.featurize(self.diamond)
        self.assertAlmostEqual(fvec[-1], 24, places=3)
        self.assertAlmostEqual(fvec[0], 0, places=3)


if __name__ == "__main__":
    unittest.main()
