import os
import unittest

from monty.serialization import loadfn

from matminer.featurizers.structure.composite import JarvisCFID
from matminer.featurizers.structure.tests.base import StructureFeaturesTest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


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

        # test compounds with missing elements for chemical or
        # charge descriptors raise the correct errors
        # Li4Eu4P4, mp-1211143
        s = loadfn(os.path.join(TEST_DIR, "JarvisCFID_problem_file.json"))

        jcf_nochem = JarvisCFID(use_chem=False)
        jcf_wchem = JarvisCFID(use_chem=True)
        with self.assertRaises(ValueError):
            jcf_wchem.featurize(s)

        self.assertAlmostEqual(jcf_nochem.featurize(s)[0], 3.04999, 3)


if __name__ == "__main__":
    unittest.main()
