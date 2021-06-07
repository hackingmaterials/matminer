import unittest

from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.structure.symmetry import (
    GlobalSymmetryFeatures,
    Dimensionality,
)
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class StructureSymmetryFeaturesTest(StructureFeaturesTest):
    def test_global_symmetry(self):
        gsf = GlobalSymmetryFeatures()
        self.assertEqual(gsf.featurize(self.diamond), [227, "cubic", 1, True, 48])

    def test_dimensionality(self):
        cscl = PymatgenTest.get_structure("CsCl")
        graphite = PymatgenTest.get_structure("Graphite")

        df = Dimensionality()

        self.assertEqual(df.featurize(cscl)[0], 3)
        self.assertEqual(df.featurize(graphite)[0], 2)


if __name__ == "__main__":
    unittest.main()
