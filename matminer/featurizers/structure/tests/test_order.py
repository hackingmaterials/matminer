import copy
import os
import unittest

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from matminer.featurizers.structure.order import (
    ChemicalOrdering,
    DensityFeatures,
    MaximumPackingEfficiency,
    StructuralComplexity,
)
from matminer.featurizers.structure.tests.base import StructureFeaturesTest

test_dir = os.path.join(os.path.dirname(__file__))


class OrderStructureFeaturesTest(StructureFeaturesTest):
    def test_density_features(self):
        df = DensityFeatures()
        f = df.featurize(self.diamond)
        self.assertAlmostEqual(f[0], 3.49, 2)
        self.assertAlmostEqual(f[1], 5.71, 2)
        self.assertAlmostEqual(f[2], 0.25, 2)

        f = df.featurize(self.nacl)
        self.assertAlmostEqual(f[0], 2.105, 2)
        self.assertAlmostEqual(f[1], 23.046, 2)
        self.assertAlmostEqual(f[2], 0.620, 2)

        nacl_disordered = copy.deepcopy(self.nacl)
        nacl_disordered.replace_species({"Cl1-": "Cl0.99H0.01"})
        self.assertFalse(df.precheck(nacl_disordered))
        structures = [self.diamond, self.nacl, nacl_disordered]
        df2 = pd.DataFrame({"structure": structures})
        self.assertAlmostEqual(df.precheck_dataframe(df2, "structure"), 2 / 3)

    def test_ordering_param(self):
        f = ChemicalOrdering()

        # Check that elemental structures return zero
        features = f.featurize(self.diamond)
        self.assertArrayAlmostEqual([0, 0, 0], features)

        # Check result for CsCl
        #   These were calculated by hand by Logan Ward
        features = f.featurize(self.cscl)
        self.assertAlmostEqual(0.551982, features[0], places=5)
        self.assertAlmostEqual(0.241225, features[1], places=5)

        # Check for L1_2
        features = f.featurize(self.ni3al)
        self.assertAlmostEqual(1.0 / 3.0, features[0], places=5)
        self.assertAlmostEqual(0.0303, features[1], places=5)

    def test_packing_efficiency(self):
        f = MaximumPackingEfficiency()

        # Test L1_2
        self.assertArrayAlmostEqual([np.pi / 3 / np.sqrt(2)], f.featurize(self.ni3al))

        # Test B1
        self.assertArrayAlmostEqual([np.pi / 6], f.featurize(self.nacl), decimal=3)

    def test_structural_complexity(self):
        s = Structure.from_file(os.path.join(test_dir, "Dy2HfS5_mp-1198001_computed.cif"))
        featurizer = StructuralComplexity()
        ig, igbits = featurizer.featurize(s)
        self.assertAlmostEqual(2.5, ig, places=3)
        self.assertAlmostEqual(80, igbits, places=3)
        s = Structure.from_file(os.path.join(test_dir, "Cs2CeN5O17_mp-1198000_computed.cif"))
        featurizer = StructuralComplexity()
        ig, igbits = featurizer.featurize(s)
        self.assertAlmostEqual(3.764, ig, places=3)


if __name__ == "__main__":
    unittest.main()
