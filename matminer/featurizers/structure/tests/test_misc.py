import unittest

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure.misc import (
    EwaldEnergy,
    StructureComposition,
    XRDPowderPattern,
)
from matminer.featurizers.structure.tests.base import StructureFeaturesTest


class MiscStructureFeaturesTest(StructureFeaturesTest):
    def test_ewald(self):
        # Add oxidation states to all of the structures
        for s in [self.nacl, self.cscl, self.diamond]:
            s.add_oxidation_state_by_guess()

        # Test basic
        ewald = EwaldEnergy(accuracy=2)
        self.assertArrayAlmostEqual(ewald.featurize(self.diamond), [0])
        self.assertAlmostEqual(ewald.featurize(self.nacl)[0], -4.418439, 2)
        self.assertLess(ewald.featurize(self.nacl), ewald.featurize(self.cscl))  # Atoms are closer in NaCl

    def test_composition_features(self):
        comp = ElementProperty.from_preset("magpie")
        f = StructureComposition(featurizer=comp)

        # Test the fitting (should not crash)
        f.fit([self.nacl, self.diamond])

        # Test the features
        features = f.featurize(self.nacl)
        self.assertArrayAlmostEqual(comp.featurize(self.nacl.composition), features)

        # Test the citations/implementors
        self.assertEqual(comp.citations(), f.citations())
        self.assertEqual(comp.implementors(), f.implementors())

    def test_xrd_powderPattern(self):

        # default settings test
        xpp = XRDPowderPattern()
        pattern = xpp.featurize(self.diamond)
        self.assertAlmostEqual(pattern[44], 0.19378, places=2)
        self.assertEqual(len(pattern), 128)

        # reduced range
        xpp = XRDPowderPattern(two_theta_range=(0, 90))
        pattern = xpp.featurize(self.diamond)
        self.assertAlmostEqual(pattern[44], 0.4083, places=2)
        self.assertEqual(len(pattern), 91)
        self.assertEqual(len(xpp.feature_labels()), 91)


if __name__ == "__main__":
    unittest.main()
