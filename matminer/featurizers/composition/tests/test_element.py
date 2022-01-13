import unittest

from pymatgen.core import Composition

from matminer.featurizers.composition.element import (
    BandCenter,
    ElementFraction,
    Stoichiometry,
    TMetalFraction,
)
from matminer.featurizers.composition.tests.base import CompositionFeaturesTest


class ElementFeaturesTest(CompositionFeaturesTest):
    def test_stoich(self):
        featurizer = Stoichiometry(num_atoms=True)
        df_stoich = Stoichiometry(num_atoms=True).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_stoich["num atoms"][0], 5)
        self.assertAlmostEqual(df_stoich["0-norm"][0], 2)
        self.assertAlmostEqual(df_stoich["7-norm"][0], 0.604895199)

        # Test whether the number of formula units affects result
        original_value = featurizer.featurize(Composition("FeO"))
        self.assertArrayAlmostEqual(featurizer.featurize(Composition("Fe0.5O0.5")), original_value)
        self.assertArrayAlmostEqual(featurizer.featurize(Composition("Fe2O2")), original_value)

    def test_fraction(self):
        df_frac = ElementFraction().featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_frac["O"][0], 0.6)
        self.assertEqual(df_frac["Fe"][0], 0.4)

    def test_tm_fraction(self):
        df_tm_frac = TMetalFraction().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_tm_frac["transition metal fraction"][0], 0.4)

    def test_band_center(self):
        df_band_center = BandCenter().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_band_center["band center"][0], 5.870418816395603)
        self.assertAlmostEqual(BandCenter().featurize(Composition("Ag33O500V200"))[0], 6.033480099340539)


if __name__ == "__main__":
    unittest.main()
