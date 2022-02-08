import unittest

from pymatgen.core import Composition
from pymatgen.core.periodic_table import Specie

from matminer.featurizers.composition.ion import (
    CationProperty,
    ElectronAffinity,
    ElectronegativityDiff,
    IonProperty,
    OxidationStates,
    is_ionic,
)
from matminer.featurizers.composition.tests.base import CompositionFeaturesTest


class IonFeaturesTest(CompositionFeaturesTest):
    def test_is_ionic(self):
        """Test checking whether a compound is ionic"""

        self.assertTrue(is_ionic(Composition({Specie("Fe", 2): 1, Specie("O", -2): 1})))
        self.assertFalse(is_ionic(Composition({Specie("Fe", 0): 1, Specie("Al", 0): 1})))

    def test_ionic(self):
        featurizer = IonProperty()
        df_ionic = featurizer.featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_ionic["compound possible"][0], 1.0)
        self.assertAlmostEqual(df_ionic["max ionic char"][0], 0.476922164)
        self.assertAlmostEqual(df_ionic["avg ionic char"][0], 0.114461319)

        # Test 'fast'
        self.assertEqual(1.0, featurizer.featurize(Composition("Fe3O4"))[0])
        featurizer.fast = True
        self.assertEqual(0, featurizer.featurize(Composition("Fe3O4"))[0])

        # Make sure 'fast' works if I use-precomputed oxidation states
        self.assertEqual(
            1,
            featurizer.featurize(Composition({Specie("Fe", 2): 1, Specie("Fe", 3): 2, Specie("O", -2): 4}))[0],
        )

    def test_cation_properties(self):
        featurizer = CationProperty.from_preset("deml")
        features = dict(
            zip(
                featurizer.feature_labels(),
                featurizer.featurize(self.df["composition"][1]),
            )
        )
        self.assertAlmostEqual(features["DemlData minimum magn_moment of cations"], 5.48)
        self.assertAlmostEqual(features["DemlData maximum magn_moment of cations"], 5.48)
        self.assertAlmostEqual(features["DemlData range magn_moment of cations"], 0)
        self.assertAlmostEqual(features["DemlData mean magn_moment of cations"], 5.48)
        self.assertAlmostEqual(features["DemlData std_dev magn_moment of cations"], 0)

    def test_elec_affin(self):
        featurizer = ElectronAffinity()
        self.assertAlmostEqual(-141000 * 2, featurizer.featurize(self.df["composition"][1])[0])

    def test_en_diff(self):
        featurizer = ElectronegativityDiff()
        features = dict(
            zip(
                featurizer.feature_labels(),
                featurizer.featurize(self.df["composition"][1]),
            )
        )
        self.assertAlmostEqual(features["minimum EN difference"], 1.6099999999)
        self.assertAlmostEqual(features["maximum EN difference"], 1.6099999999)
        self.assertAlmostEqual(features["range EN difference"], 0)
        self.assertAlmostEqual(features["mean EN difference"], 1.6099999999)
        self.assertAlmostEqual(features["std_dev EN difference"], 0)

    def test_oxidation_states(self):
        featurizer = OxidationStates.from_preset("deml")
        features = dict(
            zip(
                featurizer.feature_labels(),
                featurizer.featurize(self.df["composition"][1]),
            )
        )
        self.assertAlmostEqual(4, features["range oxidation state"])
        self.assertAlmostEqual(2, features["maximum oxidation state"])


if __name__ == "__main__":
    unittest.main()
