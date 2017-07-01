import unittest2 as unittest
import pandas as pd

from matminer.descriptors.composition_features_2 import StoichAttributes, ElemPropertyAttributes, ValenceOrbitalAttributes, IonicAttributes 
from pymatgen.util.testing import PymatgenTest

class CompositionFeaturesTest(PymatgenTest):

    def setUp(self):
        self.df = pd.DataFrame({"composition":["Fe2O3"]})

    def test_stoich(self):
        df_stoich = StoichAttributes().featurize_all(self.df)
        self.assertAlmostEqual(df_stoich["0-norm"][0], 2)
        self.assertAlmostEqual(df_stoich["7-norm"][0], 0.604895199)

    def test_elem(self):
        df_elem = ElemPropertyAttributes().featurize_all(self.df)
        self.assertAlmostEqual(df_elem["Min Number"][0], 8)
        self.assertAlmostEqual(df_elem["Max Number"][0], 26)
        self.assertAlmostEqual(df_elem["Range Number"][0], 18)
        self.assertAlmostEqual(df_elem["Mean Number"][0], 15.2)
        self.assertAlmostEqual(df_elem["AbsDev Number"][0], 8.64)
        self.assertAlmostEqual(df_elem["Mode Number"][0], 8)

    def test_valence(self):
        df_val = ValenceOrbitalAttributes().featurize_all(self.df)
        self.assertAlmostEqual(df_val["Frac s Valence Electrons"][0], 0.294117647)
        self.assertAlmostEqual(df_val["Frac d Valence Electrons"][0], 0.352941176)
        self.assertAlmostEqual(df_val["Frac p Valence Electrons"][0], 0.352941176)
        self.assertAlmostEqual(df_val["Frac f Valence Electrons"][0], 0)

    def test_ionic(self):
        df_ionic = IonicAttributes().featurize_all(self.df)
        self.assertEqual(df_ionic["compound possible"][0], 1.0)
        self.assertAlmostEqual(df_ionic["Max Ionic Char"][0], 0.476922164)
        self.assertAlmostEqual(df_ionic["Avg Ionic Char"][0], 0.114461319)

if __name__=="__main__":
    unittest.main()
