from __future__ import unicode_literals, division, print_function

import unittest

from pymatgen import Composition, Specie
from pymatgen.util.testing import PymatgenTest

import pandas as pd

from matminer.descriptors.composition_features import get_composition_oxidation_state, \
    get_pymatgen_descriptor, StoichAttributes, ElemPropertyAttributes, ValenceOrbitalAttributes, IonicAttributes


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


class PymatgenDescriptorTest(unittest.TestCase):

    def setUp(self):
        self.nacl_formula_1 = "NaCl"
        self.nacl_formula_2 = "Na+1Cl-1"
        self.fe2o3_formula_1 = "Fe2+3O3-2"
        self.fe2o3_formula_2 = "Fe2 +3 O3 -2"
        self.lifepo4 = "LiFePO4"

    def test_comp_oxstate_from_formula(self):
        fe2o3_comp_1, fe2o3_oxstates_1 = get_composition_oxidation_state(self.fe2o3_formula_1)
        oxstates_ans = {'Fe': 3, 'O': -2}
        comp_ans = Composition("Fe2O3")
        self.assertEqual(fe2o3_comp_1, comp_ans)
        self.assertDictEqual(fe2o3_oxstates_1, oxstates_ans)
        fe2o3_comp_2, fe2o3_oxstates_2 = get_composition_oxidation_state(self.fe2o3_formula_2)
        self.assertEqual(fe2o3_comp_1, fe2o3_comp_2)
        self.assertDictEqual(fe2o3_oxstates_1, fe2o3_oxstates_2)
        lifepo4_comp, lifepo4_oxstates = get_composition_oxidation_state(self.lifepo4)
        self.assertEqual(lifepo4_comp, Composition(self.lifepo4))
        self.assertDictEqual(lifepo4_oxstates, {})

    def test_descriptor_ionic_radii(self):
        ionic_radii = get_pymatgen_descriptor(self.nacl_formula_2, "ionic_radii")
        self.assertEqual(ionic_radii, [1.16, 1.67])
        with self.assertRaises(ValueError):
            get_pymatgen_descriptor(self.nacl_formula_1, "ionic_radii")
        ionic_radii = get_pymatgen_descriptor(self.fe2o3_formula_1, "ionic_radii")
        self.assertEqual(ionic_radii, [0.785, 0.785, 1.26, 1.26, 1.26])

    def test_descriptor_ionic_radii_from_composition(self):
        cscl = Composition({Specie("Cs", 1): 1, Specie("Cl", -1): 1})
        ionic_radii = get_pymatgen_descriptor(cscl, "ionic_radii")
        self.assertEqual(ionic_radii, [1.81, 1.67])
        ionic_radii_2 = get_pymatgen_descriptor("Cs+1Cl-1", "ionic_radii")
        self.assertEqual(ionic_radii, ionic_radii_2)


if __name__ == '__main__':
    unittest.main()
