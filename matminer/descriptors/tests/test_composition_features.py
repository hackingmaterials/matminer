from __future__ import unicode_literals, division, print_function

import unittest

from pymatgen.core.composition import Composition

from matminer.descriptors.composition_features import get_composition_oxidation_state, get_pymatgen_descriptor


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

    def test_get_pymatgen_descriptor(self):
        ionic_radii = get_pymatgen_descriptor(self.nacl_formula_2, "ionic_radii")
        self.assertEqual(ionic_radii, [1.16, 1.67])
        with self.assertRaises(ValueError):
            get_pymatgen_descriptor(self.nacl_formula_1, "ionic_radii")
        ionic_radii = get_pymatgen_descriptor(self.fe2o3_formula_1, "ionic_radii")
        self.assertEqual(ionic_radii, [0.785, 0.785, 1.26, 1.26, 1.26])


if __name__ == '__main__':
    unittest.main()
