import unittest2 as unittest

from matminer.descriptors.composition_features import get_stoich_attributes, get_elem_property_attributes, get_valence_orbital_attributes, get_ionic_attributes
from pymatgen.util.testing import PymatgenTest

class CompositionFeaturesTest(PymatgenTest):

    def test_stoich(self):
        self.assertAlmostEqual(get_stoich_attributes("Fe2O3", 0), 2)
        self.assertAlmostEqual(get_stoich_attributes("Fe2O3", 7), 0.604895199)

    def test_elem(self):
        all_attr = get_elem_property_attributes("Fe2O3")
        atomic_no_attr = all_attr[0]
        self.assertAlmostEqual(atomic_no_attr[0], 8)
        self.assertAlmostEqual(atomic_no_attr[1], 26)
        self.assertAlmostEqual(atomic_no_attr[2], 18)
        self.assertAlmostEqual(atomic_no_attr[3], 15.2)
        self.assertAlmostEqual(atomic_no_attr[4], 8.64)
        self.assertAlmostEqual(atomic_no_attr[5], 8)

    def test_valence(self):
        Fs, Fp, Fd, Ff = get_valence_orbital_attributes("Fe2O3")
        self.assertAlmostEqual(Fs, 0.294117647)
        self.assertAlmostEqual(Fp, 0.352941176)
        self.assertAlmostEqual(Fd, 0.352941176)
        self.assertAlmostEqual(Ff, 0)

    def test_ionic(self):
        cpd_possible, max_ionic_char, avg_ionic_char = get_ionic_attributes("Fe2O3")
        self.assertEqual(cpd_possible, True)
        self.assertAlmostEqual(max_ionic_char, 0.476922164)
        self.assertAlmostEqual(avg_ionic_char, 0.114461319)

if __name__=="__main__":
    unittest.main()
