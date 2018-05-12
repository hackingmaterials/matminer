import unittest
from unittest import TestCase

from math import isnan
from pymatgen.core.periodic_table import Specie

from matminer.utils.data import DemlData, MagpieData, PymatgenData, \
    MixingEnthalpy
from pymatgen import Element


class TestDemlData(TestCase):
    """Tests for the DemlData Class"""

    def setUp(self):
        self.data_source = DemlData()

    def test_get_property(self):
        self.assertAlmostEquals(-4.3853, self.data_source.get_elemental_property(Element("Bi"), "mus_fere"), 4)
        self.assertEquals(59600, self.data_source.get_elemental_property(Element("Li"), "electron_affin"))
        self.assertAlmostEquals(2372300, self.data_source.get_elemental_property(Element("He"), "first_ioniz"))
        self.assertAlmostEquals(sum([2372300,5250500]),
                                self.data_source.get_charge_dependent_property_from_specie(Specie("He", 2),
                                                                                           "total_ioniz"))
        self.assertAlmostEquals(18.6, self.data_source.get_charge_dependent_property_from_specie(Specie("V", 3),
                                                                                                 "xtal_field_split"))

    def test_get_oxidation(self):
        self.assertEquals([1], self.data_source.get_oxidation_states(Element("Li")))


class TestMagpieData(TestCase):

    def setUp(self):
        self.data_source = MagpieData()

    def test_get_property(self):
        self.assertAlmostEquals(9.012182, self.data_source.get_elemental_property(Element("Be"), "AtomicWeight"))

    def test_get_oxidation(self):
        self.assertEquals([-4, 2, 4], self.data_source.get_oxidation_states(Element("C")))


class TestPymatgenData(TestCase):

    def setUp(self):
        self.data_source = PymatgenData()

    def test_get_property(self):
        self.assertAlmostEquals(9.012182, self.data_source.get_elemental_property(Element("Be"), "atomic_mass"))
        self.assertAlmostEquals(1.26, self.data_source.get_charge_dependent_property(Element("Ac"), 3, "ionic_radii"))

    def test_get_oxidation(self):
        self.assertEquals((3,), self.data_source.get_oxidation_states(Element("Nd")))
        self.data_source.use_common_oxi_states = False
        self.assertEquals((2, 3), self.data_source.get_oxidation_states(Element("Nd")))


class TestMixingEnthalpy(TestCase):

    def setUp(self):
        self.data = MixingEnthalpy()

    def test_get_data(self):
        self.assertEquals(-27, self.data.get_mixing_enthalpy(Element('H'),
                                                             Element('Pd')))
        self.assertEquals(-27, self.data.get_mixing_enthalpy(Element('Pd'),
                                                             Element('H')))
        self.assertTrue(isnan(self.data.get_mixing_enthalpy(Element('He'),
                                                            Element('H'))))

if __name__ == "__main__":
    unittest.main()
