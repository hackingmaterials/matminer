import unittest
from unittest import TestCase

from matminer.utils.data import DemlData, MagpieData, PymatgenData
from pymatgen import Element


class TestDemlData(TestCase):
    """Tests for the DemlData Class"""

    def setUp(self):
        self.data_source = DemlData()

    def test_get_property(self):
        self.assertAlmostEquals(-4.3853, self.data_source.get_elemental_property(Element("Bi"), "mus_fere"), 4)
        self.assertEquals(59600, self.data_source.get_elemental_property(Element("Li"), "electron_affin"))

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

    def test_get_oxidation(self):
        self.assertEquals((3,), self.data_source.get_oxidation_states(Element("Nd"), common=True))
        self.assertEquals((2, 3), self.data_source.get_oxidation_states(Element("Nd"), common=False))

if __name__ == "__main__":
    unittest.main()
