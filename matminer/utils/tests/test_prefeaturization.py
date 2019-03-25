import copy
import unittest

from pymatgen import Composition, Structure, Lattice, Element

from matminer.utils.prefeaturization import basic_composition_stats, \
    element_is_metal, basic_structure_stats


class TestPrefeaturization(unittest.TestCase):
    def setUp(self):
        self.ni3al = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ] + ["Ni"] * 3,
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False, site_properties=None)

        self.ni3al_partial = copy.deepcopy(self.ni3al)
        self.ni3al_partial.replace_species({"Al": "Al0.99H0.01"})
        self.sc = Structure(Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
                            ["Al"], [[0, 0, 0]], validate_proximity=False,
                            to_unit_cell=False,
                            coords_are_cartesian=False)
        self.test_structures = [self.ni3al, self.sc, self.ni3al_partial, self.sc]
        comps = ["CH4", "Li2O3", "WC", "PtAu", "LaO3", "AcTh", "Nd"]
        self.test_compositions = [Composition(c) for c in comps]

    def test_element_is_metal(self):
        # Alkali(ne)
        self.assertTrue(element_is_metal(Element("Li")))
        self.assertTrue(element_is_metal(Element("Mg")))

        # Transition metals
        self.assertTrue(element_is_metal(Element("W")))

        # Lathanides
        self.assertTrue(element_is_metal(Element("Nd")))
        self.assertTrue(element_is_metal(Element("La")))

        # Actinides
        self.assertTrue(element_is_metal(Element("Th")))
        self.assertTrue(element_is_metal(Element("Ac")))

        # Post transition metals
        self.assertTrue(element_is_metal(Element("Bi")))
        self.assertTrue(element_is_metal(Element("Al")))

        # Nonmetal
        self.assertFalse(element_is_metal(Element("C")))
        self.assertFalse(element_is_metal(Element("O")))

        # Metalloids
        self.assertFalse(element_is_metal(Element("Si")))
        self.assertFalse(element_is_metal(Element("Po")))

    def test_composition_percentage_stats(self):
        stats = basic_composition_stats(self.test_compositions)
        self.assertAlmostEqual(stats["fraction_all_metal"], 3/7)
        self.assertAlmostEqual(stats["fraction_all_rare_earth_metal"], 2/7)
        self.assertAlmostEqual(stats["fraction_all_transition_metal"], 1/7)
        self.assertAlmostEqual(stats["fraction_contains_metal"], 6/7)
        self.assertAlmostEqual(stats["fraction_contains_transition_metal"], 4/7)
        self.assertAlmostEqual(stats["fraction_contains_rare_earth_metal"], 3/7)

    def test_structure_percentage_stats(self):
        stats = basic_structure_stats(self.test_structures)
        self.assertAlmostEqual(stats["fraction_ordered"], 0.75, places=2)
        self.assertEqual(stats["max_n_sites"], 4)
        self.assertEqual(stats["min_n_sites"], 1)
        self.assertAlmostEqual(stats["avg_n_sites"], 2.5, places=2)
