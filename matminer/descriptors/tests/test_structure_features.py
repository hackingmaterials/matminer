# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals

import unittest2 as unittest

from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from matminer.descriptors.structure_features import get_order_parameters, \
        get_neighbors_of_site_with_index

class StructureFeaturesTest(PymatgenTest):

    def setUp(self):
        self.diamond = Structure(
                Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264],
                [0, 0, 2.528]]), ["C", "C"], [[2.554, 1.806, 4.423],
                [0.365, 0.258, 0.632]], validate_proximity=False,
                to_unit_cell=False, coords_are_cartesian=True,
                site_properties=None)
        self.nacl = Structure(
                Lattice([[3.485, 0, 2.012], [1.162, 3.286, 2.012],
                [0, 0, 4.025]]), ["Na", "Cl"], [[0, 0, 0],
                [2.324, 1.643, 4.025]], validate_proximity=False,
                to_unit_cell=False, coords_are_cartesian=True,
                site_properties=None)
        self.cscl = Structure(
                Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
                ["Cl", "Cs"], [[2.105,2.105, 2.105], [0, 0, 0]],
                validate_proximity=False, to_unit_cell=False,
                coords_are_cartesian=True, site_properties=None)

    def test_get_neighbors_of_site_with_index(self):
        self.assertAlmostEqual(len(get_neighbors_of_site_with_index(
                self.diamond, 0)), 4)
        self.assertAlmostEqual(len(get_neighbors_of_site_with_index(
                self.nacl, 0)), 6)
        self.assertAlmostEqual(len(get_neighbors_of_site_with_index(
                self.cscl, 0)), 8)

    def test_get_order_parameters(self):
        opvals = get_order_parameters(self.diamond)
        self.assertAlmostEqual(int(opvals[0][37] * 1000), 999)
        self.assertAlmostEqual(int(opvals[1][37] * 1000), 999)
        opvals = get_order_parameters(self.nacl)
        self.assertAlmostEqual(int(opvals[0][38] * 1000), 999)
        self.assertAlmostEqual(int(opvals[1][38] * 1000), 999)
        opvals = get_order_parameters(self.cscl)
        self.assertAlmostEqual(int(opvals[0][39] * 1000), 975)
        self.assertAlmostEqual(int(opvals[1][39] * 1000), 975)

    def tearDown(self):
        del self.diamond
        del self.nacl
        del self.cscl


if __name__ == '__main__':
    unittest.main()
