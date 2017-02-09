# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals

import unittest2 as unittest

from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest

from matminer.descriptors.structure_features import get_packing_fraction, \
        get_vol_per_site, get_density, get_rdf, get_rdf_peaks, get_redf, \
        get_min_relative_distances, get_neighbors_of_site_with_index, \
        get_order_parameters


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

    def test_get_packing_fraction(self):
        self.assertAlmostEqual(int(1000 * get_packing_fraction(
                self.diamond)), 251)
        self.assertAlmostEqual(int(1000 * get_packing_fraction(
                self.nacl)), 620)
        self.assertAlmostEqual(int(1000 * get_packing_fraction(
                self.cscl)), 1043)

    def test_get_vol_per_site(self):
        self.assertAlmostEqual(int(1000 * get_vol_per_site(
                self.diamond)), 5710)
        self.assertAlmostEqual(int(1000 * get_vol_per_site(
                self.nacl)), 23046)
        self.assertAlmostEqual(int(1000 * get_vol_per_site(
                self.cscl)), 37282)

    def test_get_density(self):
        self.assertAlmostEqual(int(100 * get_density(
                self.diamond)), 349)
        self.assertAlmostEqual(int(100 * get_density(
                self.nacl)), 210)
        self.assertAlmostEqual(int(100 * get_density(
                self.cscl)), 374)

    def test_get_rdf(self):
        l = sorted([[k, v] for  k, v in get_rdf(self.diamond).items()])
        self.assertAlmostEqual(len(l), 116)
        self.assertAlmostEqual(int(10*l[0][0]), 15)
        self.assertAlmostEqual(int(1000*l[0][1]), 810)
        self.assertAlmostEqual(int(10*l[115][0]), 199)
        self.assertAlmostEqual(int(1000*l[115][1]), 41)
        l = sorted([[k, v] for  k, v in get_rdf(self.nacl).items()])
        self.assertAlmostEqual(len(l), 44)
        self.assertAlmostEqual(int(10*l[0][0]), 28)
        self.assertAlmostEqual(int(1000*l[0][1]), 578)
        self.assertAlmostEqual(int(10*l[43][0]), 199)
        self.assertAlmostEqual(int(1000*l[43][1]), 103)
        l = sorted([[k, v] for  k, v in get_rdf(self.cscl).items()])
        self.assertAlmostEqual(len(l), 32)
        self.assertAlmostEqual(int(10*l[0][0]), 36)
        self.assertAlmostEqual(int(1000*l[0][1]), 262)
        self.assertAlmostEqual(int(10*l[31][0]), 197)
        self.assertAlmostEqual(int(1000*l[31][1]), 26)

        #for s in [self.diamond, self.nacl, self.cscl]:
        #    d = get_rdf(s)
        #    l = sorted([[k, v] for  k, v in d.items()])
        #    for i, j in l:
        #        print("{} {}".format(i, j))
        #    print("")
        #    #print("{}".format(s.volume))
        #    #for site in s.sites:
        #    #    print("{} {}".format(site.species_string, site.specie.atomic_radius))

    def test_get_neighbors_of_site_with_index(self):
        self.assertAlmostEqual(len(get_neighbors_of_site_with_index(
                self.diamond, 0)), 4)
        self.assertAlmostEqual(len(get_neighbors_of_site_with_index(
                self.nacl, 0)), 6)
        self.assertAlmostEqual(len(get_neighbors_of_site_with_index(
                self.cscl, 0)), 8)

    def test_get_order_parameters(self):
        opvals = get_order_parameters(self.diamond)
        self.assertAlmostEqual(int(opvals[0][1] * 1000), 999)
        self.assertAlmostEqual(int(opvals[1][1] * 1000), 999)
        #self.assertAlmostEqual(int(opvals[0][37] * 1000), 999)
        #self.assertAlmostEqual(int(opvals[1][37] * 1000), 999)
        opvals = get_order_parameters(self.nacl)
        self.assertAlmostEqual(int(opvals[0][2] * 1000), 999)
        self.assertAlmostEqual(int(opvals[1][2] * 1000), 999)
        #self.assertAlmostEqual(int(opvals[0][38] * 1000), 999)
        #self.assertAlmostEqual(int(opvals[1][38] * 1000), 999)
        opvals = get_order_parameters(self.cscl)
        self.assertAlmostEqual(int(opvals[0][3] * 1000), 975)
        self.assertAlmostEqual(int(opvals[1][3] * 1000), 975)
        #self.assertAlmostEqual(int(opvals[0][39] * 1000), 975)
        #self.assertAlmostEqual(int(opvals[1][39] * 1000), 975)

    def tearDown(self):
        del self.diamond
        del self.nacl
        del self.cscl


if __name__ == '__main__':
    unittest.main()
