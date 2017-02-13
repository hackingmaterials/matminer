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
        get_order_parameters, get_order_parameter_stats


class StructureFeaturesTest(PymatgenTest):

    def setUp(self):
        self.diamond = Structure(
                Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264],
                [0, 0, 2.528]]), ["C0+", "C0+"], [[2.554, 1.806, 4.423],
                [0.365, 0.258, 0.632]], validate_proximity=False,
                to_unit_cell=False, coords_are_cartesian=True,
                site_properties=None)
        self.diamond_no_oxi = Structure(
                Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264],
                [0, 0, 2.528]]), ["C", "C"], [[2.554, 1.806, 4.423],
                [0.365, 0.258, 0.632]], validate_proximity=False,
                to_unit_cell=False, coords_are_cartesian=True,
                site_properties=None)
        self.nacl = Structure(
                Lattice([[3.485, 0, 2.012], [1.162, 3.286, 2.012],
                [0, 0, 4.025]]), ["Na1+", "Cl1-"], [[0, 0, 0],
                [2.324, 1.643, 4.025]], validate_proximity=False,
                to_unit_cell=False, coords_are_cartesian=True,
                site_properties=None)
        self.cscl = Structure(
                Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
                ["Cl1-", "Cs1+"], [[2.105,2.105, 2.105], [0, 0, 0]],
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

    def test_get_rdf_and_peaks(self):
        d = get_rdf(self.diamond)
        l = sorted([[k, v] for  k, v in d.items()])
        self.assertAlmostEqual(len(l), 116)
        self.assertAlmostEqual(int(10*l[0][0]), 15)
        self.assertAlmostEqual(int(1000*l[0][1]), 810)
        self.assertAlmostEqual(int(10*l[115][0]), 199)
        self.assertAlmostEqual(int(1000*l[115][1]), 41)
        l = get_rdf_peaks(d)
        self.assertAlmostEqual(int(10 * l[0]), 25)
        self.assertAlmostEqual(int(10 * l[1]), 15)
        d = get_rdf(self.nacl)
        l = sorted([[k, v] for  k, v in d.items()])
        self.assertAlmostEqual(len(l), 44)
        self.assertAlmostEqual(int(10*l[0][0]), 28)
        self.assertAlmostEqual(int(1000*l[0][1]), 578)
        self.assertAlmostEqual(int(10*l[43][0]), 199)
        self.assertAlmostEqual(int(1000*l[43][1]), 103)
        l = get_rdf_peaks(d)
        self.assertAlmostEqual(int(10 * l[0]), 28)
        self.assertAlmostEqual(int(10 * l[1]), 40)
        d = get_rdf(self.cscl)
        l = sorted([[k, v] for  k, v in d.items()])
        self.assertAlmostEqual(len(l), 32)
        self.assertAlmostEqual(int(10*l[0][0]), 36)
        self.assertAlmostEqual(int(1000*l[0][1]), 262)
        self.assertAlmostEqual(int(10*l[31][0]), 197)
        self.assertAlmostEqual(int(1000*l[31][1]), 26)
        l = get_rdf_peaks(d)
        self.assertAlmostEqual(int(10 * l[0]), 36)
        self.assertAlmostEqual(int(10 * l[1]), 69)

    def test_get_redf(self):
        d = get_redf(self.diamond)
        self.assertAlmostEqual(int(1000 * d["distances"][0]), 25)
        self.assertAlmostEqual(int(1000 * d["redf"][0]), 0)
        self.assertAlmostEqual(int(1000 * d["distances"][len(
                d["distances"])-1]), 6175)
        self.assertAlmostEqual(int(1000 * d["redf"][len(
                d["redf"])-1]), 0)
        d = get_redf(self.nacl)
        self.assertAlmostEqual(int(1000 * d["distances"][0]), 25)
        self.assertAlmostEqual(int(1000 * d["redf"][0]), 0)
        self.assertAlmostEqual(int(1000 * d["distances"][len(
                d["distances"])-1]), 9875)
        self.assertAlmostEqual(int(1000 * d["redf"][len(
                d["redf"])-1]), 202)
        d = get_redf(self.cscl)
        self.assertAlmostEqual(int(1000 * d["distances"][0]), 25)
        self.assertAlmostEqual(int(1000 * d["redf"][0]), 0)
        self.assertAlmostEqual(int(1000 * d["distances"][len(
                d["distances"])-1]), 7275)
        self.assertAlmostEqual(int(1000 * d["redf"][len(
                d["redf"])-1]), 1097)

    def test_get_min_relative_distances(self):
        self.assertAlmostEqual(int(1000*get_min_relative_distances(
                self.diamond_no_oxi)[0]), 1105)
        self.assertAlmostEqual(int(1000*get_min_relative_distances(
                self.nacl)[0]), 1005)
        self.assertAlmostEqual(int(1000*get_min_relative_distances(
                self.cscl)[0]), 1006)

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

    def test_get_order_parameter_stats(self):
        opstats = get_order_parameter_stats(self.diamond)
        self.assertAlmostEqual(int(opstats["tet"]["min"] * 1000), 999)
        self.assertAlmostEqual(int(opstats["tet"]["max"] * 1000), 999)
        self.assertAlmostEqual(int(opstats["tet"]["mean"] * 1000), 999)
        self.assertAlmostEqual(int(opstats["tet"]["std"] * 1000), 0)
        opstats = get_order_parameter_stats(self.nacl)
        self.assertAlmostEqual(int(opstats["oct"]["min"] * 1000), 999)
        self.assertAlmostEqual(int(opstats["oct"]["max"] * 1000), 999)
        self.assertAlmostEqual(int(opstats["oct"]["mean"] * 1000), 999)
        self.assertAlmostEqual(int(opstats["oct"]["std"] * 1000), 0)
        opstats = get_order_parameter_stats(self.cscl)
        self.assertAlmostEqual(int(opstats["bcc"]["min"] * 1000), 975)
        self.assertAlmostEqual(int(opstats["bcc"]["max"] * 1000), 975)
        self.assertAlmostEqual(int(opstats["bcc"]["mean"] * 1000), 975)
        self.assertAlmostEqual(int(opstats["bcc"]["std"] * 1000), 0)

    def tearDown(self):
        del self.diamond
        del self.nacl
        del self.cscl


if __name__ == '__main__':
    unittest.main()
