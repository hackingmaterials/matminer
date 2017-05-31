# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals

import numpy as np

import unittest2 as unittest

from pymatgen import Structure, Lattice, Molecule
from pymatgen.util.testing import PymatgenTest

from matminer.descriptors.structure_features import get_packing_fraction, \
        get_vol_per_site, get_density, get_rdf, get_rdf_peaks, get_redf, \
        get_min_relative_distances, get_neighbors_of_site_with_index, \
        get_order_parameters, get_order_parameter_stats, get_prdf, \
        get_coulomb_matrix


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
        self.ni3al = Structure(
            Lattice([[3.52,0,0],[0,3.52,0],[0,0,3.52]]),
            ["Al",]+["Ni"]*3, 
            [[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False, site_properties=None)

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
        ## Test diamond
        rdf, bin_radius = get_rdf(self.diamond)
        
        # Make sure it the last bin is cutoff-bin_max
        self.assertAlmostEquals(bin_radius.max(), 19.9)
        
        # Verify bin sizes
        self.assertEquals(len(rdf), len(bin_radius))
        self.assertEquals(len(rdf), 200)
        
        # Make sure it gets all of the peaks
        self.assertEquals(np.count_nonzero(rdf), 116) 
        
        # Check the values for a few individual peaks
        self.assertAlmostEqual(rdf[round(1.5/0.1)], 15.12755155)
        self.assertAlmostEqual(rdf[round(2.9/0.1)], 12.53193948)
        self.assertAlmostEqual(rdf[round(19.9/0.1)], 0.822126129)
        
        # Make sure it finds the locations of non-zero peaks correctly
        peaks = get_rdf_peaks(rdf, bin_radius)
        self.assertEquals(len(peaks), 2)
        self.assertAlmostEquals(2.5, peaks[0])
        self.assertAlmostEquals(1.5, peaks[1])
        
        # Repeat test with NaCl (omitting comments). Altering cutoff distance
        rdf, bin_radius = get_rdf(self.nacl, cutoff=10)
        self.assertAlmostEquals(bin_radius.max(), 9.9)
        self.assertEquals(len(rdf), len(bin_radius))
        self.assertEquals(len(rdf), 100)
        self.assertEquals(np.count_nonzero(rdf), 11) 
        self.assertAlmostEqual(rdf[round(2.8/0.1)], 27.09214168)
        self.assertAlmostEqual(rdf[round(4.0/0.1)], 26.83338723)
        self.assertAlmostEqual(rdf[round(9.8/0.1)], 3.024406467)
        
        peaks = get_rdf_peaks(rdf, bin_radius)
        self.assertEquals(len(peaks), 2)
        self.assertAlmostEquals(2.8, peaks[0])
        self.assertAlmostEquals(4.0, peaks[1])
        
        # Repeat test with CsCl. Altering cutoff distance and bin_size
        rdf, bin_radius = get_rdf(self.cscl, cutoff=8, bin_size=0.5)
        self.assertAlmostEquals(bin_radius.max(), 7.5)
        self.assertEquals(len(rdf), len(bin_radius))
        self.assertEquals(len(rdf), 16)
        self.assertEquals(np.count_nonzero(rdf), 5) 
        self.assertAlmostEqual(rdf[round(3.5/0.5)], 6.741265585)
        self.assertAlmostEqual(rdf[round(4.0/0.5)], 3.937582548)
        self.assertAlmostEqual(rdf[round(7.0/0.5)], 1.805505363)
        
        peaks = get_rdf_peaks(rdf, bin_radius, 3)
        self.assertEquals(len(peaks), 3)
        self.assertAlmostEquals(3.5, peaks[0])
        self.assertAlmostEquals(6.5, peaks[1])
        self.assertAlmostEquals(5,5, peaks[2])
        
    def test_prdf(self):
        # Test a few peaks in diamond
        #  These expected numbers were derived by performing the calculation in another code
        p, r = get_prdf(self.diamond)
        self.assertEquals(len(p), 1)
        self.assertEquals(p[('C','C')][round(1.4/0.1)], 0)
        self.assertAlmostEqual(p[('C','C')][round(1.5/0.1)], 1.324451676)
        self.assertAlmostEqual(r.max(), 19.9)
        self.assertAlmostEqual(p[('C','C')][round(19.9/0.1)], 0.07197902)
        
        # Test a few peaks in CsCl, make sure it gets all types correctly
        p, r = get_prdf(self.cscl, cutoff=10)
        self.assertEquals(len(p), 4)
        self.assertAlmostEqual(r.max(), 9.9)
        self.assertAlmostEquals(p[('Cs','Cl')][round(3.6/0.1)], 0.477823197)
        self.assertAlmostEquals(p[('Cl','Cs')][round(3.6/0.1)], 0.477823197)
        self.assertAlmostEquals(p[('Cs','Cs')][round(3.6/0.1)], 0)
        
        # Do Ni3Al, make sure it captures the antisymmetry of Ni/Al sites
        p,r = get_prdf(self.ni3al, cutoff=10, bin_size=0.5)
        self.assertEquals(len(p), 4)
        self.assertAlmostEquals(p[('Ni','Al')][round(2/0.5)], 0.125236677)
        self.assertAlmostEquals(p[('Al','Ni')][round(2/0.5)], 0.37571003)
        self.assertAlmostEquals(p[('Al','Al')][round(2/0.5)], 0)

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

    def test_get_coulomb_matrix(self):
        species = ["C", "C", "H", "H"]
        coords = [[0,0,0], [0,0,1.203], [0,0,-1.06], [0,0,2.263]]
        acetylene = Molecule(species, coords)
        morig = get_coulomb_matrix(acetylene, diag_elems=True)
        mtarget = [[36.858,29.925,5.66,2.651],[29.925,36.858,2.651,5.66],\
                [5.55,2.651,0.5,0.301],[2.651,5.66,0.301,0.5]]
        self.assertAlmostEqual(
                int(np.linalg.norm(morig - np.array(mtarget))), 0)
        m = get_coulomb_matrix(acetylene)
        self.assertAlmostEqual(m[0][0], 0.0)
        self.assertAlmostEqual(m[1][1], 0.0)
        self.assertAlmostEqual(m[2][2], 0.0)
        self.assertAlmostEqual(m[3][3], 0.0)

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
