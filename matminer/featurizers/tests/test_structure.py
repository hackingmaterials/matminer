# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals, division

import unittest

import numpy as np

from pymatgen import Structure, Lattice, Molecule
from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.structure import DensityFeatures, \
    RadialDistributionFunction, \
    RadialDistributionFunctionPeaks, PartialRadialDistributionFunction, \
    ElectronicRadialDistributionFunction, \
    MinimumRelativeDistances, \
    OPStructureFingerprint, \
    CoulombMatrix, SineCoulombMatrix, OrbitalFieldMatrix, GlobalSymmetryFeatures


class StructureFeaturesTest(PymatgenTest):
    def setUp(self):
        self.diamond = Structure(
            Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264],
                     [0, 0, 2.528]]), ["C0+", "C0+"], [[2.554, 1.806, 4.423],
                                                       [0.365, 0.258, 0.632]],
            validate_proximity=False,
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
                                                         [2.324, 1.643, 4.025]],
            validate_proximity=False,
            to_unit_cell=False, coords_are_cartesian=True,
            site_properties=None)
        self.cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"], [[2.105, 2.105, 2.105], [0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=True, site_properties=None)
        self.ni3al = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ] + ["Ni"] * 3,
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False, site_properties=None)
        self.bond_angles = range(5, 180, 5)

    def test_density_features(self):
        df = DensityFeatures()
        f = df.featurize(self.diamond)
        self.assertAlmostEqual(f[0], 3.49, 2)
        self.assertAlmostEqual(f[1], 5.71, 2)
        self.assertAlmostEqual(f[2], 0.25, 2)

        f = df.featurize(self.nacl)
        self.assertAlmostEqual(f[0], 2.105, 2)
        self.assertAlmostEqual(f[1], 23.046, 2)
        self.assertAlmostEqual(f[2], 0.620, 2)

    def test_global_symmetry(self):
        gsf = GlobalSymmetryFeatures()
        self.assertEqual(gsf.featurize(self.diamond), [227, "cubic", 1, True])

    def test_rdf_and_peaks(self):
        ## Test diamond
        rdforig = RadialDistributionFunction().featurize(
            self.diamond)
        rdf = rdforig[0]

        # Make sure it the last bin is cutoff-bin_max
        self.assertAlmostEqual(max(rdf['distances']), 19.9)

        # Verify bin sizes
        self.assertEqual(len(rdf['distribution']), 200)

        # Make sure it gets all of the peaks
        self.assertEqual(np.count_nonzero(rdf['distribution']), 116)

        # Check the values for a few individual peaks
        self.assertAlmostEqual(
            rdf['distribution'][int(round(1.5 / 0.1))], 15.12755155)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(2.9 / 0.1))], 12.53193948)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(19.9 / 0.1))], 0.822126129)

        # Make sure it finds the locations of non-zero peaks correctly
        peaks = RadialDistributionFunctionPeaks().featurize(rdforig)[0]
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(2.5, peaks[0])
        self.assertAlmostEqual(1.5, peaks[1])

        # Repeat test with NaCl (omitting comments). Altering cutoff distance
        rdforig = RadialDistributionFunction(cutoff=10).featurize(self.nacl)
        rdf = rdforig[0]
        self.assertAlmostEqual(max(rdf['distances']), 9.9)
        self.assertEqual(len(rdf['distribution']), 100)
        self.assertEqual(np.count_nonzero(rdf['distribution']), 11)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(2.8 / 0.1))], 27.09214168)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(4.0 / 0.1))], 26.83338723)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(9.8 / 0.1))], 3.024406467)

        peaks = RadialDistributionFunctionPeaks().featurize(rdforig)[0]
        self.assertEqual(len(peaks), 2)
        self.assertAlmostEqual(2.8, peaks[0])
        self.assertAlmostEqual(4.0, peaks[1])

        # Repeat test with CsCl. Altering cutoff distance and bin_size
        rdforig = RadialDistributionFunction(
            cutoff=8, bin_size=0.5).featurize(self.cscl)
        rdf = rdforig[0]
        self.assertAlmostEqual(max(rdf['distances']), 7.5)
        self.assertEqual(len(rdf['distribution']), 16)
        self.assertEqual(np.count_nonzero(rdf['distribution']), 5)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(3.5 / 0.5))], 6.741265585)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(4.0 / 0.5))], 3.937582548)
        self.assertAlmostEqual(
            rdf['distribution'][int(round(7.0 / 0.5))], 1.805505363)

        peaks = RadialDistributionFunctionPeaks(n_peaks=3).featurize(
            rdforig)[0]
        self.assertEqual(len(peaks), 3)
        self.assertAlmostEqual(3.5, peaks[0])
        self.assertAlmostEqual(6.5, peaks[1])
        self.assertAlmostEqual(5, 5, peaks[2])

    def test_prdf(self):
        # Test a few peaks in diamond
        # These expected numbers were derived by performing
        # the calculation in another code
        prdf = PartialRadialDistributionFunction().featurize(self.diamond)[0]
        self.assertEqual(len(prdf.values()), 1)
        self.assertAlmostEqual(
            prdf[('C', 'C')]['distribution'][int(round(1.4 / 0.1))], 0)
        self.assertAlmostEqual(
            prdf[('C', 'C')]['distribution'][int(round(1.5 / 0.1))], 1.32445167622)
        self.assertAlmostEqual(max(prdf[('C', 'C')]['distances']), 20.0)
        self.assertAlmostEqual(
            prdf[('C', 'C')]['distribution'][int(round(19.9 / 0.1))], 0.07197902)

        # Test a few peaks in CsCl, make sure it gets all types correctly
        prdf = PartialRadialDistributionFunction(cutoff=10).featurize(
            self.cscl)[0]
        self.assertEqual(len(prdf.values()), 4)
        self.assertAlmostEqual(max(prdf[('Cs', 'Cl')]['distances']), 10.0)
        self.assertAlmostEqual(
            prdf[('Cs', 'Cl')]['distribution'][int(round(3.6 / 0.1))], 0.477823197)
        self.assertAlmostEqual(
            prdf[('Cl', 'Cs')]['distribution'][int(round(3.6 / 0.1))], 0.477823197)
        self.assertAlmostEqual(
            prdf[('Cs', 'Cs')]['distribution'][int(round(3.6 / 0.1))], 0)

        # Do Ni3Al, make sure it captures the antisymmetry of Ni/Al sites
        prdf = PartialRadialDistributionFunction(
            cutoff=10, bin_size=0.5).featurize(self.ni3al)[0]
        self.assertEqual(len(prdf.values()), 4)
        self.assertAlmostEqual(
            prdf[('Ni', 'Al')]['distribution'][int(round(2 / 0.5))], 0.125236677)
        self.assertAlmostEqual(
            prdf[('Al', 'Ni')]['distribution'][int(round(2 / 0.5))], 0.37571003)
        self.assertAlmostEqual(
            prdf[('Al', 'Al')]['distribution'][int(round(2 / 0.5))], 0)

    def test_redf(self):
        d = ElectronicRadialDistributionFunction().featurize(
            self.diamond)[0]
        self.assertAlmostEqual(int(1000 * d["distances"][0]), 25)
        self.assertAlmostEqual(int(1000 * d["distribution"][0]), 0)
        self.assertAlmostEqual(int(1000 * d["distances"][len(
            d["distances"]) - 1]), 6175)
        self.assertAlmostEqual(int(1000 * d["distribution"][len(
            d["distances"]) - 1]), 0)
        d = ElectronicRadialDistributionFunction().featurize(
            self.nacl)[0]
        self.assertAlmostEqual(int(1000 * d["distances"][0]), 25)
        self.assertAlmostEqual(int(1000 * d["distribution"][0]), 0)
        self.assertAlmostEqual(int(1000 * d["distances"][56]), 2825)
        self.assertAlmostEqual(int(1000 * d["distribution"][56]), -2108)
        self.assertAlmostEqual(int(1000 * d["distances"][len(
            d["distances"]) - 1]), 9875)
        d = ElectronicRadialDistributionFunction().featurize(
            self.cscl)[0]
        self.assertAlmostEqual(int(1000 * d["distances"][0]), 25)
        self.assertAlmostEqual(int(1000 * d["distribution"][0]), 0)
        self.assertAlmostEqual(int(1000 * d["distances"][72]), 3625)
        self.assertAlmostEqual(int(1000 * d["distribution"][72]), -2194)
        self.assertAlmostEqual(int(1000 * d["distances"][len(
            d["distances"]) - 1]), 7275)

    def test_coulomb_matrix(self):
        species = ["C", "C", "H", "H"]
        coords = [[0, 0, 0], [0, 0, 1.203], [0, 0, -1.06], [0, 0, 2.263]]
        acetylene = Molecule(species, coords)
        morig = CoulombMatrix().featurize(acetylene)
        mtarget = [[36.858, 15.835391290, 2.995098235, 1.402827813], \
                   [15.835391290, 36.858, 1.4028278132103624, 2.9950982], \
                   [2.9368896127, 1.402827813, 0.5, 0.159279959], \
                   [1.4028278132, 2.995098235, 0.159279959, 0.5]]
        self.assertAlmostEqual(
            int(np.linalg.norm(morig - np.array(mtarget))), 0)
        m = CoulombMatrix(False).featurize(acetylene)[0]
        self.assertAlmostEqual(m[0][0], 0.0)
        self.assertAlmostEqual(m[1][1], 0.0)
        self.assertAlmostEqual(m[2][2], 0.0)
        self.assertAlmostEqual(m[3][3], 0.0)

    def test_sine_coulomb_matrix(self):
        scm = SineCoulombMatrix()
        sin_mat = scm.featurize(self.diamond)
        mtarget = [[36.8581, 6.147068], [6.147068, 36.8581]]
        self.assertAlmostEqual(
            np.linalg.norm(sin_mat - np.array(mtarget)), 0.0, places=4)
        scm = SineCoulombMatrix(False)
        sin_mat = scm.featurize(self.diamond)[0]
        self.assertEqual(sin_mat[0][0], 0)
        self.assertEqual(sin_mat[1][1], 0)

    def test_orbital_field_matrix(self):
        ofm_maker = OrbitalFieldMatrix()
        ofm = ofm_maker.featurize(self.diamond)[0]
        mtarget = np.zeros((32, 32))
        mtarget[1][1] = 1.4789015  # 1.3675444
        mtarget[1][3] = 1.4789015  # 1.3675444
        mtarget[3][1] = 1.4789015  # 1.3675444
        mtarget[3][3] = 1.4789015  # 1.3675444 if for a coord# of exactly 4
        for i in range(32):
            for j in range(32):
                if not i in [1, 3] and not j in [1, 3]:
                    self.assertEqual(ofm[i, j], 0.0)
        mtarget = np.matrix(mtarget)
        self.assertAlmostEqual(
            np.linalg.norm(ofm - mtarget), 0.0, places=4)

        ofm_maker = OrbitalFieldMatrix(True)
        ofm = ofm_maker.featurize(self.diamond)[0]
        mtarget = np.zeros((39, 39))
        mtarget[1][1] = 1.4789015
        mtarget[1][3] = 1.4789015
        mtarget[3][1] = 1.4789015
        mtarget[3][3] = 1.4789015
        mtarget[1][33] = 1.4789015
        mtarget[3][33] = 1.4789015
        mtarget[33][1] = 1.4789015
        mtarget[33][3] = 1.4789015
        mtarget[33][33] = 1.4789015
        mtarget = np.matrix(mtarget)
        self.assertAlmostEqual(
            np.linalg.norm(ofm - mtarget), 0.0, places=4)

    def test_min_relative_distances(self):
        self.assertAlmostEqual(int(
            1000 * MinimumRelativeDistances().featurize(
                self.diamond_no_oxi)[0][0]), 1105)
        self.assertAlmostEqual(int(
            1000 * MinimumRelativeDistances().featurize(
                self.nacl)[0][0]), 1005)
        self.assertAlmostEqual(int(
            1000 * MinimumRelativeDistances().featurize(
                self.cscl)[0][0]), 1006)

    def test_op_structure_fingerprint(self):
        # Test matrix.
        op_struct_fp = OPStructureFingerprint(stats=None)
        opvals = op_struct_fp.featurize(self.diamond)
        oplabels = op_struct_fp.feature_labels()
        self.assertAlmostEqual(int(opvals[10][0] * 1000 + 0.5), 1000)
        self.assertAlmostEqual(int(opvals[10][1] * 1000 + 0.5), 1000)
        opvals = op_struct_fp.featurize(self.nacl)
        self.assertAlmostEqual(int(opvals[16][0] * 1000 + 0.5), 1000)
        self.assertAlmostEqual(int(opvals[16][1] * 1000 + 0.5), 1000)
        opvals = op_struct_fp.featurize(self.cscl)
        self.assertAlmostEqual(int(opvals[20][0] * 1000), 998)
        self.assertAlmostEqual(int(opvals[20][1] * 1000), 998)

        # Test stats.
        op_struct_fp = OPStructureFingerprint()
        opvals = op_struct_fp.featurize(self.diamond)
        self.assertAlmostEqual(int(opvals[0] * 10000 + 0.5), 5)
        self.assertAlmostEqual(int(opvals[1] * 10000 + 0.5), 0)
        self.assertAlmostEqual(int(opvals[2] * 10000 + 0.5), 5)
        self.assertAlmostEqual(int(opvals[3] * 10000 + 0.5), 5)
        self.assertAlmostEqual(int(opvals[4] * 10000 + 0.5), 5)
        self.assertAlmostEqual(int(opvals[32] * 1000 + 0.5), 38)
        self.assertAlmostEqual(int(opvals[40] * 1000 + 0.5), 1000)
        self.assertAlmostEqual(int(opvals[41] * 1000 + 0.5), 0)
        self.assertAlmostEqual(int(opvals[42] * 1000 + 0.5), 1000)
        self.assertAlmostEqual(int(opvals[43] * 1000 + 0.5), 1000)
        self.assertAlmostEqual(int(opvals[44] * 1000 + 0.5), 1)
        for i in range(52, len(opvals)):
            self.assertAlmostEqual(int(opvals[i] * 500 + 0.5), 0)

    def tearDown(self):
        del self.diamond
        del self.nacl
        del self.cscl


if __name__ == '__main__':
    unittest.main()
