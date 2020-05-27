# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import unicode_literals, division
import os
import copy
import unittest
import csv
import json
import numpy as np
import pandas as pd
from multiprocessing import set_start_method
from sklearn.exceptions import NotFittedError

from pymatgen import Structure, Lattice, Molecule
from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import SiteElementalProperty
from matminer.featurizers.structure import DensityFeatures, \
    RadialDistributionFunction, PartialRadialDistributionFunction, \
    ElectronicRadialDistributionFunction, \
    MinimumRelativeDistances, SiteStatsFingerprint, CoulombMatrix, \
    SineCoulombMatrix, OrbitalFieldMatrix, GlobalSymmetryFeatures, \
    EwaldEnergy, BondFractions, BagofBonds, StructuralHeterogeneity, \
    MaximumPackingEfficiency, ChemicalOrdering, StructureComposition, \
    Dimensionality, XRDPowderPattern, CGCNNFeaturizer, JarvisCFID, \
    GlobalInstabilityIndex, \
    StructuralComplexity

# For the CGCNNFeaturizer
try:
    import torch
    import cgcnn
except ImportError:
    torch, cgcnn = None, None

test_dir = os.path.join(os.path.dirname(__file__))


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
                                                   [0.365, 0.258, 0.632]],
            validate_proximity=False,
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
            ["Cl1-", "Cs1+"], [[2.105, 2.1045, 2.1045], [0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=True, site_properties=None)
        self.ni3al = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ] + ["Ni"] * 3,
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False, site_properties=None)
        self.sc = Structure(Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al"], [[0, 0, 0]], validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)
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

        nacl_disordered = copy.deepcopy(self.nacl)
        nacl_disordered.replace_species({"Cl1-": "Cl0.99H0.01"})
        self.assertFalse(df.precheck(nacl_disordered))
        structures = [self.diamond, self.nacl, nacl_disordered]
        df2 = pd.DataFrame({"structure": structures})
        self.assertAlmostEqual(df.precheck_dataframe(df2, "structure"), 2 / 3)

    def test_global_symmetry(self):
        gsf = GlobalSymmetryFeatures()
        self.assertEqual(gsf.featurize(self.diamond), [227, "cubic", 1, True])

    def test_dimensionality(self):
        cscl = PymatgenTest.get_structure("CsCl")
        graphite = PymatgenTest.get_structure("Graphite")

        df = Dimensionality()

        self.assertEqual(df.featurize(cscl)[0], 3)
        self.assertEqual(df.featurize(graphite)[0], 2)

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

    def test_prdf(self):
        # Test a few peaks in diamond
        # These expected numbers were derived by performing
        # the calculation in another code
        distances, prdf = PartialRadialDistributionFunction().compute_prdf(self.diamond)
        self.assertEqual(len(prdf.values()), 1)
        self.assertAlmostEqual(prdf[('C', 'C')][int(round(1.4 / 0.1))], 0)
        self.assertAlmostEqual(prdf[('C', 'C')][int(round(1.5 / 0.1))], 1.32445167622)
        self.assertAlmostEqual(max(distances), 19.9)
        self.assertAlmostEqual(prdf[('C', 'C')][int(round(19.9 / 0.1))], 0.07197902)

        # Test a few peaks in CsCl, make sure it gets all types correctly
        distances, prdf = PartialRadialDistributionFunction(cutoff=10).compute_prdf(self.cscl)
        self.assertEqual(len(prdf.values()), 4)
        self.assertAlmostEqual(max(distances), 9.9)
        self.assertAlmostEqual(prdf[('Cs', 'Cl')][int(round(3.6 / 0.1))], 0.477823197)
        self.assertAlmostEqual(prdf[('Cl', 'Cs')][int(round(3.6 / 0.1))], 0.477823197)
        self.assertAlmostEqual(prdf[('Cs', 'Cs')][int(round(3.6 / 0.1))], 0)

        # Do Ni3Al, make sure it captures the antisymmetry of Ni/Al sites
        distances, prdf = PartialRadialDistributionFunction(cutoff=10, bin_size=0.5)\
            .compute_prdf(self.ni3al)
        self.assertEqual(len(prdf.values()), 4)
        self.assertAlmostEqual(prdf[('Ni', 'Al')][int(round(2 / 0.5))], 0.125236677)
        self.assertAlmostEqual(prdf[('Al', 'Ni')][int(round(2 / 0.5))], 0.37571003)
        self.assertAlmostEqual(prdf[('Al', 'Al')][int(round(2 / 0.5))], 0)

        # Check the fit operation
        featurizer = PartialRadialDistributionFunction()
        featurizer.fit([self.diamond, self.cscl, self.ni3al])
        self.assertEqual({'Cs', 'Cl', 'C', 'Ni', 'Al'}, set(featurizer.elements_))

        featurizer.exclude_elems = ['Cs', 'Al']
        featurizer.fit([self.diamond, self.cscl, self.ni3al])
        self.assertEqual({'Cl', 'C', 'Ni'}, set(featurizer.elements_))

        featurizer.include_elems = ['H']
        featurizer.fit([self.diamond, self.cscl, self.ni3al])
        self.assertEqual({'H', 'Cl', 'C', 'Ni'}, set(featurizer.elements_))

        # Check the feature labels
        featurizer.exclude_elems = ()
        featurizer.include_elems = ()
        featurizer.elements_ = ['Al', 'Ni']
        labels = featurizer.feature_labels()
        n_bins = len(featurizer._make_bins()) - 1

        self.assertEqual(3 * n_bins, len(labels))
        self.assertIn('Al-Ni PRDF r=0.00-0.10', labels)

        # Check the featurize method
        featurizer.elements_ = ['C']
        features = featurizer.featurize(self.diamond)
        prdf = featurizer.compute_prdf(self.diamond)[1]
        self.assertArrayAlmostEqual(features, prdf[('C', 'C')])

        # Check the featurize_dataframe
        df = pd.DataFrame.from_dict({"structure": [self.diamond, self.cscl]})
        featurizer.fit(df["structure"])
        df = featurizer.featurize_dataframe(df, col_id="structure")
        self.assertEqual(df["Cs-Cl PRDF r=0.00-0.10"][0], 0.0)
        self.assertAlmostEqual(df["Cl-Cl PRDF r=19.70-19.80"][1], 0.049, 3)
        self.assertEqual(df["Cl-Cl PRDF r=19.90-20.00"][0], 0.0)

        # Make sure labels and features are in the same order
        featurizer.elements_ = ['Al', 'Ni']
        features = featurizer.featurize(self.ni3al)
        labels = featurizer.feature_labels()
        prdf = featurizer.compute_prdf(self.ni3al)[1]
        self.assertEqual((n_bins * 3,), features.shape)
        self.assertTrue(labels[0].startswith('Al-Al'))
        self.assertTrue(labels[n_bins].startswith('Al-Ni'))
        self.assertTrue(labels[2 * n_bins].startswith('Ni-Ni'))
        self.assertArrayAlmostEqual(features, np.hstack(
            [prdf[('Al', 'Al')], prdf[('Al', 'Ni')], prdf[('Ni', 'Ni')]]))

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
        # flat
        cm = CoulombMatrix(flatten=True)
        df = pd.DataFrame({"s": [self.diamond, self.nacl]})
        with self.assertRaises(NotFittedError):
            df = cm.featurize_dataframe(df, "s")
        df = cm.fit_featurize_dataframe(df, "s")
        labels = cm.feature_labels()
        self.assertListEqual(labels,
                             ["coulomb matrix eig 0", "coulomb matrix eig 1"])
        self.assertArrayAlmostEqual(df[labels].iloc[0],
                                    [49.169453, 24.546758],
                                    decimal=5)
        self.assertArrayAlmostEqual(df[labels].iloc[1],
                                    [153.774731, 452.894322],
                                    decimal=5)

        # matrix
        species = ["C", "C", "H", "H"]
        coords = [[0, 0, 0], [0, 0, 1.203], [0, 0, -1.06], [0, 0, 2.263]]
        acetylene = Molecule(species, coords)
        morig = CoulombMatrix(flatten=False).featurize(acetylene)
        mtarget = [[36.858, 15.835391290, 2.995098235, 1.402827813], \
                   [15.835391290, 36.858, 1.4028278132103624, 2.9950982], \
                   [2.9368896127, 1.402827813, 0.5, 0.159279959], \
                   [1.4028278132, 2.995098235, 0.159279959, 0.5]]
        self.assertAlmostEqual(
            int(np.linalg.norm(morig - np.array(mtarget))), 0)
        m = CoulombMatrix(diag_elems=False,
                          flatten=False).featurize(acetylene)[0]
        self.assertAlmostEqual(m[0][0], 0.0)
        self.assertAlmostEqual(m[1][1], 0.0)
        self.assertAlmostEqual(m[2][2], 0.0)
        self.assertAlmostEqual(m[3][3], 0.0)

    def test_sine_coulomb_matrix(self):
        # flat
        scm = SineCoulombMatrix(flatten=True)
        df = pd.DataFrame({"s": [self.sc, self.ni3al]})
        with self.assertRaises(NotFittedError):
            df = scm.featurize_dataframe(df, "s")
        df = scm.fit_featurize_dataframe(df, "s")
        labels = scm.feature_labels()
        self.assertEqual(labels[0], "sine coulomb matrix eig 0")
        self.assertArrayAlmostEqual(
            df[labels].iloc[0],
            [235.740418, 0.0, 0.0, 0.0],
            decimal=5)
        self.assertArrayAlmostEqual(
            df[labels].iloc[1],
            [232.578562, 1656.288171, 1403.106576, 1403.106576],
            decimal=5)

        # matrix
        scm = SineCoulombMatrix(flatten=False)
        sin_mat = scm.featurize(self.diamond)
        mtarget = [[36.8581, 6.147068], [6.147068, 36.8581]]
        self.assertAlmostEqual(
            np.linalg.norm(sin_mat - np.array(mtarget)), 0.0, places=4)
        scm = SineCoulombMatrix(diag_elems=False, flatten=False)
        sin_mat = scm.featurize(self.diamond)[0]
        self.assertEqual(sin_mat[0][0], 0)
        self.assertEqual(sin_mat[1][1], 0)

    def test_orbital_field_matrix(self):
        ofm_maker = OrbitalFieldMatrix(flatten=False)
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

        ofm_maker = OrbitalFieldMatrix(True, flatten=False)
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

        ofm_flat = OrbitalFieldMatrix(period_tag=False, flatten=True)
        self.assertEqual(len(ofm_flat.feature_labels()), 1024)

        ofm_flat = OrbitalFieldMatrix(period_tag=True, flatten=True)
        self.assertEqual(len(ofm_flat.feature_labels()), 1521)
        ofm_vector = ofm_flat.featurize(self.diamond)
        for ix in [40, 42, 72, 118, 120, 150, 1288, 1320]:
            self.assertAlmostEqual(ofm_vector[ix], 1.4789015345821415)

    def test_min_relative_distances(self):
        self.assertAlmostEqual(MinimumRelativeDistances().featurize(
                self.diamond_no_oxi)[0][0], 1.1052576)
        self.assertAlmostEqual(MinimumRelativeDistances().featurize(
                self.nacl)[0][0], 0.8891443)
        self.assertAlmostEqual(MinimumRelativeDistances().featurize(
                self.cscl)[0][0], 0.9877540)

    def test_sitestatsfingerprint(self):
        # Test matrix.
        op_struct_fp = SiteStatsFingerprint.from_preset("OPSiteFingerprint",
                                                        stats=None)
        opvals = op_struct_fp.featurize(self.diamond)
        oplabels = op_struct_fp.feature_labels()
        self.assertAlmostEqual(opvals[10][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[10][1], 0.9995, places=7)
        opvals = op_struct_fp.featurize(self.nacl)
        self.assertAlmostEqual(opvals[18][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[18][1], 0.9995, places=7)
        opvals = op_struct_fp.featurize(self.cscl)
        self.assertAlmostEqual(opvals[22][0], 0.9995, places=7)
        self.assertAlmostEqual(opvals[22][1], 0.9995, places=7)

        # Test stats.
        op_struct_fp = SiteStatsFingerprint.from_preset("OPSiteFingerprint")
        opvals = op_struct_fp.featurize(self.diamond)
        print(opvals, '**')
        self.assertAlmostEqual(opvals[0], 0.0005, places=7)
        self.assertAlmostEqual(opvals[1], 0, places=7)
        self.assertAlmostEqual(opvals[2], 0.0005, places=7)
        self.assertAlmostEqual(opvals[3], 0.0, places=7)
        self.assertAlmostEqual(opvals[4], 0.0005, places=7)
        self.assertAlmostEqual(opvals[18], 0.0805, places=7)
        self.assertAlmostEqual(opvals[20], 0.9995, places=7)
        self.assertAlmostEqual(opvals[21], 0, places=7)
        self.assertAlmostEqual(opvals[22], 0.0075, places=7)
        self.assertAlmostEqual(opvals[24], 0.2355, places=7)
        self.assertAlmostEqual(opvals[-1], 0.0, places=7)

        # Test coordination number
        cn_fp = SiteStatsFingerprint.from_preset("JmolNN", stats=("mean",))
        cn_vals = cn_fp.featurize(self.diamond)
        self.assertEqual(cn_vals[0], 4.0)

        # Test the covariance
        prop_fp = SiteStatsFingerprint(SiteElementalProperty(properties=["Number", "AtomicWeight"]),
                                       stats=["mean"], covariance=True)

        # Test the feature labels
        labels = prop_fp.feature_labels()
        self.assertEqual(3, len(labels))

        #  Test a structure with all the same type (cov should be zero)
        features = prop_fp.featurize(self.diamond)
        self.assertArrayAlmostEqual(features, [6, 12.0107, 0])

        #  Test a structure with only one atom (cov should be zero too)
        features = prop_fp.featurize(self.sc)
        self.assertArrayAlmostEqual([13, 26.9815386, 0], features)

        #  Test a structure with nonzero covariance
        features = prop_fp.featurize(self.nacl)
        self.assertArrayAlmostEqual([14, 29.22138464, 37.38969216], features)

    def test_ewald(self):
        # Add oxidation states to all of the structures
        for s in [self.nacl, self.cscl, self.diamond]:
            s.add_oxidation_state_by_guess()

        # Test basic
        ewald = EwaldEnergy(accuracy=2)
        self.assertArrayAlmostEqual(ewald.featurize(self.diamond), [0])
        self.assertAlmostEqual(ewald.featurize(self.nacl)[0], -8.84173626, 2)
        self.assertLess(ewald.featurize(self.nacl),
                        ewald.featurize(self.cscl))  # Atoms are closer in NaCl

        # Perform Ewald summation by "hand",
        #  Using the result from GULP
        self.assertArrayAlmostEqual([-8.84173626], ewald.featurize(self.nacl), 2)

    def test_bondfractions(self):

        # Test individual structures with featurize
        bf_md = BondFractions.from_preset("MinimumDistanceNN")
        bf_md.no_oxi = True
        bf_md.fit([self.diamond_no_oxi])
        self.assertArrayEqual(bf_md.featurize(self.diamond), [1.0])
        self.assertArrayEqual(bf_md.featurize(self.diamond_no_oxi), [1.0])

        bf_voronoi = BondFractions.from_preset("VoronoiNN")
        bf_voronoi.bbv = float("nan")
        bf_voronoi.fit([self.nacl])
        bond_fracs = bf_voronoi.featurize(self.nacl)
        bond_names = bf_voronoi.feature_labels()
        ref = {'Na+ - Na+ bond frac.': 0.25, 'Cl- - Na+ bond frac.': 0.5,
               'Cl- - Cl- bond frac.': 0.25}
        self.assertDictEqual(dict(zip(bond_names, bond_fracs)), ref)

        # Test to make sure dataframe behavior is as intended
        s_list = [self.diamond_no_oxi, self.ni3al]
        df = pd.DataFrame.from_dict({'s': s_list})
        bf_voronoi.fit(df['s'])
        df = bf_voronoi.featurize_dataframe(df, 's')

        # Ensure all data is properly labelled and organized
        self.assertArrayEqual(df['C - C bond frac.'].to_numpy(), [1.0, np.nan])
        self.assertArrayEqual(df['Al - Ni bond frac.'].to_numpy(), [np.nan, 0.5])
        self.assertArrayEqual(df['Al - Al bond frac.'].to_numpy(), [np.nan, 0.0])
        self.assertArrayEqual(df['Ni - Ni bond frac.'].to_numpy(), [np.nan, 0.5])

        # Test to make sure bad_bond_values (bbv) are still changed correctly
        # and check inplace behavior of featurize dataframe.
        bf_voronoi.bbv = 0.0
        df = pd.DataFrame.from_dict({'s': s_list})
        df = bf_voronoi.featurize_dataframe(df, 's')
        self.assertArrayEqual(df['C - C bond frac.'].to_numpy(), [1.0, 0.0])
        self.assertArrayEqual(df['Al - Ni bond frac.'].to_numpy(), [0.0, 0.5])
        self.assertArrayEqual(df['Al - Al bond frac.'].to_numpy(), [0.0, 0.0])
        self.assertArrayEqual(df['Ni - Ni bond frac.'].to_numpy(), [0.0, 0.5])

    def test_bob(self):

        # Test a single fit and featurization
        scm = SineCoulombMatrix(flatten=False)
        bob = BagofBonds(coulomb_matrix=scm, token=' - ')
        bob.fit([self.ni3al])
        truth1 = [235.74041833262768, 1486.4464890775491, 1486.4464890775491,
                  1486.4464890775491, 38.69353092306119, 38.69353092306119,
                  38.69353092306119, 38.69353092306119, 38.69353092306119,
                  38.69353092306119, 83.33991275736257, 83.33991275736257,
                  83.33991275736257, 83.33991275736257, 83.33991275736257,
                  83.33991275736257]
        truth1_labels = ['Al site #0', 'Ni site #0', 'Ni site #1', 'Ni site #2',
                         'Al - Ni bond #0', 'Al - Ni bond #1',
                         'Al - Ni bond #2', 'Al - Ni bond #3',
                         'Al - Ni bond #4', 'Al - Ni bond #5',
                         'Ni - Ni bond #0', 'Ni - Ni bond #1',
                         'Ni - Ni bond #2', 'Ni - Ni bond #3',
                         'Ni - Ni bond #4', 'Ni - Ni bond #5']
        self.assertArrayAlmostEqual(bob.featurize(self.ni3al), truth1)
        self.assertEqual(bob.feature_labels(), truth1_labels)

        # Test padding from fitting and dataframe featurization
        bob.coulomb_matrix = CoulombMatrix(flatten=False)
        bob.fit([self.ni3al, self.cscl, self.diamond_no_oxi])
        df = pd.DataFrame({'structures': [self.cscl]})
        df = bob.featurize_dataframe(df, 'structures')
        self.assertEqual(len(df.columns.values), 25)
        self.assertAlmostEqual(df['Cs site #0'][0], 7513.468312122532)
        self.assertAlmostEqual(df['Al site #0'][0], 0.0)
        self.assertAlmostEqual(df['Cs - Cl bond #1'][0], 135.74726437398044, 3)
        self.assertAlmostEqual(df['Al - Ni bond #0'][0], 0.0)

        # Test error handling for bad fits or null fits
        bob = BagofBonds(CoulombMatrix(flatten=False))
        self.assertRaises(NotFittedError, bob.featurize, self.nacl)
        bob.fit([self.ni3al, self.diamond])
        self.assertRaises(ValueError, bob.featurize, self.nacl)\

    def test_ward_prb_2017_lpd(self):
        """Test the local property difference attributes from Ward 2017"""
        f = SiteStatsFingerprint.from_preset(
            "LocalPropertyDifference_ward-prb-2017"
        )

        # Test diamond
        features = f.featurize(self.diamond)
        self.assertArrayAlmostEqual(features, [0] * (22 * 5))
        features = f.featurize(self.diamond_no_oxi)
        self.assertArrayAlmostEqual(features, [0] * (22 * 5))

        # Test CsCl
        big_face_area = np.sqrt(3) * 3 / 2 * (2 / 4 / 4)
        small_face_area = 0.125
        big_face_diff = 55 - 17
        features = f.featurize(self.cscl)
        labels = f.feature_labels()
        my_label = 'mean local difference in Number'
        self.assertAlmostEqual((8 * big_face_area * big_face_diff) /
                               (8 * big_face_area + 6 * small_face_area),
                               features[labels.index(my_label)], places=3)
        my_label = 'range local difference in Electronegativity'
        self.assertAlmostEqual(0, features[labels.index(my_label)], places=3)

    def test_ward_prb_2017_efftcn(self):
        """Test the effective coordination number attributes of Ward 2017"""
        f = SiteStatsFingerprint.from_preset(
            "CoordinationNumber_ward-prb-2017"
        )

        # Test Ni3Al
        features = f.featurize(self.ni3al)
        labels = f.feature_labels()
        my_label = 'mean CN_VoronoiNN'
        self.assertAlmostEqual(12, features[labels.index(my_label)])
        self.assertArrayAlmostEqual([12, 12, 0, 12, 0], features)

    def test_ward_prb_2017_strhet(self):
        f = StructuralHeterogeneity()

        # Test Ni3Al, which is uniform
        features = f.featurize(self.ni3al)
        self.assertArrayAlmostEqual([0, 1, 1, 0, 0, 0, 0, 0, 0], features)

        # Do CsCl, which has variation in the neighbors
        big_face_area = np.sqrt(3) * 3 / 2 * (2 / 4 / 4)
        small_face_area = 0.125
        average_dist = (8 * np.sqrt(
            3) / 2 * big_face_area + 6 * small_face_area) \
                       / (8 * big_face_area + 6 * small_face_area)
        rel_var = (8 * abs(np.sqrt(3) / 2 - average_dist) * big_face_area +
                   6 * abs(1 - average_dist) * small_face_area) \
                  / (8 * big_face_area + 6 * small_face_area) / average_dist
        cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"], [[0.5, 0.5, 0.5], [0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False, site_properties=None)
        features = f.featurize(cscl)
        self.assertArrayAlmostEqual(
            [0, 1, 1, rel_var, rel_var, 0, rel_var, 0, 0],
            features)

    def test_packing_efficiency(self):
        f = MaximumPackingEfficiency()

        # Test L1_2
        self.assertArrayAlmostEqual([np.pi / 3 / np.sqrt(2)],
                                    f.featurize(self.ni3al))

        # Test B1
        self.assertArrayAlmostEqual([np.pi / 6], f.featurize(self.nacl),
                                    decimal=3)

    def test_ordering_param(self):
        f = ChemicalOrdering()

        # Check that elemental structures return zero
        features = f.featurize(self.diamond)
        self.assertArrayAlmostEqual([0, 0, 0], features)

        # Check result for CsCl
        #   These were calculated by hand by Logan Ward
        features = f.featurize(self.cscl)
        self.assertAlmostEqual(0.551982, features[0], places=5)
        self.assertAlmostEqual(0.241225, features[1], places=5)

        # Check for L1_2
        features = f.featurize(self.ni3al)
        self.assertAlmostEqual(1./3., features[0], places=5)
        self.assertAlmostEqual(0.0303, features[1], places=5)

    def test_composition_features(self):
        comp = ElementProperty.from_preset("magpie")
        f = StructureComposition(featurizer=comp)

        # Test the fitting (should not crash)
        f.fit([self.nacl, self.diamond])

        # Test the features
        features = f.featurize(self.nacl)
        self.assertArrayAlmostEqual(comp.featurize(self.nacl.composition),
                                    features)

        # Test the citations/implementors
        self.assertEqual(comp.citations(), f.citations())
        self.assertEqual(comp.implementors(), f.implementors())

    def test_xrd_powderPattern(self):

        # default settings test
        xpp = XRDPowderPattern()
        pattern = xpp.featurize(self.diamond)
        self.assertAlmostEqual(pattern[44], 0.19378, places=2)
        self.assertEqual(len(pattern), 128)

        # reduced range
        xpp = XRDPowderPattern(two_theta_range=(0, 90))
        pattern = xpp.featurize(self.diamond)
        self.assertAlmostEqual(pattern[44], 0.4083, places=2)
        self.assertEqual(len(pattern), 91)
        self.assertEqual(len(xpp.feature_labels()), 91)

    @unittest.skipIf(not (torch and cgcnn),
                     "pytorch or cgcnn not installed.")
    def test_cgcnn_featurizer(self):
        # Test regular classification.
        cla_props, cla_atom_features, cla_structs = self._get_cgcnn_data()
        atom_fea_len = 64
        cgcnn_featurizer = \
            CGCNNFeaturizer(atom_init_fea=cla_atom_features,
                            train_size=5, val_size=2, test_size=3,
                            atom_fea_len=atom_fea_len)

        cgcnn_featurizer.fit(X=cla_structs, y=cla_props)
        self.assertEqual(len(cgcnn_featurizer.feature_labels()), atom_fea_len)
        state_dict = cgcnn_featurizer.model.state_dict()
        self.assertEqual(state_dict['embedding.weight'].size(),
                         torch.Size([64, 92]))
        self.assertEqual(state_dict['embedding.bias'].size(),
                         torch.Size([64]))
        self.assertEqual(state_dict['convs.0.fc_full.weight'].size(),
                         torch.Size([128, 169]))
        self.assertEqual(state_dict['convs.1.bn1.weight'].size(),
                         torch.Size([128]))
        self.assertEqual(state_dict['convs.2.bn2.bias'].size(),
                         torch.Size([64]))
        self.assertEqual(state_dict['conv_to_fc.weight'].size(),
                         torch.Size([128, 64]))
        self.assertEqual(state_dict['fc_out.weight'].size(),
                         torch.Size([2, 128]))

        for struct in cla_structs:
            result = cgcnn_featurizer.featurize(struct)
            self.assertEqual(len(result), atom_fea_len)

        # Test regular regression and default atom_init_fea and featurize_many.
        reg_props, reg_atom_features, reg_structs = \
            self._get_cgcnn_data("regression")
        cgcnn_featurizer = \
            CGCNNFeaturizer(task="regression", atom_fea_len=atom_fea_len,
                            train_size=6, val_size=2, test_size=2)

        cgcnn_featurizer.fit(X=reg_structs, y=reg_props)
        cgcnn_featurizer.set_n_jobs(1)

        result = cgcnn_featurizer.featurize_many(entries=reg_structs)
        self.assertEqual(np.array(result).shape,
                         (len(reg_structs), atom_fea_len))

        # Test classification from pre-trained model.
        cgcnn_featurizer = \
            CGCNNFeaturizer(h_fea_len=32, n_conv=4,
                            pretrained_name='semi-metal-classification',
                            atom_init_fea=cla_atom_features, train_size=5,
                            val_size=2, test_size=3, atom_fea_len=atom_fea_len)
        cgcnn_featurizer.fit(X=cla_structs, y=cla_props)
        self.assertEqual(len(cgcnn_featurizer.feature_labels()), atom_fea_len)

        validate_features = [2.1295, 2.1288, 1.8504, 1.9175, 2.1094,
                             1.7770, 2.0471, 1.7426, 1.7288, 1.7770]
        for struct, validate_feature in zip(cla_structs, validate_features):
            result = cgcnn_featurizer.featurize(struct)
            self.assertEqual(len(result), atom_fea_len)
            self.assertAlmostEqual(result[0], validate_feature, 4)

        # Test regression from pre-trained model.
        cgcnn_featurizer = \
            CGCNNFeaturizer(task="regression", h_fea_len=32, n_conv=4,
                            pretrained_name='formation-energy-per-atom',
                            atom_init_fea=reg_atom_features,
                            train_size=5, val_size=2, test_size=3,
                            atom_fea_len=atom_fea_len)
        cgcnn_featurizer.fit(X=reg_structs, y=reg_props)
        self.assertEqual(len(cgcnn_featurizer.feature_labels()), atom_fea_len)

        validate_features = [1.6871, 1.5679, 1.5316, 1.6419, 1.6031,
                             1.4333, 1.5709, 1.5070, 1.5038, 1.4333]

        for struct, validate_feature in zip(reg_structs, validate_features):
            result = cgcnn_featurizer.featurize(struct)
            self.assertEqual(len(result), atom_fea_len)
            self.assertAlmostEqual(result[-1], validate_feature, 4)

        # Test warm start regression.
        warm_start_file = os.path.join(test_dir,
                                       'cgcnn_test_regression_model.pth.tar')
        warm_start_model = torch.load(warm_start_file)
        self.assertEqual(warm_start_model['epoch'], 31)
        self.assertEqual(warm_start_model['best_epoch'], 9)
        self.assertAlmostEqual(warm_start_model['best_mae_error'].numpy(),
                               2.3700, 4)

        cgcnn_featurizer = \
            CGCNNFeaturizer(task="regression", warm_start_file=warm_start_file,
                            epochs=100, atom_fea_len=atom_fea_len,
                            atom_init_fea=reg_atom_features,
                            train_size=6, val_size=2, test_size=2)
        cgcnn_featurizer.fit(X=reg_structs, y=reg_props)

        # If use CGCNN featurize_many(), you should change the multiprocessing
        # start_method to 'spawn', because Gloo (that uses Infiniband) and
        # NCCL2 are not fork safe, pytorch don't support them or just
        # set n_jobs = 1 to avoid multiprocessing as follows.
        set_start_method('spawn', force=True)
        result = cgcnn_featurizer.featurize_many(entries=reg_structs)
        self.assertEqual(np.array(result).shape,
                         (len(reg_structs), atom_fea_len))

        # Test featurize_dataframe.
        df = pd.DataFrame.from_dict({"structure": cla_structs})
        cgcnn_featurizer = \
            CGCNNFeaturizer(atom_init_fea=cla_atom_features,
                            train_size=5, val_size=2, test_size=3,
                            atom_fea_len=atom_fea_len)
        cgcnn_featurizer.fit(X=df["structure"], y=cla_props)
        self.assertEqual(len(cgcnn_featurizer.feature_labels()), atom_fea_len)
        cgcnn_featurizer.set_n_jobs(1)
        result = cgcnn_featurizer.featurize_dataframe(df, "structure")
        self.assertTrue("CGCNN_feature_{}".format(atom_fea_len - 1)
                        in result.columns)
        self.assertEqual(np.array(result).shape,
                         (len(reg_structs), atom_fea_len + 1))

        # Test fit_featurize_dataframe.
        df = pd.DataFrame.from_dict({"structure": cla_structs})
        cgcnn_featurizer = \
            CGCNNFeaturizer(atom_init_fea=cla_atom_features,
                            train_size=5, val_size=2, test_size=3,
                            atom_fea_len=atom_fea_len)
        result = cgcnn_featurizer.fit_featurize_dataframe(
            df, "structure", fit_args=[cla_props])
        self.assertEqual(len(cgcnn_featurizer.feature_labels()), atom_fea_len)
        self.assertTrue("CGCNN_feature_{}".format(atom_fea_len - 1)
                        in result.columns)
        self.assertEqual(np.array(result).shape,
                         (len(reg_structs), atom_fea_len + 1))

    @staticmethod
    def _get_cgcnn_data(task="classification"):
        """
        Get cgcnn sample data.
        Args:
            task (str): Classification or regression,
                        decided which sample data to return.

        Returns:
            id_prop_data (list): List of property data.
            elem_embedding (list): List of element features.
            struct_list (list): List of structure object.
        """
        if task == "classification":
            cgcnn_data_path = os.path.join(os.path.dirname(cgcnn.__file__),
                                           "..",
                                           "data", "sample-classification")
        else:
            cgcnn_data_path = os.path.join(os.path.dirname(cgcnn.__file__),
                                           "..",
                                           "data", "sample-regression")

        struct_list = list()
        cif_list = list()
        with open(os.path.join(cgcnn_data_path, "id_prop.csv")) as f:
            reader = csv.reader(f)
            id_prop_data = [row[1] for row in reader]
        with open(os.path.join(cgcnn_data_path, "atom_init.json")) as f:
            elem_embedding = json.load(f)

        for file in os.listdir(cgcnn_data_path):
            if file.endswith('.cif'):
                cif_list.append(int(file[:-4]))
                cif_list = sorted(cif_list)
        for cif_name in cif_list:
            crystal = Structure.from_file(os.path.join(cgcnn_data_path,
                                                       '{}.cif'.format(
                                                           cif_name)))
            struct_list.append(crystal)
        return id_prop_data, elem_embedding, struct_list

    def test_jarvisCFID(self):

        # default (all descriptors)
        jcf = JarvisCFID()
        self.assertEqual(len(jcf.feature_labels()), 1557)
        fvec = jcf.featurize(self.cscl)
        self.assertEqual(len(fvec), 1557)
        self.assertAlmostEqual(fvec[-1], 0.0, places=3)
        self.assertAlmostEqual(fvec[1], 591.5814, places=3)
        self.assertAlmostEqual(fvec[0], 1346.755, places=3)

        # a combination of descriptors
        jcf = JarvisCFID(use_chem=False, use_chg=False, use_cell=False)
        self.assertEqual(len(jcf.feature_labels()), 737)
        fvec = jcf.featurize(self.diamond)
        self.assertAlmostEqual(fvec[-1], 24, places=3)
        self.assertAlmostEqual(fvec[0], 0, places=3)

    def test_GlobalInstabilityIndex(self):
        # Test diamond and ni3al fail precheck
        gii = GlobalInstabilityIndex(r_cut=4.0, disordered_pymatgen=False)
        self.assertFalse(gii.precheck(self.diamond))
        self.assertFalse(gii.precheck(self.ni3al))
        # Test they raise errors when featurizing
        with self.assertRaises(AttributeError):
            gii.featurize(self.ni3al)
        with self.assertRaises(ValueError):
            gii.featurize(self.diamond)

        # Ordinary case of nacl
        self.assertTrue(gii.precheck(self.nacl))
        self.assertAlmostEqual(gii.featurize(self.nacl)[0], 0.08491655709)

        # Test bond valence sums are accurate for NaCl.
        # Values are closer to 0.915 than 1.0 due to structure specified here.
        # Using CollCode181148 from the ICSD, I see bond valence sums of 0.979
        site1, site2 = (self.nacl[0], self.nacl[1])
        neighs1 = self.nacl.get_neighbors(site1, 4)
        neighs2 = self.nacl.get_neighbors(site2, 4)
        site_val1 = site1.species.elements[0].oxi_state
        site_el1 = str(site1.species.element_composition.elements[0])
        site_val2 = site2.species.elements[0].oxi_state
        site_el2 = str(site2.species.element_composition.elements[0])
        self.assertAlmostEqual(gii.calc_bv_sum(site_val1,
                                               site_el1,
                                               neighs1), 0.9150834429025214)
        self.assertAlmostEqual(gii.calc_bv_sum(site_val2,
                                               site_el2,
                                               neighs2), -0.915083442902522)

        # Behavior when disorder is present
        gii_pymat = GlobalInstabilityIndex(r_cut=4.0, disordered_pymatgen=True)
        nacl_disordered = copy.deepcopy(self.nacl)
        nacl_disordered.replace_species({"Cl1-": "Cl0.5Br0.5"})
        nacl_disordered.add_oxidation_state_by_element({'Na': 1, 'Cl': -1, 'Br': -1})
        self.assertTrue(gii.precheck(nacl_disordered))
        with self.assertRaises(ValueError):
            gii.featurize(nacl_disordered)
        self.assertAlmostEqual(gii_pymat.featurize(nacl_disordered)[0], 0.39766464)

    def test_structural_complexity(self):
        s = Structure.from_file(
            os.path.join(test_dir, "Dy2HfS5_mp-1198001_computed.cif")
        )

        featurizer = StructuralComplexity()
        ig, igbits = featurizer.featurize(s)

        self.assertAlmostEqual(2.5, ig, places=3)
        self.assertAlmostEqual(80, igbits, places=3)

        s = Structure.from_file(
            os.path.join(test_dir, "Cs2CeN5O17_mp-1198000_computed.cif")
        )

        featurizer = StructuralComplexity()
        ig, igbits = featurizer.featurize(s)

        self.assertAlmostEqual(3.764, ig, places=3)
