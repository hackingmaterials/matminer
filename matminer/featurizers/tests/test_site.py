from __future__ import unicode_literals, division, print_function

import numpy as np
import pandas as pd
from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.local_env import VoronoiNN, JmolNN, CrystalNN

from matminer.featurizers.site import AGNIFingerprints, \
    OPSiteFingerprint, CrystalNNFingerprint, \
    EwaldSiteEnergy, \
    VoronoiFingerprint, IntersticeDistribution, ChemEnvSiteFingerprint, \
    CoordinationNumber, ChemicalSRO, GaussianSymmFunc, \
    GeneralizedRadialDistributionFunction, AngularFourierSeries, \
    LocalPropertyDifference, SOAP, BondOrientationalParameter, \
    SiteElementalProperty, AverageBondLength, AverageBondAngle
from matminer.featurizers.deprecated import CrystalSiteFingerprint
from matminer.featurizers.utils.grdf import Gaussian


class FingerprintTests(PymatgenTest):
    def setUp(self):
        self.sc = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ],
            [[0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)
        self.cscl = Structure(
            Lattice([[4.209, 0, 0], [0, 4.209, 0], [0, 0, 4.209]]),
            ["Cl1-", "Cs1+"], [[0.45, 0.5, 0.5], [0, 0, 0]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)
        self.b1 = Structure(
            Lattice([[0,1,1],[1,0,1],[1,1,0]]),
            ["H", "He"], [[0,0,0],[0.5,0.5,0.5]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False)
        self.diamond = Structure(
            Lattice([[2.189, 0, 1.264], [0.73, 2.064, 1.264],
                     [0, 0, 2.528]]), ["C0+", "C0+"], [[2.554, 1.806, 4.423],
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
        self.ni3al = Structure(
            Lattice([[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]),
            ["Al", ] + ["Ni"] * 3,
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            validate_proximity=False, to_unit_cell=False,
            coords_are_cartesian=False, site_properties=None)

    def test_simple_cubic(self):
        """Test with an easy structure"""

        # Make sure direction-dependent fingerprints are zero
        agni = AGNIFingerprints(directions=['x', 'y', 'z'])

        features = agni.featurize(self.sc, 0)
        self.assertEqual(8 * 3, len(features))
        self.assertEqual(8 * 3, len(set(agni.feature_labels())))
        self.assertArrayAlmostEqual([0, ] * 24, features)

        # Compute the "atomic fingerprints"
        agni.directions = [None]
        agni.cutoff = 3.75  # To only get 6 neighbors to deal with

        features = agni.featurize(self.sc, 0)
        self.assertEqual(8, len(features))
        self.assertEqual(8, len(set(agni.feature_labels())))

        self.assertEqual(0.8, agni.etas[0])
        self.assertAlmostEqual(6 * np.exp(-(3.52 / 0.8) ** 2) * 0.5 * (np.cos(np.pi * 3.52 / 3.75) + 1), features[0])
        self.assertAlmostEqual(6 * np.exp(-(3.52 / 16) ** 2) * 0.5 * (np.cos(np.pi * 3.52 / 3.75) + 1), features[-1])

        # Test that passing etas to constructor works
        new_etas = np.logspace(-4, 2, 6)
        agni = AGNIFingerprints(directions=['x', 'y', 'z'], etas=new_etas)
        self.assertArrayAlmostEqual(new_etas, agni.etas)

    def test_off_center_cscl(self):
        agni = AGNIFingerprints(directions=[None, 'x', 'y', 'z'], cutoff=4)

        # Compute the features on both sites
        site1 = agni.featurize(self.cscl, 0)
        site2 = agni.featurize(self.cscl, 1)

        # The atomic attributes should be equal
        self.assertArrayAlmostEqual(site1[:8], site2[:8])

        # The direction-dependent ones should be equal and opposite in sign
        self.assertArrayAlmostEqual(-1 * site1[8:], site2[8:])

        # Make sure the site-ones are as expected.
        right_dist = 4.209 * np.sqrt(0.45 ** 2 + 2 * 0.5 ** 2)
        right_xdist = 4.209 * 0.45
        left_dist = 4.209 * np.sqrt(0.55 ** 2 + 2 * 0.5 ** 2)
        left_xdist = 4.209 * 0.55
        self.assertAlmostEqual(4 * (
            right_xdist / right_dist * np.exp(-(right_dist / 0.8) ** 2) * 0.5 * (np.cos(np.pi * right_dist / 4) + 1) -
            left_xdist / left_dist * np.exp(-(left_dist / 0.8) ** 2) * 0.5 * (np.cos(np.pi * left_dist / 4) + 1)),
                                site1[8])

    def test_dataframe(self):
        data = pd.DataFrame({'strc': [self.cscl, self.cscl, self.sc], 'site': [0, 1, 0]})

        agni = AGNIFingerprints()
        agni.featurize_dataframe(data, ['strc', 'site'])

    def test_op_site_fingerprint(self):
        opsf = OPSiteFingerprint()
        l = opsf.feature_labels()
        t = ['sgl_bd CN_1', 'L-shaped CN_2', 'water-like CN_2', \
             'bent 120 degrees CN_2', 'bent 150 degrees CN_2', \
             'linear CN_2', 'trigonal planar CN_3', \
             'trigonal non-coplanar CN_3', 'T-shaped CN_3', \
             'square co-planar CN_4', 'tetrahedral CN_4', \
             'rectangular see-saw-like CN_4', 'see-saw-like CN_4', \
             'trigonal pyramidal CN_4', 'pentagonal planar CN_5', \
             'square pyramidal CN_5', 'trigonal bipyramidal CN_5', \
             'hexagonal planar CN_6', 'octahedral CN_6', \
             'pentagonal pyramidal CN_6', 'hexagonal pyramidal CN_7', \
             'pentagonal bipyramidal CN_7', 'body-centered cubic CN_8', \
             'hexagonal bipyramidal CN_8', 'q2 CN_9', 'q4 CN_9', 'q6 CN_9', \
             'q2 CN_10', 'q4 CN_10', 'q6 CN_10', \
             'q2 CN_11', 'q4 CN_11', 'q6 CN_11', \
             'cuboctahedral CN_12', 'q2 CN_12', 'q4 CN_12', 'q6 CN_12']
        for i in range(len(l)):
            self.assertEqual(l[i], t[i])
        ops = opsf.featurize(self.sc, 0)
        self.assertEqual(len(ops), 37)
        self.assertAlmostEqual(ops[opsf.feature_labels().index(
            'octahedral CN_6')], 0.9995, places=7)
        ops = opsf.featurize(self.cscl, 0)
        self.assertAlmostEqual(ops[opsf.feature_labels().index(
            'body-centered cubic CN_8')], 0.8955, places=7)
        opsf = OPSiteFingerprint(dist_exp=0)
        ops = opsf.featurize(self.cscl, 0)
        self.assertAlmostEqual(ops[opsf.feature_labels().index(
            'body-centered cubic CN_8')], 0.9555, places=7)

        # The following test aims at ensuring the copying of the OP dictionaries work.
        opsfp = OPSiteFingerprint()
        cnnfp = CrystalNNFingerprint.from_preset('ops')
        self.assertEqual(len([1 for l in opsfp.feature_labels() if l.split()[0] == 'wt']), 0)

    def test_crystal_site_fingerprint(self):
        with self.assertWarns(DeprecationWarning):
            csf = CrystalSiteFingerprint.from_preset('ops')
            l = csf.feature_labels()
            t = ['wt CN_1', 'wt CN_2', 'L-shaped CN_2', 'water-like CN_2',
                 'bent 120 degrees CN_2', 'bent 150 degrees CN_2', 'linear CN_2',
                 'wt CN_3', 'trigonal planar CN_3', 'trigonal non-coplanar CN_3',
                 'T-shaped CN_3', 'wt CN_4', 'square co-planar CN_4',
                 'tetrahedral CN_4', 'rectangular see-saw-like CN_4',
                 'see-saw-like CN_4', 'trigonal pyramidal CN_4', 'wt CN_5',
                 'pentagonal planar CN_5', 'square pyramidal CN_5',
                 'trigonal bipyramidal CN_5', 'wt CN_6', 'hexagonal planar CN_6',
                 'octahedral CN_6', 'pentagonal pyramidal CN_6', 'wt CN_7',
                 'hexagonal pyramidal CN_7', 'pentagonal bipyramidal CN_7',
                 'wt CN_8', 'body-centered cubic CN_8',
                 'hexagonal bipyramidal CN_8', 'wt CN_9', 'q2 CN_9', 'q4 CN_9',
                 'q6 CN_9', 'wt CN_10', 'q2 CN_10', 'q4 CN_10', 'q6 CN_10',
                 'wt CN_11', 'q2 CN_11', 'q4 CN_11', 'q6 CN_11', 'wt CN_12',
                 'cuboctahedral CN_12', 'q2 CN_12', 'q4 CN_12', 'q6 CN_12']
            for i in range(len(l)):
                self.assertEqual(l[i], t[i])
            ops = csf.featurize(self.sc, 0)
            self.assertEqual(len(ops), 48)
            self.assertAlmostEqual(ops[csf.feature_labels().index(
                'wt CN_6')], 1, places=4)
            self.assertAlmostEqual(ops[csf.feature_labels().index(
                'octahedral CN_6')], 1, places=4)
            ops = csf.featurize(self.cscl, 0)
            self.assertAlmostEqual(ops[csf.feature_labels().index(
                'wt CN_8')], 0.5575257, places=4)
            self.assertAlmostEqual(ops[csf.feature_labels().index(
                'body-centered cubic CN_8')], 0.5329344, places=4)

    def test_crystal_nn_fingerprint(self):
        cnnfp = CrystalNNFingerprint.from_preset(
                'ops', distance_cutoffs=None, x_diff_weight=None)
        l = cnnfp.feature_labels()
        t = ['wt CN_1', 'sgl_bd CN_1', 'wt CN_2', 'L-shaped CN_2',
             'water-like CN_2', 'bent 120 degrees CN_2',
             'bent 150 degrees CN_2', 'linear CN_2', 'wt CN_3',
             'trigonal planar CN_3', 'trigonal non-coplanar CN_3',
             'T-shaped CN_3', 'wt CN_4', 'square co-planar CN_4',
             'tetrahedral CN_4', 'rectangular see-saw-like CN_4',
             'see-saw-like CN_4', 'trigonal pyramidal CN_4', 'wt CN_5',
             'pentagonal planar CN_5', 'square pyramidal CN_5',
             'trigonal bipyramidal CN_5', 'wt CN_6', 'hexagonal planar CN_6',
             'octahedral CN_6', 'pentagonal pyramidal CN_6', 'wt CN_7',
             'hexagonal pyramidal CN_7', 'pentagonal bipyramidal CN_7',
             'wt CN_8', 'body-centered cubic CN_8',
             'hexagonal bipyramidal CN_8', 'wt CN_9', 'q2 CN_9', 'q4 CN_9',
             'q6 CN_9', 'wt CN_10', 'q2 CN_10', 'q4 CN_10', 'q6 CN_10',
             'wt CN_11', 'q2 CN_11', 'q4 CN_11', 'q6 CN_11', 'wt CN_12',
             'cuboctahedral CN_12', 'q2 CN_12', 'q4 CN_12', 'q6 CN_12',
             'wt CN_13', 'wt CN_14', 'wt CN_15', 'wt CN_16', 'wt CN_17',
             'wt CN_18', 'wt CN_19', 'wt CN_20', 'wt CN_21', 'wt CN_22',
             'wt CN_23', 'wt CN_24']
        for i in range(len(l)):
            self.assertEqual(l[i], t[i])
        ops = cnnfp.featurize(self.sc, 0)
        self.assertEqual(len(ops), 61)
        self.assertAlmostEqual(ops[cnnfp.feature_labels().index(
            'wt CN_6')], 1, places=7)
        self.assertAlmostEqual(ops[cnnfp.feature_labels().index(
            'octahedral CN_6')], 1, places=7)
        ops = cnnfp.featurize(self.cscl, 0)
        self.assertAlmostEqual(ops[cnnfp.feature_labels().index(
            'wt CN_8')], 0.498099, places=3)

        self.assertAlmostEqual(ops[cnnfp.feature_labels().index(
            'body-centered cubic CN_8')], 0.47611, places=3)

        op_types = {6: ["wt", "oct_max"], 8: ["wt", "bcc"]}
        cnnfp = CrystalNNFingerprint(
            op_types, distance_cutoffs=None, \
            x_diff_weight=None)
        labels = ['wt CN_6', 'oct_max CN_6', \
                  'wt CN_8', 'bcc CN_8']
        for l1, l2 in zip(cnnfp.feature_labels(), labels):
            self.assertEqual(l1, l2)
        feats = cnnfp.featurize(self.sc, 0)
        self.assertEqual(len(feats), 4)

        chem_info = {"mass": {"Al": 26.9, "Cs+": 132.9,"Cl-": 35.4}, \
            "Pauling scale": {"Al": 1.61, "Cs+": 0.79, "Cl-": 3.16}}
        cnnchemfp = CrystalNNFingerprint(
            op_types, chem_info=chem_info, distance_cutoffs=None, \
            x_diff_weight=None)
        labels = labels + ['mass local diff', \
            'Pauling scale local diff']
        for l1, l2 in zip(cnnchemfp.feature_labels(), labels):
            self.assertEqual(l1, l2)

        feats = cnnchemfp.featurize(self.sc, 0)
        self.assertEqual(len(feats), 6)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'wt CN_6')], 1, places=7)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'oct_max CN_6')], 1, places=7)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'mass local diff')], 0, places=7)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'Pauling scale local diff')], 0, places=7)

        feats = cnnchemfp.featurize(self.cscl, 0)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'bcc CN_8')], 0.4761107, places=3)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'mass local diff')], 97.5, places=3)
        self.assertAlmostEqual(feats[cnnchemfp.feature_labels().index(
            'Pauling scale local diff')], -2.37, places=3)

    def test_chemenv_site_fingerprint(self):
        cefp = ChemEnvSiteFingerprint.from_preset('multi_weights')
        l = cefp.feature_labels()
        cevals = cefp.featurize(self.sc, 0)
        self.assertEqual(len(cevals), 66)
        self.assertAlmostEqual(cevals[l.index('O:6')], 1, places=7)
        self.assertAlmostEqual(cevals[l.index('C:8')], 0, places=7)
        cevals = cefp.featurize(self.cscl, 0)
        self.assertAlmostEqual(cevals[l.index('C:8')],  0.9953721, places=7)
        self.assertAlmostEqual(cevals[l.index('O:6')], 0, places=7)
        cefp = ChemEnvSiteFingerprint.from_preset('simple')
        l = cefp.feature_labels()
        cevals = cefp.featurize(self.sc, 0)
        self.assertEqual(len(cevals), 66)
        self.assertAlmostEqual(cevals[l.index('O:6')], 1, places=7)
        self.assertAlmostEqual(cevals[l.index('C:8')], 0, places=7)
        cevals = cefp.featurize(self.cscl, 0)
        self.assertAlmostEqual(cevals[l.index('C:8')], 0.9953721, places=7)
        self.assertAlmostEqual(cevals[l.index('O:6')], 0, places=7)

    def test_voronoifingerprint(self):
        df_sc= pd.DataFrame({'struct': [self.sc], 'site': [0]})
        vorofp = VoronoiFingerprint(use_symm_weights=True)
        vorofps = vorofp.featurize_dataframe(df_sc, ['struct', 'site'])
        self.assertAlmostEqual(vorofps['Voro_index_3'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_index_4'][0], 6.0)
        self.assertAlmostEqual(vorofps['Voro_index_5'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_index_6'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_index_7'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_index_8'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_index_9'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_index_10'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_3'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_4'][0], 1.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_5'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_6'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_7'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_8'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_9'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_index_10'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_3'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_4'][0], 1.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_5'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_6'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_7'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_8'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_9'][0], 0.0)
        self.assertAlmostEqual(vorofps['Symmetry_weighted_index_10'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_vol_sum'][0], 43.614208)
        self.assertAlmostEqual(vorofps['Voro_area_sum'][0], 74.3424)
        self.assertAlmostEqual(vorofps['Voro_vol_mean'][0], 7.269034667)
        self.assertAlmostEqual(vorofps['Voro_vol_std_dev'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_vol_minimum'][0], 7.269034667)
        self.assertAlmostEqual(vorofps['Voro_vol_maximum'][0], 7.269034667)
        self.assertAlmostEqual(vorofps['Voro_area_mean'][0], 12.3904)
        self.assertAlmostEqual(vorofps['Voro_area_std_dev'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_area_minimum'][0], 12.3904)
        self.assertAlmostEqual(vorofps['Voro_area_maximum'][0], 12.3904)
        self.assertAlmostEqual(vorofps['Voro_dist_mean'][0], 3.52)
        self.assertAlmostEqual(vorofps['Voro_dist_std_dev'][0], 0.0)
        self.assertAlmostEqual(vorofps['Voro_dist_minimum'][0], 3.52)
        self.assertAlmostEqual(vorofps['Voro_dist_maximum'][0], 3.52)

    def test_interstice_distribution_of_crystal(self):
        bcc_li = Structure(Lattice([[3.51, 0, 0], [0, 3.51, 0], [0, 0, 3.51]]),
                           ["Li"] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]])
        df_bcc_li= pd.DataFrame({'struct': [bcc_li], 'site': [1]})

        interstice_distribution = IntersticeDistribution()
        intersticefp = interstice_distribution.featurize_dataframe(
            df_bcc_li, ['struct', 'site'])

        self.assertAlmostEqual(intersticefp['Interstice_vol_mean'][0], 0.32, 2)
        self.assertAlmostEqual(intersticefp['Interstice_vol_std_dev'][0], 0)
        self.assertAlmostEqual(intersticefp['Interstice_vol_minimum'][0], 0.32, 2)
        self.assertAlmostEqual(intersticefp['Interstice_vol_maximum'][0], 0.32, 2)
        self.assertAlmostEqual(intersticefp['Interstice_area_mean'][0], 0.16682, 5)
        self.assertAlmostEqual(intersticefp['Interstice_area_std_dev'][0], 0)
        self.assertAlmostEqual(intersticefp['Interstice_area_minimum'][0], 0.16682, 5)
        self.assertAlmostEqual(intersticefp['Interstice_area_maximum'][0], 0.16682, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_mean'][0], 0.06621, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_std_dev'][0], 0.07655, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_minimum'][0], 0, 3)
        self.assertAlmostEqual(intersticefp['Interstice_dist_maximum'][0], 0.15461, 5)

    def test_interstice_distribution_of_glass(self):
        cuzr_glass = Structure(Lattice([[25, 0, 0], [0, 25, 0], [0, 0, 25]]),
                           ["Cu", "Cu", "Cu", "Cu", "Cu", "Zr", "Cu", "Zr",
                            "Cu", "Zr", "Cu", "Zr", "Cu", "Cu"],
                           [[11.81159679, 16.49480537, 21.69139442],
                            [11.16777208, 17.87850033, 18.57877144],
                            [12.22394796, 15.83218325, 19.37763412],
                            [13.07053548, 14.34025424, 21.77557646],
                            [10.78147725, 19.61647494, 20.77595531],
                            [10.87541011, 14.65986432, 23.61517624],
                            [12.76631002, 18.41479521, 20.46717947],
                            [14.63911675, 16.47487037, 20.52671362],
                            [14.2470256, 18.44215167, 22.56257566],
                            [9.38050168, 16.87974592, 20.51885879],
                            [10.66332986, 14.43900833, 20.545186],
                            [11.57096832, 18.79848982, 23.26073408],
                            [13.27048138, 16.38613795, 23.59697472],
                            [9.55774984, 17.09220537, 23.1856528]],
                           coords_are_cartesian=True)
        df_glass= pd.DataFrame({'struct': [cuzr_glass], 'site': [0]})

        interstice_distribution = IntersticeDistribution()
        intersticefp = interstice_distribution.featurize_dataframe(
            df_glass, ['struct', 'site'])

        self.assertAlmostEqual(intersticefp['Interstice_vol_mean'][0], 0.28905, 5)
        self.assertAlmostEqual(intersticefp['Interstice_vol_std_dev'][0], 0.04037, 5)
        self.assertAlmostEqual(intersticefp['Interstice_vol_minimum'][0], 0.21672, 5)
        self.assertAlmostEqual(intersticefp['Interstice_vol_maximum'][0], 0.39084, 5)
        self.assertAlmostEqual(intersticefp['Interstice_area_mean'][0], 0.16070, 5)
        self.assertAlmostEqual(intersticefp['Interstice_area_std_dev'][0], 0.05245, 5)
        self.assertAlmostEqual(intersticefp['Interstice_area_minimum'][0], 0.07132, 5)
        self.assertAlmostEqual(intersticefp['Interstice_area_maximum'][0], 0.26953, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_mean'][0], 0.08154, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_std_dev'][0], 0.14778, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_minimum'][0], -0.04668, 5)
        self.assertAlmostEqual(intersticefp['Interstice_dist_maximum'][0], 0.37565, 5)

    def test_chemicalSRO(self):
        df_sc = pd.DataFrame({'struct': [self.sc], 'site': [0]})
        df_cscl = pd.DataFrame({'struct': [self.cscl], 'site': [0]})
        vnn = ChemicalSRO.from_preset("VoronoiNN", cutoff=6.0)
        vnn.fit(df_sc[['struct', 'site']])
        vnn_csros = vnn.featurize_dataframe(df_sc, ['struct', 'site'])
        self.assertAlmostEqual(vnn_csros['CSRO_Al_VoronoiNN'][0], 0.0)
        vnn = ChemicalSRO(VoronoiNN(), includes="Cs")
        vnn.fit(df_cscl[['struct', 'site']])
        vnn_csros = vnn.featurize_dataframe(df_cscl, ['struct', 'site'])
        self.assertAlmostEqual(vnn_csros['CSRO_Cs_VoronoiNN'][0], 0.0714285714)
        vnn = ChemicalSRO(VoronoiNN(), excludes="Cs")
        vnn.fit(df_cscl[['struct', 'site']])
        vnn_csros = vnn.featurize_dataframe(df_cscl, ['struct', 'site'])
        self.assertAlmostEqual(vnn_csros['CSRO_Cl_VoronoiNN'][0], -0.0714285714)
        jmnn = ChemicalSRO.from_preset("JmolNN", el_radius_updates={"Al": 1.55})
        jmnn.fit(df_sc[['struct', 'site']])
        jmnn_csros = jmnn.featurize_dataframe(df_sc, ['struct', 'site'])
        self.assertAlmostEqual(jmnn_csros['CSRO_Al_JmolNN'][0], 0.0)
        jmnn = ChemicalSRO.from_preset("JmolNN")
        jmnn.fit(df_cscl[['struct', 'site']])
        jmnn_csros = jmnn.featurize_dataframe(df_cscl, ['struct', 'site'])
        self.assertAlmostEqual(jmnn_csros['CSRO_Cs_JmolNN'][0], -0.5)
        self.assertAlmostEqual(jmnn_csros['CSRO_Cl_JmolNN'][0], -0.5)
        mdnn = ChemicalSRO.from_preset("MinimumDistanceNN")
        mdnn.fit(df_cscl[['struct', 'site']])
        mdnn_csros = mdnn.featurize_dataframe(df_cscl, ['struct', 'site'])
        self.assertAlmostEqual(mdnn_csros['CSRO_Cs_MinimumDistanceNN'][0], 0.5)
        self.assertAlmostEqual(mdnn_csros['CSRO_Cl_MinimumDistanceNN'][0], -0.5)
        monn = ChemicalSRO.from_preset("MinimumOKeeffeNN")
        monn.fit(df_cscl[['struct', 'site']])
        monn_csros = monn.featurize_dataframe(df_cscl, ['struct', 'site'])
        self.assertAlmostEqual(monn_csros['CSRO_Cs_MinimumOKeeffeNN'][0], 0.5)
        self.assertAlmostEqual(monn_csros['CSRO_Cl_MinimumOKeeffeNN'][0], -0.5)
        mvnn = ChemicalSRO.from_preset("MinimumVIRENN")
        mvnn.fit(df_cscl[['struct', 'site']])
        mvnn_csros = mvnn.featurize_dataframe(df_cscl, ['struct', 'site'])
        self.assertAlmostEqual(mvnn_csros['CSRO_Cs_MinimumVIRENN'][0], 0.5)
        self.assertAlmostEqual(mvnn_csros['CSRO_Cl_MinimumVIRENN'][0], -0.5)
        # test fit + transform
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn.fit(df_cscl[['struct', 'site']])  # dataframe
        vnn_csros = vnn.transform(df_cscl[['struct', 'site']].values)
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn.fit(df_cscl[['struct', 'site']].values)  # np.array
        vnn_csros = vnn.transform(df_cscl[['struct', 'site']].values)
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn.fit([[self.cscl, 0]])  # list
        vnn_csros = vnn.transform([[self.cscl, 0]])
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)
        # test fit_transform
        vnn = ChemicalSRO.from_preset("VoronoiNN")
        vnn_csros = vnn.fit_transform(df_cscl[['struct', 'site']].values)
        self.assertAlmostEqual(vnn_csros[0][0], 0.071428571428571286)
        self.assertAlmostEqual(vnn_csros[0][1], -0.071428571428571286)

    def test_gaussiansymmfunc(self):
        data = pd.DataFrame({'struct': [self.cscl], 'site': [0]})
        gsf = GaussianSymmFunc()
        gsfs = gsf.featurize_dataframe(data, ['struct', 'site'])
        self.assertAlmostEqual(gsfs['G2_0.05'][0], 5.0086817867593822)
        self.assertAlmostEqual(gsfs['G2_4.0'][0], 1.2415138042932981)
        self.assertAlmostEqual(gsfs['G2_20.0'][0], 0.00696)
        self.assertAlmostEqual(gsfs['G2_80.0'][0], 0.0)
        self.assertAlmostEqual(gsfs['G4_0.005_1.0_1.0'][0], 2.6399416897128658)
        self.assertAlmostEqual(gsfs['G4_0.005_1.0_-1.0'][0], 0.90049182882301426)
        self.assertAlmostEqual(gsfs['G4_0.005_4.0_1.0'][0], 1.1810690738596332)
        self.assertAlmostEqual(gsfs['G4_0.005_4.0_-1.0'][0], 0.033850556557100071)

    def test_ewald_site(self):
        ewald = EwaldSiteEnergy(accuracy=4)

        # Set the charges
        for s in [self.sc, self.cscl]:
            s.add_oxidation_state_by_guess()

        # Run the sc-Al structure
        self.assertArrayAlmostEqual(ewald.featurize(self.sc, 0), [0])

        # Run the cscl-structure
        #   Compared to a result computed using GULP
        self.assertAlmostEqual(ewald.featurize(self.cscl, 0), ewald.featurize(self.cscl, 1))
        self.assertAlmostEqual(ewald.featurize(self.cscl, 0)[0], -6.98112443 / 2, 3)

        # Re-run the Al structure to make sure it is accurate
        #  This is to test the caching feature
        self.assertArrayAlmostEqual(ewald.featurize(self.sc, 0), [0])

    def test_cns(self):
        cnv = CoordinationNumber.from_preset('VoronoiNN')
        self.assertEqual(len(cnv.feature_labels()), 1)
        self.assertEqual(cnv.feature_labels()[0], 'CN_VoronoiNN')
        self.assertAlmostEqual(cnv.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 14)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 14)
        self.assertEqual(len(cnv.citations()), 2)
        cnv = CoordinationNumber(VoronoiNN(), use_weights='sum')
        self.assertEqual(cnv.feature_labels()[0], 'CN_VoronoiNN')
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 9.2584516)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 9.2584516)
        self.assertEqual(len(cnv.citations()), 2)
        cnv = CoordinationNumber(VoronoiNN(), use_weights='effective')
        self.assertEqual(cnv.feature_labels()[0], 'CN_VoronoiNN')
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 11.648923254)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 11.648923254)
        self.assertEqual(len(cnv.citations()), 2)
        cnj = CoordinationNumber.from_preset('JmolNN')
        self.assertEqual(cnj.feature_labels()[0], 'CN_JmolNN')
        self.assertAlmostEqual(cnj.featurize(self.sc, 0)[0], 0)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 0)[0], 0)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 1)[0], 0)
        self.assertEqual(len(cnj.citations()), 1)
        jmnn = JmolNN(el_radius_updates={"Al": 1.55, "Cl": 1.7, "Cs": 1.7})
        cnj = CoordinationNumber(jmnn)
        self.assertEqual(cnj.feature_labels()[0], 'CN_JmolNN')
        self.assertAlmostEqual(cnj.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 1)[0], 8)
        self.assertEqual(len(cnj.citations()), 1)
        cnmd = CoordinationNumber.from_preset('MinimumDistanceNN')
        self.assertEqual(cnmd.feature_labels()[0], 'CN_MinimumDistanceNN')
        self.assertAlmostEqual(cnmd.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnmd.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnmd.featurize(self.cscl, 1)[0], 8)
        self.assertEqual(len(cnmd.citations()), 1)
        cnmok = CoordinationNumber.from_preset('MinimumOKeeffeNN')
        self.assertEqual(cnmok.feature_labels()[0], 'CN_MinimumOKeeffeNN')
        self.assertAlmostEqual(cnmok.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnmok.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnmok.featurize(self.cscl, 1)[0], 6)
        self.assertEqual(len(cnmok.citations()), 2)
        cnmvire = CoordinationNumber.from_preset('MinimumVIRENN')
        self.assertEqual(cnmvire.feature_labels()[0], 'CN_MinimumVIRENN')
        self.assertAlmostEqual(cnmvire.featurize(self.sc, 0)[0], 6)
        self.assertAlmostEqual(cnmvire.featurize(self.cscl, 0)[0], 8)
        self.assertAlmostEqual(cnmvire.featurize(self.cscl, 1)[0], 14)
        self.assertEqual(len(cnmvire.citations()), 2)
        self.assertEqual(len(cnmvire.implementors()), 2)
        self.assertEqual(cnmvire.implementors()[0], 'Nils E. R. Zimmermann')

    def test_grdf(self):
        f1 = Gaussian(1, 0)
        f2 = Gaussian(1, 1)
        f3 = Gaussian(1, 5)
        s_tuples = [(self.sc, 0), (self.cscl, 0)]

        # test fit, transform, and featurize dataframe for both run modes GRDF mode
        grdf = GeneralizedRadialDistributionFunction(bins=[f1, f2, f3], mode='GRDF')
        grdf.fit(s_tuples)
        features = grdf.transform(s_tuples)
        self.assertArrayAlmostEqual(features, [[4.4807e-06, 0.00031, 0.02670],
                                               [3.3303e-06, 0.00026, 0.01753]],
                                    3)
        features = grdf.featurize_dataframe(pd.DataFrame(s_tuples), [0, 1])
        self.assertArrayEqual(list(features.columns.values),
                              [0, 1, 'Gaussian center=0 width=1', 'Gaussian center=1 width=1',
                               'Gaussian center=5 width=1'])

        # pairwise GRDF mode
        grdf = GeneralizedRadialDistributionFunction(bins=[f1, f2, f3],
                                                     mode='pairwise_GRDF')
        grdf.fit(s_tuples)
        features = grdf.transform(s_tuples)
        self.assertArrayAlmostEqual(features[0],
                                    [4.4807e-06, 3.1661e-04, 0.0267],
                                    3)
        self.assertArrayAlmostEqual(features[1],
                                    [2.1807e-08, 6.1119e-06, 0.0142,
                                     3.3085e-06, 2.5898e-04, 0.0032],
                                    3)
        features = grdf.featurize_dataframe(pd.DataFrame(s_tuples),
                                            [0, 1])
        self.assertArrayEqual(list(features.columns.values),
                              [0, 1, 'site2 0 Gaussian center=0 width=1',
                               'site2 1 Gaussian center=0 width=1',
                               'site2 0 Gaussian center=1 width=1',
                               'site2 1 Gaussian center=1 width=1',
                               'site2 0 Gaussian center=5 width=1',
                               'site2 1 Gaussian center=5 width=1'])

        # test preset
        grdf = GeneralizedRadialDistributionFunction.from_preset('gaussian')
        grdf.featurize(self.sc, 0)
        self.assertArrayEqual([bin.name() for bin in grdf.bins],
                              ['Gaussian center={} width=1.0'.format(i) for i in np.arange(10.0)])

    def test_afs(self):
        f1 = Gaussian(1, 0)
        f2 = Gaussian(1, 1)
        f3 = Gaussian(1, 5)
        s_tuples = [(self.sc, 0), (self.cscl, 0)]

        # test transform,and featurize dataframe
        afs = AngularFourierSeries(bins=[f1, f2, f3])
        features = afs.transform(s_tuples)
        self.assertArrayAlmostEqual(features,
                                    [[-1.0374e-10, -4.3563e-08, -2.7914e-06,
                                      -4.3563e-08, -1.8292e-05, -0.0011,
                                      -2.7914e-06, -0.0011, -12.7863],
                                     [-1.7403e-11, -1.0886e-08, -3.5985e-06,
                                      -1.0886e-08, -6.0597e-06, -0.0016,
                                      -3.5985e-06, -0.0016, -3.9052]],
                                    3)
        features = afs.featurize_dataframe(pd.DataFrame(s_tuples),
                                           [0, 1])
        self.assertArrayEqual(list(features.columns.values),
                              [0, 1, 'AFS (Gaussian center=0 width=1, Gaussian center=0 width=1)',
                               'AFS (Gaussian center=0 width=1, Gaussian center=1 width=1)',
                               'AFS (Gaussian center=0 width=1, Gaussian center=5 width=1)',
                               'AFS (Gaussian center=1 width=1, Gaussian center=0 width=1)',
                               'AFS (Gaussian center=1 width=1, Gaussian center=1 width=1)',
                               'AFS (Gaussian center=1 width=1, Gaussian center=5 width=1)',
                               'AFS (Gaussian center=5 width=1, Gaussian center=0 width=1)',
                               'AFS (Gaussian center=5 width=1, Gaussian center=1 width=1)',
                               'AFS (Gaussian center=5 width=1, Gaussian center=5 width=1)'])

        # test preset
        afs = AngularFourierSeries.from_preset('gaussian')
        afs.featurize(self.sc, 0)
        self.assertArrayEqual([bin.name() for bin in afs.bins],
                              ['Gaussian center={} width=0.5'.format(i)
                               for i in np.arange(0, 10, 0.5)])

        afs = AngularFourierSeries.from_preset('histogram')
        afs.featurize(self.sc, 0)
        self.assertArrayEqual([bin.name() for bin in afs.bins],
                              ['Histogram start={} width=0.5'.format(i)
                               for i in np.arange(0, 10, 0.5)])

    def test_local_prop_diff(self):
        f = LocalPropertyDifference()

        # Test for Al, all features should be zero
        features = f.featurize(self.sc, 0)
        self.assertArrayAlmostEqual(features, [0])

        # Change the property to Number, compute for B1
        f.set_params(properties=['Number'])
        for i in range(2):
            features = f.featurize(self.b1, i)
            self.assertArrayAlmostEqual(features, [1])

    def test_bop(self):
        f = BondOrientationalParameter(max_l=10, compute_w=True, compute_w_hat=True)

        # Check the feature count
        self.assertEqual(30, len(f.feature_labels()))
        self.assertEqual(30, len(f.featurize(self.sc, 0)))

        f.compute_W = False
        self.assertEqual(20, len(f.feature_labels()))
        self.assertEqual(20, len(f.featurize(self.sc, 0)))

        f.compute_What = False
        self.assertEqual(10, len(f.featurize(self.sc, 0)))
        self.assertEqual(10, len(f.featurize(self.sc, 0)))

        f.compute_W = f.compute_What = True

        # Compute it for SC and B1
        sc_features = f.featurize(self.sc, 0)
        b1_features = f.featurize(self.b1, 0)

        # They should be equal
        self.assertArrayAlmostEqual(sc_features, b1_features)

        # Comparing Q's to results from https://aip.scitation.org/doi/10.1063/1.4774084
        self.assertArrayAlmostEqual([0, 0, 0, 0.764, 0, 0.354, 0, 0.718, 0, 0.411],
                                    sc_features[:10], decimal=3)

        # Comparing W's to results from https://link.aps.org/doi/10.1103/PhysRevB.28.784
        self.assertArrayAlmostEqual([0, 0, 0, 0.043022, 0, 0.000612, 0, 0.034055, 0, 0.013560],
                                    sc_features[10:20], decimal=3)

        self.assertArrayAlmostEqual([0, 0, 0, 0.159317, 0, 0.013161, 0, 0.058455, 0, 0.090130],
                                    sc_features[20:], decimal=3)

    def test_site_elem_prop(self):
        f = SiteElementalProperty.from_preset("seko-prb-2017")

        # Make sure it does the B1 structure correctly
        feat_labels = f.feature_labels()
        feats = f.featurize(self.b1, 0)
        self.assertAlmostEqual(1, feats[feat_labels.index("site Number")])

        feats = f.featurize(self.b1, 1)
        self.assertAlmostEqual(2, feats[feat_labels.index("site Number")])

        # Test the citations
        citations = f.citations()
        self.assertEqual(1, len(citations))
        self.assertIn("Seko2017", citations[0])

    def test_AverageBondLength(self):
        ft = AverageBondLength(VoronoiNN())
        self.assertAlmostEqual(ft.featurize(self.sc, 0)[0], 3.52)

        for i in range(len(self.cscl.sites)):
            self.assertAlmostEqual(ft.featurize(self.cscl, i)[0], 3.758562645051973)

        for i in range(len(self.b1.sites)):
            self.assertAlmostEqual(ft.featurize(self.b1, i)[0], 1.0)

        ft = AverageBondLength(CrystalNN())
        for i in range(len(self.cscl.sites)):
            self.assertAlmostEqual(ft.featurize(self.cscl, i)[0], 3.649153279231275)

    def test_AverageBondAngle(self):
        ft = AverageBondAngle(VoronoiNN())

        self.assertAlmostEqual(ft.featurize(self.sc, 0)[0], np.pi / 2)

        for i in range(len(self.cscl.sites)):
            self.assertAlmostEqual(ft.featurize(self.cscl, i)[0], 0.9289637531152273)

        for i in range(len(self.b1.sites)):
            self.assertAlmostEqual(ft.featurize(self.b1, i)[0], np.pi / 2)

        ft = AverageBondAngle(CrystalNN())
        for i in range(len(self.b1.sites)):
            self.assertAlmostEqual(ft.featurize(self.b1, i)[0], np.pi / 2)

    def test_SOAP(self):

        def n_soap_feat(soaper):
            n_elems = len(soaper.elements_sorted)
            lmax = soap.lmax
            nmax = soap.nmax
            n_blocks = n_elems * (n_elems + 1)/2
            n_element_features = int((lmax + 1) * nmax * (nmax + 1)/2)
            return int(n_element_features * n_blocks)

        # Test individual samples
        soap = SOAP(rcut=3.0, nmax=4, lmax=2, sigma=1, periodic=True)
        soap.fit([self.diamond])
        v = soap.featurize(self.diamond, 0)
        self.assertEqual(len(v), n_soap_feat(soap))

        soap.fit([self.ni3al])
        v = soap.featurize(self.ni3al, 0)
        self.assertEqual(len(v), n_soap_feat(soap))

        # Test dataframe fitting
        df = pd.DataFrame({"s": [self.diamond, self.ni3al, self.nacl], 'idx': [0, 1, 0]})
        soap.fit(df["s"])
        df = soap.featurize_dataframe(df, ['s', 'idx'])
        self.assertTupleEqual(df.shape, (3, n_soap_feat(soap)+2))

        # Check that only the first has carbon features
        carbon_label = df["Z=6,Z'=6,l=0,n=0,n'=0"]
        self.assertTrue(carbon_label[0] != 0)
        self.assertTrue(carbon_label[1] == 0)
        self.assertTrue(carbon_label[2] == 0)

    def tearDown(self):
        del self.sc
        del self.cscl

if __name__ == '__main__':
    import unittest
    unittest.main()
