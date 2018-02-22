from __future__ import unicode_literals, division, print_function

import numpy as np
import pandas as pd
from pymatgen import Structure, Lattice
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.local_env import VoronoiNN, JMolNN

from matminer.featurizers.site import AGNIFingerprints, \
    OPSiteFingerprint, CrystalSiteFingerprint, EwaldSiteEnergy, \
    VoronoiFingerprint, ChemEnvSiteFingerprint, \
    CoordinationNumber, ChemicalSRO, GaussianSymmFunc

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

    def test_crystal_site_fingerprint(self):
        csf = CrystalSiteFingerprint.from_preset('ops')
        l = csf.feature_labels()
        t = ['wt CN_1', 'wt CN_2', 'L-shaped CN_2', 'water-like CN_2', \
             'bent 120 degrees CN_2', 'bent 150 degrees CN_2', 'linear CN_2', \
             'wt CN_3', 'trigonal planar CN_3', 'trigonal non-coplanar CN_3', \
             'T-shaped CN_3', 'wt CN_4', 'square co-planar CN_4', \
             'tetrahedral CN_4', 'rectangular see-saw-like CN_4', \
             'see-saw-like CN_4', 'trigonal pyramidal CN_4', 'wt CN_5', \
             'pentagonal planar CN_5', 'square pyramidal CN_5', \
             'trigonal bipyramidal CN_5', 'wt CN_6', 'hexagonal planar CN_6', \
             'octahedral CN_6', 'pentagonal pyramidal CN_6', 'wt CN_7', \
             'hexagonal pyramidal CN_7', 'pentagonal bipyramidal CN_7', \
             'wt CN_8', 'body-centered cubic CN_8', \
             'hexagonal bipyramidal CN_8', 'wt CN_9', 'q2 CN_9', 'q4 CN_9', \
             'q6 CN_9', 'wt CN_10', 'q2 CN_10', 'q4 CN_10', 'q6 CN_10', \
             'wt CN_11', 'q2 CN_11', 'q4 CN_11', 'q6 CN_11', 'wt CN_12', \
             'cuboctahedral CN_12', 'q2 CN_12', 'q4 CN_12', 'q6 CN_12']
        for i in range(len(l)):
            self.assertEqual(l[i], t[i])
        ops = csf.featurize(self.sc, 0)
        self.assertEqual(len(ops), 48)
        self.assertAlmostEqual(ops[csf.feature_labels().index(
            'wt CN_6')], 1, places=7)
        self.assertAlmostEqual(ops[csf.feature_labels().index(
            'octahedral CN_6')], 1, places=7)
        ops = csf.featurize(self.cscl, 0)
        self.assertAlmostEqual(ops[csf.feature_labels().index(
            'wt CN_8')], 0.5575257, places=7)
        self.assertAlmostEqual(ops[csf.feature_labels().index(
            'body-centered cubic CN_8')], 0.5329344, places=7)

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
        data = pd.DataFrame({'struct': [self.sc], 'site': [0]})
        vorofp = VoronoiFingerprint(use_weights=True)
        vorofps = vorofp.featurize_dataframe(data, ['struct', 'site'])
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

    def test_chemicalSRO(self):
        data = pd.DataFrame({'struct': [self.sc], 'site': [0]})
        csros = ChemicalSRO.from_preset("VoronoiNN").\
            featurize_dataframe(data, ['struct', 'site'])
        self.assertAlmostEqual(csros['CSRO_Al_VoronoiNN'][0], 0.0)
        csros = ChemicalSRO(JMolNN(el_radius_updates = {"Al": 1.55})).\
            featurize_dataframe(data, ['struct', 'site'])
        self.assertAlmostEqual(csros['CSRO_Al_JMolNN'][0], 0.0)
        csros = ChemicalSRO.from_preset("MinimumDistanceNN").\
            featurize_dataframe(data, ['struct', 'site'])
        self.assertAlmostEqual(csros['CSRO_Al_MinimumDistanceNN'][0], 0.0)
        csros = ChemicalSRO.from_preset("MinimumOKeeffeNN").\
            featurize_dataframe(data, ['struct', 'site'])
        self.assertAlmostEqual(csros['CSRO_Al_MinimumOKeeffeNN'][0], 0.0)
        csros = ChemicalSRO.from_preset("MinimumVIRENN").\
            featurize_dataframe(data, ['struct', 'site'])
        self.assertAlmostEqual(csros['CSRO_Al_MinimumVIRENN'][0], 0.0)

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
        cnv = CoordinationNumber(VoronoiNN(), use_weights=True)
        self.assertEqual(cnv.feature_labels()[0], 'CN_VoronoiNN')
        self.assertAlmostEqual(cnv.featurize(self.cscl, 0)[0], 9.2584516)
        self.assertAlmostEqual(cnv.featurize(self.cscl, 1)[0], 9.2584516)
        self.assertEqual(len(cnv.citations()), 2)
        cnj = CoordinationNumber.from_preset('JMolNN')
        self.assertEqual(cnj.feature_labels()[0], 'CN_JMolNN')
        self.assertAlmostEqual(cnj.featurize(self.sc, 0)[0], 0)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 0)[0], 0)
        self.assertAlmostEqual(cnj.featurize(self.cscl, 1)[0], 0)
        self.assertEqual(len(cnj.citations()), 1)
        jmnn = JMolNN(el_radius_updates={"Al": 1.55, "Cl": 1.7, "Cs": 1.7})
        cnj = CoordinationNumber(jmnn)
        self.assertEqual(cnj.feature_labels()[0], 'CN_JMolNN')
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
        self.assertEqual(len(cnmvire.implementors()), 1)
        self.assertEqual(cnmvire.implementors()[0], 'Nils E. R. Zimmermann')

    def tearDown(self):
        del self.sc
        del self.cscl

if __name__ == '__main__':
    import unittest
    unittest.main()
