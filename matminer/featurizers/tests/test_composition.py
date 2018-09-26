from __future__ import unicode_literals, division, print_function

import math
import unittest
import warnings
from unittest import SkipTest

import pandas as pd
import numpy as np
from pymatgen import Composition, MPRester
from pymatgen.core.periodic_table import Specie, Element
from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.composition import Stoichiometry, ElementProperty, \
    ValenceOrbital, IonProperty, \
    ElementFraction, TMetalFraction, ElectronAffinity, ElectronegativityDiff, \
    CohesiveEnergy, \
    BandCenter, Miedema, CationProperty, OxidationStates, AtomicOrbitals, \
    YangSolidSolution, AtomicPackingEfficiency


class CompositionFeaturesTest(PymatgenTest):

    def setUp(self):
        self.df = pd.DataFrame({"composition": [Composition("Fe2O3"),
                                                Composition({Specie("Fe", 2): 1, Specie("O", -2): 1})]})

    def test_stoich(self):
        featurizer = Stoichiometry(num_atoms=True)
        df_stoich = Stoichiometry(num_atoms=True).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_stoich["num atoms"][0], 5)
        self.assertAlmostEqual(df_stoich["0-norm"][0], 2)
        self.assertAlmostEqual(df_stoich["7-norm"][0], 0.604895199)

        # Test whether the number of formula units affects result
        original_value = featurizer.featurize(Composition("FeO"))
        self.assertArrayAlmostEqual(featurizer.featurize(Composition("Fe0.5O0.5")), original_value)
        self.assertArrayAlmostEqual(featurizer.featurize(Composition("Fe2O2")), original_value)

    def test_elem(self):
        df_elem = ElementProperty.from_preset("magpie").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["minimum Number"][0], 8)
        self.assertAlmostEqual(df_elem["maximum Number"][0], 26)
        self.assertAlmostEqual(df_elem["range Number"][0], 18)
        self.assertAlmostEqual(df_elem["mean Number"][0], 15.2)
        self.assertAlmostEqual(df_elem["avg_dev Number"][0], 8.64)
        self.assertAlmostEqual(df_elem["mode Number"][0], 8)

    def test_elem_deml(self):
        df_elem_deml = ElementProperty.from_preset("deml").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem_deml["minimum atom_num"][0], 8)
        self.assertAlmostEqual(df_elem_deml["maximum atom_num"][0], 26)
        self.assertAlmostEqual(df_elem_deml["range atom_num"][0], 18)
        self.assertAlmostEqual(df_elem_deml["mean atom_num"][0], 15.2)
        self.assertAlmostEqual(df_elem_deml["std_dev atom_num"][0], 12.7279, 4)

    def test_cation_properties(self):
        featurizer = CationProperty.from_preset("deml")
        features = dict(zip(featurizer.feature_labels(), featurizer.featurize(self.df["composition"][1])))
        self.assertAlmostEqual(features["minimum magn_moment of cations"], 5.48)
        self.assertAlmostEqual(features["maximum magn_moment of cations"], 5.48)
        self.assertAlmostEqual(features["range magn_moment of cations"], 0)
        self.assertAlmostEqual(features["mean magn_moment of cations"], 5.48)
        self.assertAlmostEqual(features["std_dev magn_moment of cations"], 0)

    def test_elem_matminer(self):
        df_elem = ElementProperty.from_preset("matminer").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["minimum melting_point"][0], 54.8, 1)
        self.assertTrue(math.isnan(df_elem["maximum bulk_modulus"][0]))
        self.assertAlmostEqual(df_elem["range X"][0], 1.61, 1)
        self.assertAlmostEqual(df_elem["mean X"][0], 2.796, 1)
        self.assertAlmostEqual(df_elem["maximum block"][0], 3, 1)

    def test_valence(self):
        df_val = ValenceOrbital().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_val["avg s valence electrons"][0], 2.0)
        self.assertAlmostEqual(df_val["avg p valence electrons"][0], 2.4)
        self.assertAlmostEqual(df_val["avg d valence electrons"][0], 2.4)
        self.assertAlmostEqual(df_val["avg f valence electrons"][0], 0.0)
        self.assertAlmostEqual(df_val["frac s valence electrons"][0], 0.294117647)
        self.assertAlmostEqual(df_val["frac d valence electrons"][0], 0.352941176)
        self.assertAlmostEqual(df_val["frac p valence electrons"][0], 0.352941176)
        self.assertAlmostEqual(df_val["frac f valence electrons"][0], 0)

    def test_ionic(self):
        featurizer = IonProperty()
        df_ionic = featurizer.featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_ionic["compound possible"][0], 1.0)
        self.assertAlmostEqual(df_ionic["max ionic char"][0], 0.476922164)
        self.assertAlmostEqual(df_ionic["avg ionic char"][0], 0.114461319)

        # Test 'fast'
        self.assertEquals(1.0, featurizer.featurize(Composition("Fe3O4"))[0])
        featurizer.fast = True
        self.assertEquals(0, featurizer.featurize(Composition("Fe3O4"))[0])

        # Make sure 'fast' works if I use-precomputed oxidation states
        self.assertEquals(1, featurizer.featurize(Composition({
            Specie('Fe', 2): 1,
            Specie('Fe', 3): 2,
            Specie('O', -2): 4
        }))[0])

    def test_fraction(self):
        df_frac = ElementFraction().featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_frac["O"][0], 0.6)
        self.assertEqual(df_frac["Fe"][0], 0.4)

    def test_tm_fraction(self):
        df_tm_frac = TMetalFraction().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_tm_frac["transition metal fraction"][0], 0.4)

    def test_elec_affin(self):
        featurizer = ElectronAffinity()
        self.assertAlmostEqual(-141000*2, featurizer.featurize(self.df["composition"][1])[0])

    def test_en_diff(self):
        featurizer = ElectronegativityDiff()
        features = dict(zip(featurizer.feature_labels(), featurizer.featurize(self.df["composition"][1])))
        self.assertAlmostEqual(features["minimum EN difference"], 1.6099999999)
        self.assertAlmostEqual(features["maximum EN difference"], 1.6099999999)
        self.assertAlmostEqual(features["range EN difference"], 0)
        self.assertAlmostEqual(features["mean EN difference"], 1.6099999999)
        self.assertAlmostEqual(features["std_dev EN difference"], 0)

    def test_fere_corr(self):
        df_fere_corr = ElementProperty(features=["FERE correction"],
                                       stats=["minimum", "maximum", "range", "mean", "std_dev"],
                                       data_source="deml")\
            .featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_fere_corr["minimum FERE correction"][0], -0.15213431610903)
        self.assertAlmostEqual(df_fere_corr["maximum FERE correction"][0], 0.23)
        self.assertAlmostEqual(df_fere_corr["range FERE correction"][0], 0.382134316)
        self.assertAlmostEqual(df_fere_corr["mean FERE correction"][0], 0.077146274)
        self.assertAlmostEqual(df_fere_corr["std_dev FERE correction"][0], 0.270209766)

    def test_atomic_orbitals(self):
        df_atomic_orbitals = AtomicOrbitals().featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_atomic_orbitals['HOMO_character'][0], 'd')
        self.assertEqual(df_atomic_orbitals['HOMO_element'][0], 'Fe')
        self.assertEqual(df_atomic_orbitals['HOMO_energy'][0], -0.295049)
        self.assertEqual(df_atomic_orbitals['LUMO_character'][0], 'd')
        self.assertEqual(df_atomic_orbitals['LUMO_element'][0], 'Fe')
        self.assertEqual(df_atomic_orbitals['LUMO_energy'][0], -0.295049)
        self.assertEqual(df_atomic_orbitals['gap_AO'][0], 0.0)

        # test that fractional compositions return the same features
        self.assertEqual(AtomicOrbitals().featurize(Composition('Na0.5Cl0.5')),
                         AtomicOrbitals().featurize(Composition('NaCl')))

        # test if warning is raised upon composition truncation in dilute cases
        self.assertWarns(UserWarning, AtomicOrbitals().featurize,
                         Composition('Fe1C0.00000001'))


    def test_band_center(self):
        df_band_center = BandCenter().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_band_center["band center"][0], -2.672486385)

    def test_oxidation_states(self):
        featurizer = OxidationStates.from_preset("deml")
        features = dict(zip(featurizer.feature_labels(), featurizer.featurize(self.df["composition"][1])))
        self.assertAlmostEquals(4, features["range oxidation state"])
        self.assertAlmostEquals(2, features["maximum oxidation state"])

    def test_cohesive_energy(self):
        mpr = MPRester()
        if not mpr.api_key:
            raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")
        df_cohesive_energy = CohesiveEnergy().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_cohesive_energy["cohesive energy"][0], 5.15768, 2)

    def test_miedema_all(self):
        df = pd.DataFrame({"composition": [Composition("TiZr"),
                                           Composition("Mg10Cu50Ca40"),
                                           Composition("Fe2O3")]})
        miedema = Miedema(struct_types='all')
        mfps = miedema.featurize_dataframe(df, col_id="composition")
        self.assertAlmostEqual(mfps['Miedema_deltaH_inter'][0], -0.003445022152)
        self.assertAlmostEqual(mfps['Miedema_deltaH_amor'][0], 0.0707658836300)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_min'][0], 0.03663599755)

        self.assertAlmostEqual(mfps['Miedema_deltaH_inter'][1], -0.235125978427)
        self.assertAlmostEqual(mfps['Miedema_deltaH_amor'][1], -0.164541848271)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_min'][1], -0.05280843311)

        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_inter'][2]), True)
        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_amor'][2]), True)
        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_ss_min'][2]), True)

    def test_miedema_ss(self):
        df = pd.DataFrame({"composition": [Composition("TiZr"),
                                           Composition("Mg10Cu50Ca40"),
                                           Composition("Fe2O3")]})
        miedema = Miedema(struct_types='ss', ss_types=['min', 'fcc', 'bcc',
                                                       'hcp', 'no_latt'])
        mfps = miedema.featurize_dataframe(df, col_id="composition")
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_min'][0], 0.03663599755)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_fcc'][0], 0.04700027066)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_bcc'][0], 0.08327522653)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_hcp'][0], 0.03663599755)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_no_latt'][0], 0.036635998)

        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_min'][1], -0.05280843311)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_fcc'][1], 0.03010575174)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_bcc'][1], -0.05280843311)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_hcp'][1], 0.03010575174)
        self.assertAlmostEqual(mfps['Miedema_deltaH_ss_no_latt'][1], -0.0035781359)

        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_ss_min'][2]), True)
        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_ss_fcc'][2]), True)
        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_ss_bcc'][2]), True)
        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_ss_hcp'][2]), True)
        self.assertAlmostEqual(math.isnan(mfps['Miedema_deltaH_ss_no_latt'][2]), True)

    def test_yang(self):
        comps = list(map(Composition, ["ZrHfTiCuNi", "CuNi",
                                       "CoCrFeNiCuAl0.3", "CoCrFeNiCuAl"]))

        # Run the featurization
        feat = YangSolidSolution()
        feat.set_n_jobs(1)
        features = feat.featurize_many(comps)

        # Check the results
        #  These are compared to results from the original paper,
        #   except for CoCrFeNiCuAl0.3, where the paper reports a value
        #   exactly 1/10th of what I compute using Excel and matminer
        # I use a high tolerance because matminer uses a different source
        #   of radii than the original paper (do not have Kittel's atomic
        #   radii available)
        self.assertEqual((4, 2), np.array(features).shape)
        self.assertArrayAlmostEqual([0.95, 0.1021], features[0], decimal=2)
        self.assertArrayAlmostEqual([2.22, 0.0], features[1], decimal=2)
        self.assertArrayAlmostEqual([158.5, 0.0315], features[2], decimal=1)
        self.assertArrayAlmostEqual([5.06, 0.0482], features[3], decimal=1)

    def test_ape(self):
        f = AtomicPackingEfficiency()
        ef = ElementFraction()
        ef.set_n_jobs(1)

        # Test the APE calculation routines
        self.assertAlmostEqual(1.11632, f.get_ideal_radius_ratio(15))
        self.assertAlmostEqual(0.154701, f.get_ideal_radius_ratio(2))
        self.assertAlmostEqual(1.65915, f.get_ideal_radius_ratio(27))
        self.assertAlmostEqual(15, f.find_ideal_cluster_size(1.116)[0])
        self.assertAlmostEqual(3, f.find_ideal_cluster_size(0.1)[0])
        self.assertAlmostEqual(24, f.find_ideal_cluster_size(2)[0])

        # Test the nearest neighbor lookup tool
        nn_lookup = f.create_cluster_lookup_tool([Element('Cu'),
                                                  Element('Zr')])

        #  Check that the table gets the correct structures
        stable_clusters = [Composition('CuZr10'), Composition('Cu6Zr6'),
                           Composition('Cu8Zr5'), Composition('Cu13Zr1'),
                           Composition('Cu3Zr12'), Composition('Cu8Zr8'),
                           Composition('Cu12Zr5'), Composition('Cu17Zr')]
        ds, _ = nn_lookup.kneighbors(
            ef.featurize_many(stable_clusters),
            n_neighbors=1)
        self.assertArrayAlmostEqual([[0]]*8, ds)
        self.assertEquals(8, nn_lookup._fit_X.shape[0])

        # Swap the order of the clusters, make sure it gets the same list
        nn_lookup_swapped = f.create_cluster_lookup_tool([Element('Zr'),
                                                          Element('Cu')])
        self.assertArrayAlmostEqual(sorted(nn_lookup._fit_X.tolist()),
                                    sorted(nn_lookup_swapped._fit_X.tolist()))

        # Make sure we had a cache hit
        self.assertEquals(1, f._create_cluster_lookup_tool.cache_info().misses)
        self.assertEquals(1, f._create_cluster_lookup_tool.cache_info().hits)

        # Change the tolerance, see if it changes the results properly
        f.threshold = 0.002
        nn_lookup = f.create_cluster_lookup_tool([Element('Cu'),
                                                  Element('Zr')])
        self.assertEquals(2, nn_lookup._fit_X.shape[0])
        ds, _ = nn_lookup.kneighbors(
            ef.featurize_many([Composition('CuZr10'), Composition('Cu3Zr12')]),
            n_neighbors=1)
        self.assertArrayAlmostEqual([[0]]*2, ds)

        # Make sure we had a cache miss
        self.assertEquals(2, f._create_cluster_lookup_tool.cache_info().misses)
        self.assertEquals(1, f._create_cluster_lookup_tool.cache_info().hits)

        # Compute the distances from Cu50Zr50
        mean_dists = f.compute_nearest_cluster_distance(Composition('CuZr'))
        self.assertArrayAlmostEqual([0.424264, 0.667602, 0.800561], mean_dists,
                                    decimal=6)

        # Compute the optimal APE for Cu50Zr50
        self.assertArrayAlmostEqual([0.000233857, 0.003508794],
                                    f.compute_simultaneous_packing_efficiency(
                                        Composition('Cu50Zr50')
                                    ))

        # Test the dataframe calculator
        df = pd.DataFrame({'comp': [Composition('CuZr')]})
        f.featurize_dataframe(df, 'comp')

        self.assertEquals(6, len(df.columns))
        self.assertIn('dist from 5 clusters |APE| < 0.002', df.columns)

        self.assertAlmostEqual(0.003508794,
                               df['mean abs simul. packing efficiency'][0])

        # Make sure it works with composition that do not match any efficient clusters
        feat = f.compute_nearest_cluster_distance(Composition('Al'))
        self.assertArrayAlmostEqual([1]*3, feat)

if __name__ == '__main__':
    unittest.main()
