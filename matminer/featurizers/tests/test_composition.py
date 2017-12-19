from __future__ import unicode_literals, division, print_function

import unittest
from unittest import SkipTest

import math
import pandas as pd

from pymatgen import Composition, MPRester
from pymatgen.util.testing import PymatgenTest

from matminer.featurizers.composition import Stoichiometry, ElementProperty, ValenceOrbital, IonProperty, ElementFraction, TMetalFraction, ElectronAffinity, ElectronegativityDiff, FERECorrection, CohesiveEnergy, BandCenter, Miedema


class CompositionFeaturesTest(PymatgenTest):

    def setUp(self):
        self.df = pd.DataFrame({"composition":[Composition("Fe2O3")]})

    def test_stoich(self):
        df_stoich = Stoichiometry(num_atoms=True).featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_stoich["num atoms"][0], 5)
        self.assertAlmostEqual(df_stoich["0-norm"][0], 2)
        self.assertAlmostEqual(df_stoich["7-norm"][0], 0.604895199)

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
        self.assertAlmostEqual(df_elem_deml["std_dev atom_num"][0], 8.81816307)
        #Charge dependent property
        self.assertAlmostEqual(df_elem_deml["minimum magn_moment"][0], 0)
        self.assertAlmostEqual(df_elem_deml["maximum magn_moment"][0], 5.2)
        self.assertAlmostEqual(df_elem_deml["range magn_moment"][0], 5.2)
        self.assertAlmostEqual(df_elem_deml["mean magn_moment"][0], 2.08)
        self.assertAlmostEqual(df_elem_deml["std_dev magn_moment"][0], 2.547469332)

    def test_elem_matminer(self):
        df_elem = ElementProperty.from_preset("matminer").featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elem["minimum melting_point"][0], 54.8, 1)
        self.assertTrue(math.isnan(df_elem["maximum bulk_modulus"][0]))
        self.assertAlmostEqual(df_elem["range X"][0], 1.61, 1)
        self.assertAlmostEqual(df_elem["mean X"][0], 2.796, 1)

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
        df_ionic = IonProperty().featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_ionic["compound possible"][0], 1.0)
        self.assertAlmostEqual(df_ionic["max ionic char"][0], 0.476922164)
        self.assertAlmostEqual(df_ionic["avg ionic char"][0], 0.114461319)

    def test_fraction(self):
        df_frac = ElementFraction().featurize_dataframe(self.df, col_id="composition")
        self.assertEqual(df_frac["O"][0], 0.6)
        self.assertEqual(df_frac["Fe"][0], 0.4)
        #self.assertAlmostEqual(df_frac["Fe"][1], 0.42857143)
        #self.assertAlmostEqual(df_frac["Li"][1], 0.57142857)

    def test_tm_fraction(self):
        df_tm_frac = TMetalFraction().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_tm_frac["transition metal fraction"][0], 0.4)

    def test_elec_affin(self):
        df_elec_affin = ElectronAffinity().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_elec_affin["avg anion electron affinity "][0], -169200)

    def test_en_diff(self):
        df_en_diff = ElectronegativityDiff().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_en_diff["minimum EN difference"][0], 1.6099999999)
        self.assertAlmostEqual(df_en_diff["maximum EN difference"][0], 1.6099999999)
        self.assertAlmostEqual(df_en_diff["range EN difference"][0], 0)
        self.assertAlmostEqual(df_en_diff["mean EN difference"][0], 1.6099999999)
        self.assertAlmostEqual(df_en_diff["std_dev EN difference"][0], 0)

    def test_fere_corr(self):
        df_fere_corr = FERECorrection().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_fere_corr["minimum FERE correction"][0], -0.15213431610903)
        self.assertAlmostEqual(df_fere_corr["maximum FERE correction"][0], 0.23)
        self.assertAlmostEqual(df_fere_corr["range FERE correction"][0], 0.382134316)
        self.assertAlmostEqual(df_fere_corr["mean FERE correction"][0], 0.077146274)
        self.assertAlmostEqual(df_fere_corr["std_dev FERE correction"][0], 0.270209766)

    def test_band_center(self):
        df_band_center = BandCenter().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_band_center["band center"][0], -2.672486385)

    def test_cohesive_energy(self):
        mpr = MPRester()
        if not mpr.api_key:
            raise SkipTest("Materials Project API key not set; Skipping cohesive energy test")
        df_cohesive_energy = CohesiveEnergy().featurize_dataframe(self.df, col_id="composition")
        self.assertAlmostEqual(df_cohesive_energy["cohesive energy"][0], 5.15768, 2)

    def test_miedema_all(self):
        miedema_df = pd.DataFrame({"composition": [Composition("TiZr"), Composition("Mg10Cu50Ca40"), Composition("Fe2O3")]})
        df_miedema = Miedema(struct_types='all').featurize_dataframe(miedema_df, col_id="composition")
        self.assertAlmostEqual(df_miedema['formation_enthalpy_inter'][0], -0.0034450221522328503)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_amor'][0], 0.070765883630040161)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_min'][0], 0.036635997549833224)

        self.assertAlmostEqual(df_miedema['formation_enthalpy_inter'][1], -0.23512597842733007)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_amor'][1], -0.16454184827089643)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_min'][1], -0.052808433113994087)

        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_inter'][2]), True)
        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_amor'][2]), True)
        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_ss_min'][2]), True)

    def test_miedema_ss(self):
        miedema_df = pd.DataFrame({"composition": [Composition("TiZr"), Composition("Mg10Cu50Ca40"), Composition("Fe2O3")]})
        df_miedema = Miedema(struct_types='ss',
                             ss_types=['min', 'fcc', 'bcc', 'hcp', 'no_latt']).featurize_dataframe(miedema_df, col_id="composition")
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_min'][0], 0.036635997549833224)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_fcc'][0], 0.047000270656721008)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_bcc'][0], 0.083275226530828264)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_hcp'][0], 0.036635997549833224)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_no_latt'][0], 0.036635997549833224)

        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_min'][1], -0.052808433113994087)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_fcc'][1], 0.030105751741108196)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_bcc'][1], -0.052808433113994087)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_hcp'][1], 0.030105751741108196)
        self.assertAlmostEqual(df_miedema['formation_enthalpy_ss_no_latt'][1], -0.0035781358562771083)

        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_ss_min'][2]), True)
        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_ss_fcc'][2]), True)
        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_ss_bcc'][2]), True)
        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_ss_hcp'][2]), True)
        self.assertAlmostEqual(math.isnan(df_miedema['formation_enthalpy_ss_no_latt'][2]), True)

if __name__ == '__main__':
    unittest.main()
