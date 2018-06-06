from __future__ import unicode_literals, division, print_function

import json
import os
import pandas as pd
import unittest

from matminer.featurizers.dos import DOSFeaturizer, DopingFermi, BandEdge
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.util.testing import PymatgenTest

test_dir = os.path.join(os.path.dirname(__file__))


class DOSFeaturesTest(PymatgenTest):

    def setUp(self):
        with open(os.path.join(test_dir, 'si_dos.json'), 'r') as sDOS:
            si_dos = CompleteDos.from_dict(json.load(sDOS))
        self.df = pd.DataFrame({'dos': [si_dos]})

    def test_DOSFeaturizer(self):
        df_df = DOSFeaturizer(contributors=2).featurize_dataframe(self.df, col_id=['dos'])
        # CBM:
        self.assertAlmostEqual(df_df['cbm_score_1'][0], 0.258, 3)
        self.assertAlmostEqual(df_df['cbm_score_2'][0], 0.258, 3)
        self.assertEqual(df_df['cbm_location_1'][0], '0.0;0.0;0.0')
        self.assertEqual(df_df['cbm_location_2'][0], '0.25;0.25;0.25')
        self.assertEqual(df_df['cbm_specie_1'][0], 'Si')
        self.assertEqual(df_df['cbm_character_1'][0], 's')
        self.assertEqual(df_df['cbm_nsignificant'][0], 4)
        # VBM:
        self.assertAlmostEqual(df_df['vbm_score_1'][0], 0.490, 3)
        self.assertAlmostEqual(df_df['vbm_score_2'][0], 0.490, 3)
        self.assertEqual(df_df['vbm_location_1'][0], '0.0;0.0;0.0')
        self.assertEqual(df_df['vbm_location_2'][0], '0.25;0.25;0.25')
        self.assertEqual(df_df['vbm_specie_1'][0], 'Si')
        self.assertEqual(df_df['vbm_character_1'][0], 'p')
        self.assertEqual(df_df['vbm_nsignificant'][0], 2)

    def test_DopingFermi(self):
        dopings = [-1e18, -1e20, 1e18, 1e20]
        df = DopingFermi(dopings=dopings, eref="midgap", return_eref=True
                         ).featurize_dataframe(self.df, col_id=['dos'])
        self.assertAlmostEqual(df['fermi_c-1e+18T300'][0], 6.138458, places=4)
        self.assertAlmostEqual(df['fermi_c-1e+20T300'][0], 6.258075, places=4)
        self.assertAlmostEqual(df['fermi_c1e+18T300'][0], 5.497809, places=4)
        self.assertAlmostEqual(df['fermi_c1e+20T300'][0], 5.37833, places=4)
        self.assertAlmostEqual(df['midgap eref'][0], 5.8162, places=4)
        # the fermi levels with experimental band gap of silicon:
        dofe = DopingFermi(dopings, return_eref=True)
        feats = dofe.featurize(dos=self.df['dos'][0], bandgap=1.14)
        self.assertAlmostEqual(feats[0], 6.217457, places=4)
        self.assertAlmostEqual(feats[1], 6.337074, places=4)
        self.assertAlmostEqual(feats[2], 5.4188086, places=4)
        self.assertAlmostEqual(feats[3], 5.2993298, places=4)
        self.assertAlmostEqual(feats[4], 5.8162, places=4) # same reference

    def test_BandEdge(self):
        be = BandEdge(energy_cutoff=0.1, species=['Si'])
        df = be.featurize_dataframe(self.df, col_id='dos', inplace=False)
        df = df.drop('dos', axis=1)
        # ensure features are in [0., 1.]
        self.assertEqual((df<0).sum().sum(), 0.0)
        self.assertEqual((df>1).sum().sum(), 0.0)
        # cbm orbitals
        self.assertAlmostEqual(df['cbm_s'][0], 0.51, 2)
        self.assertEqual(df['cbm_s'][0], df['cbm_Si_s'][0])
        self.assertAlmostEqual(df['cbm_p'][0], 0.49, 2)
        self.assertGreater(df['cbm_sp'][0], 0.98)
        # vbm orbitals
        self.assertAlmostEqual(df['vbm_s'][0], 0.016, 3)
        self.assertEqual(df['vbm_s'][0], df['vbm_Si_s'][0])
        self.assertAlmostEqual(df['vbm_p'][0], 0.984, 3)
        self.assertAlmostEqual(df['vbm_sp'][0], 0.061, 3)
        df = self.df
        df['cutoff'] = [1.0] # digging deeper inside the band
        df = be.featurize_dataframe(df, col_id=['dos', 'cutoff'])
        self.assertAlmostEqual(df['cbm_s'][0], 0.46, 2)
        self.assertAlmostEqual(df['vbm_p'][0], 0.96, 2)


if __name__ == '__main__':
    unittest.main()

