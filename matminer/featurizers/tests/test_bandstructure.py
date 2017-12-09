from __future__ import unicode_literals, division, print_function

import json
import math
import os
import pandas as pd
import unittest

from matminer.featurizers.bandstructure import BandFeaturizer, \
    BranchPointEnergy, DOSFeaturizer
from pymatgen.core import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, \
    BandStructure
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.testing import PymatgenTest

test_dir = os.path.join(os.path.dirname(__file__))


class BandstructureFeaturesTest(PymatgenTest):

    def setUp(self):
        with open(os.path.join(test_dir, 'si_structure.json'),'r') as sth:
            si_str = Structure.from_dict(json.load(sth))

        with open(os.path.join(test_dir, 'si_bandstructure_line.json'),'r') as bsh:
            si_bs_line = BandStructureSymmLine.from_dict(json.load(bsh))
        si_bs_line.structure = si_str

        with open(os.path.join(test_dir, 'si_bandstructure_uniform.json'),'r') as bsh:
            si_bs_uniform = BandStructure.from_dict(json.load(bsh))
        si_bs_uniform.structure = si_str
        self.si_kpts = list(HighSymmKpath(si_str).kpath['kpoints'].values())
        self.df = pd.DataFrame({'bs_line': [si_bs_line], 'bs_uniform': [si_bs_uniform]})

        with open(os.path.join(test_dir, 'VBr2_971787_bandstructure.json'), 'r') as bsh:
            vbr2_uniform = BandStructure.from_dict(json.load(bsh))

        self.vbr2kpts = [k.frac_coords for k in vbr2_uniform.labels_dict.values()]
        self.vbr2kpts = [[0.0, 0.0, 0.0], # \\Gamma
                         [0.2, 0.0, 0.0], # between \\Gamma and M
                         [0.5, 0.0, 0.0], # M
                         [0.5, 0.0, 0.5]] # L

        self.df2 = pd.DataFrame({'bs_line': [vbr2_uniform]})

    def test_BranchPointEnergy(self):
        df_bpe = BranchPointEnergy().featurize_dataframe(self.df,
                                                         col_id=['bs_uniform'])
        self.assertAlmostEqual(df_bpe['branch_point_energy'][0], 5.728, 3)
        self.assertAlmostEqual(df_bpe['cbm_absolute'][0], 0.497, 3)
        self.assertAlmostEqual(df_bpe['vbm_absolute'][0], -0.114, 3)

    def test_BandFeaturizer(self):
        # silicon:
        bs_featurizer = BandFeaturizer(kpoints=self.si_kpts, nbands=5)
        df_bf = bs_featurizer.featurize_dataframe(self.df, col_id='bs_line')
        self.assertAlmostEqual(df_bf['band_gap'][0], 0.612, 3)
        self.assertAlmostEqual(df_bf['direct_gap'][0], 2.557, 3)
        self.assertAlmostEqual(df_bf['n_ex1_norm'][0], 0.58413, 5)
        self.assertAlmostEqual(df_bf['p_ex1_norm'][0], 0.0, 5)
        self.assertEqual(df_bf['is_gap_direct'][0], False)
        self.assertEqual(df_bf['n_ex1_degen'][0], 6)
        self.assertEqual(df_bf['p_ex1_degen'][0], 1)
        # \\Gamma:
        self.assertAlmostEqual(df_bf['n_0.0;0.0;0.0_en4'][0], 2.5169, 4)
        self.assertAlmostEqual(df_bf['n_0.0;0.0;0.0_en1'][0], 1.945, 4)
        self.assertEqual(df_bf['p_0.0;0.0;0.0_en1'][0], 0.0)
        self.assertEqual(df_bf['p_0.0;0.0;0.0_en2'][0], 0.0)
        self.assertEqual(df_bf['p_0.0;0.0;0.0_en4'][0], -11.8118)
        # K:
        self.assertAlmostEqual(df_bf['p_0.375;0.375;0.75_en1'][0], -2.3745, 4)
        # X:
        self.assertAlmostEqual(df_bf['n_0.5;0.0;0.5_en2'][0], 0.1409, 4)
        self.assertAlmostEqual(df_bf['n_0.5;0.0;0.5_en1'][0], 0.1409, 4)
        self.assertAlmostEqual(df_bf['p_0.5;0.0;0.5_en1'][0], -2.7928, 4)
        # U:
        self.assertAlmostEqual(df_bf['p_0.625;0.25;0.625_en1'][0], -2.3745, 4)
        self.assertAlmostEqual(df_bf['p_0.625;0.25;0.625_en4'][0], -8.1598, 4)
        self.assertTrue(math.isnan(df_bf['p_0.625;0.25;0.625_en5'][0]))
        # L:
        self.assertAlmostEqual(df_bf['n_0.5;0.5;0.5_en2'][0], 2.7381, 4)
        self.assertAlmostEqual(df_bf['n_0.5;0.5;0.5_en1'][0], 0.8534, 4)
        self.assertAlmostEqual(df_bf['p_0.5;0.5;0.5_en1'][0], -1.1779, 4)
        # W:
        self.assertAlmostEqual(df_bf['n_0.5;0.25;0.75_en1'][0], 3.6587, 4)

        # VBr2 with unoccupied Spin.down electrons for ib<ib_VBM but E>E_CBM:
        bs_featurizer = BandFeaturizer(kpoints=self.vbr2kpts, nbands=3)
        df_bf2 = bs_featurizer.featurize_dataframe(self.df2, col_id='bs_line')
        self.assertTrue(math.isnan(df_bf2['p_ex1_degen'][0]))
        # \\Gamma:
        self.assertAlmostEqual(df_bf2['n_0.0;0.0;0.0_en3'][0], 0.8020, 4)
        self.assertAlmostEqual(df_bf2['n_0.0;0.0;0.0_en2'][0], 0.4243, 4)
        self.assertAlmostEqual(df_bf2['n_0.0;0.0;0.0_en1'][0], 0.4243, 4)
        self.assertAlmostEqual(df_bf2['p_0.0;0.0;0.0_en1'][0], -0.3312, 4)
        self.assertAlmostEqual(df_bf2['p_0.0;0.0;0.0_en2'][0], -0.6076, 4)
        self.assertAlmostEqual(df_bf2['p_0.0;0.0;0.0_en3'][0], -0.6076, 4)
        # M:
        self.assertAlmostEqual(df_bf2['n_0.5;0.0;0.0_en3'][0], 0.5524, 4)
        self.assertAlmostEqual(df_bf2['n_0.5;0.0;0.0_en2'][0], 0.5074, 4)
        self.assertAlmostEqual(df_bf2['n_0.5;0.0;0.0_en1'][0], 0.2985, 4)
        self.assertAlmostEqual(df_bf2['p_0.5;0.0;0.0_en1'][0], -0.0636, 4)
        self.assertAlmostEqual(df_bf2['p_0.5;0.0;0.0_en2'][0], -0.1134, 4)
        self.assertAlmostEqual(df_bf2['p_0.5;0.0;0.0_en3'][0], -0.8091, 4)
        # between \\Gamma and M:
        self.assertAlmostEqual(df_bf2['n_0.2;0.0;0.0_en3'][0], 0.6250, 4)
        self.assertAlmostEqual(df_bf2['n_0.2;0.0;0.0_en2'][0], 0.3779, 4)
        self.assertAlmostEqual(df_bf2['n_0.2;0.0;0.0_en1'][0], 0.1349, 4)
        self.assertAlmostEqual(df_bf2['p_0.2;0.0;0.0_en1'][0], -0.1049, 4)
        self.assertAlmostEqual(df_bf2['p_0.2;0.0;0.0_en2'][0], -0.3044, 4)
        self.assertAlmostEqual(df_bf2['p_0.2;0.0;0.0_en3'][0], -0.6399, 4)
        # L:
        self.assertAlmostEqual(df_bf2['n_0.5;0.0;0.5_en2'][0], 0.4448, 4)
        self.assertAlmostEqual(df_bf2['n_0.5;0.0;0.5_en1'][0], 0.3076, 4)
        self.assertAlmostEqual(df_bf2['p_0.5;0.0;0.5_en1'][0], -0.0639, 4)
        self.assertAlmostEqual(df_bf2['p_0.5;0.0;0.5_en2'][0], -0.1133, 4)


class DOSFeaturesTest(PymatgenTest):

    def setUp(self):
        with open(os.path.join(test_dir, 'si_dos.json'), 'r') as sDOS:
            si_dos = CompleteDos.from_dict(json.load(sDOS))
        self.df = pd.DataFrame({'dos': [si_dos]})

    def test_DOSFeaturizer(self):
        df_df = DOSFeaturizer(contributors=2).featurize_dataframe(self.df, col_id=['dos'])
        df_df.to_csv('test_DOS_featurized.csv')
        # CBM:
        self.assertAlmostEqual(df_df['cbm_score_1'][0], 0.258, 3)
        self.assertAlmostEqual(df_df['cbm_score_2'][0], 0.258, 3)
        self.assertEqual(df_df['cbm_location_1'][0], '0.0;0.0;0.0')
        self.assertEqual(df_df['cbm_specie_1'][0], 'Si')
        self.assertEqual(df_df['cbm_character_1'][0], 's')
        self.assertEqual(df_df['cbm_coordination_1'][0], 'tet')
        self.assertEqual(df_df['cbm_nsignificant'][0], 4)
        # VBM:
        self.assertAlmostEqual(df_df['vbm_score_1'][0], 0.490, 3)
        self.assertAlmostEqual(df_df['vbm_score_2'][0], 0.490, 3)
        self.assertEqual(df_df['vbm_location_1'][0], '0.0;0.0;0.0')
        self.assertEqual(df_df['vbm_specie_1'][0], 'Si')
        self.assertEqual(df_df['vbm_character_1'][0], 'p')
        self.assertEqual(df_df['vbm_coordination_1'][0], 'tet')
        self.assertEqual(df_df['vbm_nsignificant'][0], 2)

if __name__ == '__main__':
    unittest.main()
