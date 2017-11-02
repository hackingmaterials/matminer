from __future__ import unicode_literals, division, print_function

import json
import os
import pandas as pd
import unittest

from matminer.featurizers.bandstructure import BandFeaturizer, BranchPointEnergy
from pymatgen import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, \
    BandStructure
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

        self.df = pd.DataFrame({'bs_line': [si_bs_line], 'bs_uniform': [si_bs_uniform]})

    def test_BranchPointEnergy(self):
        df_bpe = BranchPointEnergy().featurize_dataframe(self.df,
                                                         col_id=['bs_uniform'])

        self.assertAlmostEqual(df_bpe['branch_point_energy'][0], 5.728, 3)
        self.assertAlmostEqual(df_bpe['cbm_absolute'][0], 0.497, 3)
        self.assertAlmostEqual(df_bpe['vbm_absolute'][0], -0.114, 3)

    def test_BandFeaturizer(self):
        df_bf = BandFeaturizer().featurize_dataframe(self.df, col_id='bs_line')
        self.assertAlmostEqual(df_bf['band_gap'][0], 0.612, 3)
        self.assertAlmostEqual(df_bf['direct_gap'][0], 2.557, 3)
        self.assertAlmostEqual(df_bf['n_ex1_norm'][0], 0.58413, 5)
        self.assertAlmostEqual(df_bf['p_ex1_norm'][0], 0.0, 5)
        self.assertEquals(df_bf['is_gap_direct'][0], False)
        self.assertEquals(df_bf['n_ex1_degen'][0], 6)
        self.assertEquals(df_bf['p_ex1_degen'][0], 1)

if __name__ == '__main__':
    unittest.main()
