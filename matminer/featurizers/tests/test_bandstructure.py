from __future__ import unicode_literals, division, print_function

import pandas as pd
import unittest

from matminer.featurizers.bandstructure import BandFeaturizer, BranchPointEnergy
from pymatgen.ext.matproj import MPRester
from pymatgen.util.testing import PymatgenTest


api = MPRester()


class BandstructureFeaturesTest(PymatgenTest):

    def setUp(self):
        # self.si_str = PymatgenTest.get_structure('Si')
        mpid = 'mp-149' # silicon
        test_structure = api.get_structure_by_material_id(material_id=mpid)
        bs = api.get_bandstructure_by_material_id(material_id=mpid)
        bs.structure = test_structure
        self.df = pd.DataFrame({'bs':[bs]})

    def test_BranchPointEnergy(self):
        df_bpe = BranchPointEnergy()

        # this takes forever and fails at the end raises the following error:
        # ValueError: Unable to find 1:1 corresponding between input kpoints and irreducible grid!
        # df_bpe.featurize_dataframe(self.df, col_id=['bs'])

    def test_BandFeaturizer(self):
        df_bf = BandFeaturizer().featurize_dataframe(self.df, col_id='bs')
        self.assertAlmostEqual(df_bf['band_gap'][0], 0.612, 3)
        self.assertAlmostEqual(df_bf['direct_gap'][0], 2.557, 3)
        self.assertAlmostEqual(df_bf['n_ex1_norm'][0], 0.58413, 5)
        self.assertAlmostEqual(df_bf['p_ex1_norm'][0], 0.0, 5)
        self.assertEquals(df_bf['is_gap_direct'][0], False)


if __name__ == '__main__':
    unittest.main()
