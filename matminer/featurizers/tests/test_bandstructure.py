from __future__ import unicode_literals, division, print_function

import json
import os
import pandas as pd
import unittest

from matminer.featurizers.bandstructure import BandFeaturizer, BranchPointEnergy
from pymatgen import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.util.testing import PymatgenTest

test_dir = os.path.join(os.path.dirname(__file__))

class BandstructureFeaturesTest(PymatgenTest):

    def setUp(self):
        with open(os.path.join(test_dir, 'si_structure.json'),'r') as sth:
            si_str = Structure.from_dict(json.load(sth))
        with open(os.path.join(test_dir, 'si_bandstructure.json'),'r') as bsh:
            si_bs = BandStructure.from_dict(json.load(bsh))
        si_bs.structure = si_str
        self.df = pd.DataFrame({'bs': [si_bs]})

    def test_BranchPointEnergy(self):
        df_bpe = BranchPointEnergy()

        # TODO: @computron not sure what's going on here, need to fix! -computron

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
