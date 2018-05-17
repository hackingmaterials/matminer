# coding: utf-8

from __future__ import division, unicode_literals, absolute_import
import unittest
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval, \
    make_dataframe
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class MDFDataRetrievalTest(unittest.TestCase):
    # There's a weird bug where invoking MDFDR in setUp
    # seems to screw up anonymous functionality, so it's
    # in setUpClass instead
    @classmethod
    def setUpClass(cls):
        cls.mdf_dr = MDFDataRetrieval(anonymous=True)
        cls.oqmd_version = cls.mdf_dr.forge.get_dataset_version('oqmd')

    def test_get_dataframe(self):
        results = self.mdf_dr.get_dataframe({
            "source_names": ['oqmd_v%d' % self.oqmd_version],
            "elements": ["Ag", "Be", "V"]},
            unwind_arrays=False)
        for elts in results['material.elements']:
            self.assertTrue("Be" in elts)
            self.assertTrue("Ag" in elts)
            self.assertTrue("V" in elts)

    def test_get_dataframe_by_query(self):
        qstring = "(mdf.source_name:oqmd_v{0}) AND "\
                  "(material.elements:Si AND material.elements:V AND "\
                  "oqmd_v{0}.band_gap.value:[0.5 TO *])".format(self.oqmd_version)
        mdf_df = self.mdf_dr.get_data(qstring, unwind_arrays=False)
        self.assertTrue((mdf_df['oqmd_v%d.band_gap.value'%self.oqmd_version] > 0.5).all())
        for elts in mdf_df['material.elements']:
            self.assertTrue("Si" in elts)
            self.assertTrue("V" in elts)

    def test_make_dataframe(self):
        raw = [{"material": {"elements": ["Ag", "Cr"]},
                "oqmd_v%d"%self.oqmd_version: {"band_gap": 0.5, "total_energy": 1.5}},
               {"material": {"elements": ["Ag", "Be"]},
                "oqmd_v%d"%self.oqmd_version: {"band_gap": 0.5, "total_energy": 1.5}}]
        df = make_dataframe(raw, )
        self.assertEqual(df['oqmd_v%d.band_gap'%self.oqmd_version][0], 0.5)


if __name__ == "__main__":
    unittest.main()
