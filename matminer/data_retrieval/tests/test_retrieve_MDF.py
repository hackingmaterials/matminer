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

    def test_search(self):
        results = self.mdf_dr.search(sources=['oqmd'],
                                     elements=["Ag", "Be", "V"],
                                     unwind_arrays=False)
        for elts in results['mdf.elements']:
            self.assertTrue("Be" in elts)
            self.assertTrue("Ag" in elts)
            self.assertTrue("V" in elts)


    def test_search_by_query(self):
        qstring = "(mdf.source_name:oqmd) AND "\
                  "(mdf.elements:Si AND mdf.elements:V AND "\
                  "oqmd.band_gap.value:[0.5 TO *])"
        results = self.mdf_dr.search_by_query(qstring, unwind_arrays=False)
        self.assertTrue(all(results['oqmd.band_gap.value'] > 0.5))
        for elts in results['mdf.elements']:
            self.assertTrue("Si" in elts)
            self.assertTrue("V" in elts)


    def test_make_dataframe(self):
        raw = [{"mdf": {"elements": ["Ag", "Cr"]},
                "oqmd": {"band_gap": 0.5, "total_energy": 1.5}},
               {"mdf": {"elements": ["Ag", "Be"]},
                "oqmd": {"band_gap": 0.5, "total_energy": 1.5}}]
        df = make_dataframe(raw, )
        self.assertEqual(df['oqmd.band_gap'][0], 0.5)


if __name__ == "__main__":
    unittest.main()
