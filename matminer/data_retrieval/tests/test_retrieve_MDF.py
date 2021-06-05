# coding: utf-8

import unittest
import os

from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval, make_dataframe


class MDFDataRetrievalTest(unittest.TestCase):
    # There's a weird bug where invoking MDFDR in setUp
    # seems to screw up anonymous functionality, so it's
    # in setUpClass instead
    @classmethod
    def setUpClass(cls):
        cls.mdf_dr = MDFDataRetrieval(anonymous=True)

    def test_get_dataframe(self):
        results = self.mdf_dr.get_dataframe(
            {"source_names": ["oqmd"], "elements": ["Ag", "Be", "V"]},
            unwind_arrays=False,
        )
        for elts in results["material.elements"]:
            self.assertTrue("Be" in elts)
            self.assertTrue("Ag" in elts)
            self.assertTrue("V" in elts)

    @unittest.skipIf(os.environ.get("CI", False), "Aggregations unable to run on CI")
    def test_get_dataframe_by_query(self):
        qstring = (
            "(mdf.source_name:oqmd) AND "
            "(material.elements:Si AND material.elements:V AND "
            "oqmd.band_gap.value:[0.5 TO *])"
        )
        mdf_df = self.mdf_dr.get_data(qstring, unwind_arrays=False)
        self.assertTrue((mdf_df["oqmd.band_gap.value"] > 0.5).all())
        for elts in mdf_df["material.elements"]:
            self.assertTrue("Si" in elts)
            self.assertTrue("V" in elts)

    def test_make_dataframe(self):
        raw = [
            {
                "material": {"elements": ["Ag", "Cr"]},
                "oqmd": {"band_gap": 0.5, "total_energy": 1.5},
            },
            {
                "material": {"elements": ["Ag", "Be"]},
                "oqmd": {"band_gap": 0.5, "total_energy": 1.5},
            },
        ]
        df = make_dataframe(
            raw,
        )
        self.assertEqual(df["oqmd.band_gap"][0], 0.5)


if __name__ == "__main__":
    unittest.main()
