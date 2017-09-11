# coding: utf-8

from __future__ import division, unicode_literals

import unittest
from unittest import SkipTest

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval


class MPDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.mpdr = MPDataRetrieval()

    def test_get_data(self):
        if self.mpdr.mprester.api_key:
            df = self.mpdr.get_dataframe(criteria={"material_id": "mp-23"},
                                         properties=["structure"])
            self.assertEqual(len(df["structure"]), 1)
        else:
            raise SkipTest("Skipped MPDataRetrieval test; no MAPI_KEY detected")


if __name__ == "__main__":
    unittest.main()
