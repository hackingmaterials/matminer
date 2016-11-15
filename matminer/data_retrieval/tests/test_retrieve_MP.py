# coding: utf-8

from __future__ import division, unicode_literals

import unittest2 as unittest
import os

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

mapi_key = os.environ.get('MAPI_KEY', None)


@unittest.skipIf(mapi_key is None,
                 "MAPI_KEY environment variable not set.")
class MPDataRetrievalTest(unittest.TestCase):

    def setUp(self):
        self.mpdr = MPDataRetrieval(mapi_key)

    def test_get_data(self):
        df = self.mpdr.get_dataframe(criteria={"material_id": "mp-23"}, properties=["structure"])

if __name__ == "__main__":
    unittest.main()
