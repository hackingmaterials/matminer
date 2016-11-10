# coding: utf-8

from __future__ import division, unicode_literals

import unittest2 as unittest
import os

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval

api_key = os.environ.get('CITRINE_KEY', None)

@unittest.skipIf(api_key is None,
                 "CITRINE_KEY environment variable not set.")

class MPResterTest(unittest.TestCase):

    def setUp(self):
        self.cdr = CitrineDataRetrieval(api_key)

    def test_get_data(self):
        df = self.cdr.get_dataframe(contributor="OQMD", formula="GaN")


if __name__=="__main__":
    unittest.main()
