# coding: utf-8

from __future__ import division, unicode_literals, absolute_import

import unittest2 as unittest
import os

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval

citrine_key = os.environ.get('CITRINE_KEY', None)


@unittest.skipIf(citrine_key is None,
                 "CITRINE_KEY environment variable not set.")
class CitrineDataRetrievalTest(unittest.TestCase):

    def setUp(self):
        self.cdr = CitrineDataRetrieval(citrine_key)

    def test_get_data(self):
        df = self.cdr.get_dataframe(formula="W", data_type='EXPERIMENTAL', max_results=10)

if __name__ == "__main__":
    unittest.main()
