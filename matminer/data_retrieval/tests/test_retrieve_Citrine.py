# coding: utf-8

from __future__ import division, unicode_literals, absolute_import
import os
import unittest
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

citrine_key = os.environ.get('CITRINE_KEY', None)


@unittest.skipIf(citrine_key is None, "CITRINE_KEY environment variable not set.")
class CitrineDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.cdr = CitrineDataRetrieval(citrine_key)

    def test_get_data(self):
        pifs_lst = self.cdr.get_api_data(formula="W", data_type='EXPERIMENTAL', max_results=10)
        df = self.cdr.get_dataframe(pifs_lst)
        assert df.shape[0] == 10

    def test_mutiple_items_in_list(self):
        pifs_lst = self.cdr.get_api_data(data_set_id=114192)
        df = self.cdr.get_dataframe(pifs_lst)
        for col in ["Thermal conductivity_5-conditions", "Condition_1", "Thermal conductivity_10"]:
            assert col in df.columns


if __name__ == "__main__":
    unittest.main()
