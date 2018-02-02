# coding: utf-8

from __future__ import division, unicode_literals, absolute_import
import os
import unittest
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class MDFDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.mdfdr = MDFDataRetrieval(anonymous=True)

    def test_get_match(self):
        match = self.mdfdr.generate_match(['mdf.source_name'],
                                          ['fe_cr_al_oxidation'])
        self.assertEqual(match.current_query(),
                         '(mdf.source_name:fe_cr_al_oxidation)')
        match = self.mdfdr.generate_match(['mdf.source_name', 'oqmd'],
                                          ['fe_cr_al_oxidation', '1'])
        blargh

    def test_search(self):
        match = self.mdfdr.generate_match(['mdf.source_name'],
                                          ['fe_cr_al_oxidation'])
        data = match.search()

if __name__ == "__main__":
    unittest.main()
