# coding: utf-8

from __future__ import division, unicode_literals, absolute_import
import os
import unittest
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval, \
    flatten_nested_dict
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

    def test_flatten_nested_dict(self):
        """
        # test basic functionality
        test1 = {"a": {"b": 1, "c": 2}}
        flattened = flatten_nested_dict(test1)
        self.assertEqual(flattened["a.b"], 1)
        self.assertEqual(flattened["a.c"], 2)

        # test array functionality
        flattened = flatten_nested_dict(test2)
        self.assertEqual(flattened["a.b"], (0, 1, 2))
        """

        test2 = {"a": {"b": (0, 1, 2), "c": 2}}
        flattened = flatten_nested_dict(test2, unwind_arrays=True)
        self.assertEqual(flattened["a.b.0"], 0)
        self.assertEqual(flattened["a.b.2"], 2)

if __name__ == "__main__":
    unittest.main()
