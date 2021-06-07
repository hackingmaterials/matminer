# coding: utf-8

import os
import unittest

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
from matminer.data_retrieval.tests.base import on_ci

citrine_key = os.environ.get("CITRINATION_API_KEY", None)


@unittest.skipIf(on_ci.upper() == "TRUE", "Bad Citrination-GHActions pipeline")
@unittest.skipIf(citrine_key is None, "CITRINATION_API_KEY env variable not set.")
class CitrineDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.cdr = CitrineDataRetrieval(citrine_key)

    def test_get_data(self):
        pifs_lst = self.cdr.get_data(formula="W", data_type="EXPERIMENTAL", max_results=10)
        self.assertEqual(len(pifs_lst), 10)
        df = self.cdr.get_dataframe(
            criteria={"formula": "W", "data_type": "EXPERIMENTAL", "max_results": 10},
            print_properties_options=False,
        )
        self.assertEqual(df.shape[0], 10)

    def test_multiple_items_in_list(self):
        df = self.cdr.get_dataframe(
            criteria={"data_set_id": 114192, "max_results": 102},
            print_properties_options=False,
        )
        self.assertEqual(df.shape[0], 102)
        test_cols = {
            "Thermal conductivity_5-conditions",
            "Condition_1",
            "Thermal conductivity_10",
        }
        self.assertTrue(test_cols < set(df.columns))


if __name__ == "__main__":
    unittest.main()
