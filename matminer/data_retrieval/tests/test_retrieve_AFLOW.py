import unittest
import os

import numpy as np
from pymatgen.core.structure import Structure

from matminer.data_retrieval.retrieve_AFLOW import AFLOWDataRetrieval
from matminer.data_retrieval.tests.base import on_ci


@unittest.skipIf(on_ci.upper() == "TRUE", "Bad AFLOW-GHActions pipeline")
class AFLOWDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.aflowdr = AFLOWDataRetrieval()

    def test_get_data(self):
        df = self.aflowdr.get_dataframe(
            criteria={"auid": {"$in": ["aflow:a17a2da2f3d3953a"]}},
            properties=["density", "enthalpy_formation_atom", "positions_fractional"],
            files=["structure"],
        )

        # ensures that only one result is returned for a single auid
        self.assertEqual(len(df["aurl"]), 1)

        # ensures that type-casting is working correctly
        self.assertTrue(isinstance(df["aurl"][0], str))
        self.assertTrue(isinstance(df["density"][0], float))
        self.assertTrue(isinstance(df["positions_fractional"][0], np.ndarray))

        # ensures that auid is set as the index
        self.assertTrue(df.index.values[0] == "aflow:a17a2da2f3d3953a")

        # ensures that structures are downloaded
        self.assertTrue(isinstance(df["structure"][0], Structure))


if __name__ == "__main__":
    unittest.main()
