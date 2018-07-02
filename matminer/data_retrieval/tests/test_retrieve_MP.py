# coding: utf-8

from __future__ import division, unicode_literals

import unittest
from unittest import SkipTest

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, \
    BandStructure
from pymatgen.electronic_structure.dos import CompleteDos


class MPDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.mpdr = MPDataRetrieval()

    def test_get_data(self):
        if self.mpdr.mprester.api_key:
            df = self.mpdr.get_dataframe(criteria={"material_id": "mp-23"},
                                         properties=["structure",
                                                     "bandstructure",
                                                     "bandstructure_uniform",
                                                     "dos"])
            self.assertEqual(len(df["structure"]), 1)
            self.assertEqual(df["bandstructure"][0].get_band_gap()["energy"],0)
            self.assertTrue(isinstance(df["bandstructure"][0],
                                       BandStructureSymmLine))
            self.assertTrue(isinstance(df["bandstructure_uniform"][0],
                                       BandStructure))
            self.assertTrue(isinstance(df["dos"][0], CompleteDos))
        else:
            raise SkipTest("Skipped MPDataRetrieval test; no MAPI_KEY detected")


if __name__ == "__main__":
    unittest.main()
