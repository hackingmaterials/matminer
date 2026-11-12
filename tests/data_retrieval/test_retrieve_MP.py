import math
import unittest
from types import SimpleNamespace

import pytest
from pymatgen.core import SETTINGS, Lattice, Structure
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.dos import CompleteDos

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

from ..mp_api_mock import FakeMPRester, FakeMPRestError, fake_mp_api

SI = Structure(Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])


class _MPRester(FakeMPRester):
    docs = [
        SimpleNamespace(material_id="mp-149", structure=SI, band_gap=0.6),
        SimpleNamespace(material_id="mp-2", structure=SI, band_gap=0.0),
    ]

    def get_dos_by_material_id(self, material_id):
        if material_id == "mp-2":
            raise FakeMPRestError("no dos")
        return f"dos of {material_id}"


class MPDataRetrievalMPAPITest(unittest.TestCase):
    def test_uses_mp_api(self):
        with fake_mp_api(_MPRester):
            mpdr = MPDataRetrieval(api_key="key")
            self.assertIsInstance(mpdr.mprester, _MPRester)
            self.assertEqual(mpdr.mprester.api_key, "key")

            df = mpdr.get_dataframe(criteria={"material_id": "mp-149"}, properties=["structure", "band_gap", "dos"])
            self.assertEqual(_MPRester.search_kwargs["material_ids"], ["mp-149"])
            self.assertEqual(_MPRester.search_kwargs["fields"], ["structure", "band_gap", "material_id"])
            self.assertEqual(list(df.index), ["mp-149", "mp-2"])
            self.assertEqual(df.loc["mp-149", "structure"], SI)
            self.assertEqual(df.loc["mp-149", "dos"], "dos of mp-149")
            self.assertTrue(math.isnan(df.loc["mp-2", "dos"]))

            data = mpdr.get_data("Si", ["structure"], mp_decode=False, index_mpid=False)
            self.assertEqual(_MPRester.search_kwargs["formula"], "Si")
            self.assertEqual(data[0], {"structure": SI.as_dict()})

    def test_criteria_to_kwargs(self):
        self.assertEqual(MPDataRetrieval._criteria_to_kwargs("mp-1234"), {"material_ids": ["mp-1234"]})
        self.assertEqual(MPDataRetrieval._criteria_to_kwargs("Li-Fe-O"), {"chemsys": "Li-Fe-O"})
        self.assertEqual(MPDataRetrieval._criteria_to_kwargs("Fe2O3"), {"formula": "Fe2O3"})
        self.assertEqual(
            MPDataRetrieval._criteria_to_kwargs({"material_id": ["mp-1", "mp-2"], "band_gap": (1, None)}),
            {"material_ids": ["mp-1", "mp-2"], "band_gap": (1, None)},
        )


@pytest.mark.skipif(not SETTINGS.get("PMG_MAPI_KEY", ""), reason="PMG_MAPI_KEY not in environment variables.")
@pytest.mark.xfail(reason="Needs updates to use current MP API client - flakey test.", strict=False)
class MPDataRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.mpdr = MPDataRetrieval()

    def test_get_data(self):
        df = self.mpdr.get_dataframe(
            criteria={"material_id": "mp-23"},
            properties=["structure", "bandstructure", "bandstructure_uniform", "dos"],
        )
        self.assertEqual(len(df["structure"]), 1)
        self.assertEqual(df["bandstructure"][0].get_band_gap()["energy"], 0)
        self.assertTrue(isinstance(df["bandstructure"][0], BandStructureSymmLine))
        self.assertTrue(isinstance(df["bandstructure_uniform"][0], BandStructure))
        self.assertTrue(isinstance(df["dos"][0], CompleteDos))


if __name__ == "__main__":
    unittest.main()
