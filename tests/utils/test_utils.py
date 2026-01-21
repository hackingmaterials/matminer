import os
import unittest
from unittest import TestCase

import pandas as pd
from pymatgen.core import Composition

from matminer.utils.utils import get_elem_in_data, get_pseudo_inverse

test_dir = os.path.dirname(os.path.abspath(__file__))


class UtilsTest(TestCase):
    def setUp(self):
        data = pd.read_csv(os.path.join(test_dir, "utils_dataframe.csv"))
        data.set_index("Compound", inplace=True)
        comp = []
        for c in data.index:
            comp.append(Composition(c))
        data["Composition"] = comp
        self.data = data

    def test_get_elem_in_data(self):
        elem, elem_absent = get_elem_in_data(self.data, as_pure=True)
        self.assertListEqual(elem, ["Zn", "Au"])

        elem, elem_absent = get_elem_in_data(self.data, as_pure=False)
        self.assertListEqual(elem, ["N", "O", "Al", "Si", "S", "Zn", "Ga", "Ge", "As", "Y", "Ag", "Sn", "Au"])

    def test_get_pseudo_inverse(self):
        PI = get_pseudo_inverse(self.data, cols=["n_380.0"])
        self.assertAlmostEqual(PI["n_380.0"][0], -1.6345896211391995)
        self.assertAlmostEqual(PI["n_380.0"][4], 4.155467999999999)


if __name__ == "__main__":
    unittest.main()
