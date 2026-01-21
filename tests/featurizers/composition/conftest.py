import unittest

import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Specie
from pymatgen.util.testing import PymatgenTest


class CompositionFeaturesTest(PymatgenTest):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "composition": [
                    Composition("Fe2O3"),
                    Composition({Specie("Fe", 2): 1, Specie("O", -2): 1}),
                ]
            }
        )

        self.df_nans = pd.DataFrame(
            {
                "composition": [
                    # Composition("U2Og3"),
                    Composition({Specie("U", 3): 2, Specie("Og", -2): 3}),
                ]
            }
        )


if __name__ == "__main__":
    unittest.main()
